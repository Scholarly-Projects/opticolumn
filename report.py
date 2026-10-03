#!/usr/bin/env python3
"""
Opticolumn OCR searchability report generator.
"""

import csv
import datetime
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # PyMuPDF
from spellchecker import SpellChecker

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_DIR  = "A"        # Original PDFs
OUTPUT_DIR = "B"        # OCR-processed PDFs
REPORT_DIR = "D"        # CSV output destination

TOOL_NAME    = "Opticolumn"
TOOL_VERSION = "2026"

# Local names, places and acronyms that are real words for this collection
# but not in the English dictionary (one per line, any case).  Optional.
LEXICON_PATH = "lexicon.txt"

# Unknown capitalised tokens (names) shorter than this are never accepted.
# (Names always need supporting context, so short ones like "Ada" are safe.)
MIN_PROPER_NOUN_LEN = 3
# Unknown ALL-CAPS tokens longer than this are not accepted as acronyms
# ("ASUI", "NCAA" pass; "UNAVERSEE", "RECEIPTURE" do not).
MAX_ACRONYM_LEN = 5
# An unknown name counts anywhere if it occurs at least this often in the file.
PROPER_NOUN_MIN_OCCURRENCES = 2
# 3-letter words need at least this pyspellchecker frequency ("ere" 3073,
# "rev" 1175 pass; dictionary filler sits at 50).
MIN_FREQ_3 = 500
# Line context: a line with at least LINE_MIN_TOKENS alphabetic tokens of 3+
# letters, of which fewer than LINE_MIN_REAL are dictionary words, is garbage.
LINE_MIN_TOKENS = 3
LINE_MIN_REAL   = 0.50
MAX_CONSONANT_RUN = 4

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ── CSV fields ─────────────────────────────────────────────────────────────────
# One row per matched A/B file pair, plus a trailing TOTAL row for the batch.
CSV_FIELDS = [
    "tool_name",
    "tool_version",
    "report_datetime",
    "file_name",
    "word_count_A",
    "word_count_B",
    "percent_searchability",
    "tokens_A",
    "tokens_B",
    "true_word_rate_A",
    "true_word_rate_B",
    "garbage_lines_A",
    "garbage_lines_B",
]

# ── Word lists ─────────────────────────────────────────────────────────────────
_spell = SpellChecker()              # loaded once (dictionary load is expensive)
_freq = _spell.word_frequency

SHORT_WORDS = {                      # the only 1–2 letter tokens that count
    "a", "i", "am", "an", "as", "at", "be", "by", "do", "go", "he", "if", "in",
    "is", "it", "me", "my", "no", "of", "oh", "on", "or", "so", "to", "up", "us",
    "we", "ox",
}
ABBREVIATIONS = {                    # case-insensitive, matched without periods
    "mr", "mrs", "ms", "dr", "jr", "sr", "st", "rev", "prof", "co", "inc", "pm",
    "am", "no", "vs", "ft", "lb", "lbs", "oz", "gen", "gov", "sen", "rep", "capt",
    "lt", "col", "sgt", "ave", "blvd", "dept", "univ",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}
DOTTED_LOWER = {"a.m", "p.m", "i.e", "e.g", "etc"}
LOWER_ABBREVIATIONS = {"pm", "am", "ft", "lb", "lbs", "oz", "vs", "etc", "sq", "yd", "yds", "mi"}


def _load_lexicon(path: str) -> set:
    try:
        with open(path, encoding="utf-8") as fh:
            words = {ln.strip().lower() for ln in fh if ln.strip() and not ln.startswith("#")}
        logger.info(f"Local lexicon: {len(words)} word(s) from {path}")
        return words
    except OSError:
        return set()


LEXICON = _load_lexicon(LEXICON_PATH)

# ── Compiled patterns ──────────────────────────────────────────────────────────
_EDGE_PUNCT = re.compile(r"^[^\w]+|[^\w]+$")
_WORD_RE = re.compile(r"^[A-Za-z]+(?:['’\-][A-Za-z]+)*$")
_NUM_RE = re.compile(
    r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$"            # 1,000  12,500.50
    r"|^\d+(?:\.\d+)?$"                          # 1900  3.50
    r"|^\d{1,4}(?:st|nd|rd|th|s)$"               # 15th  22nd  1920s
    r"|^\d{1,4}(?:[/\-]\d{1,4}){1,2}$"           # 12/25/1941  30-28  1950-51
    r"|^\d{1,2}:\d{2}$",                         # 7:30
    re.IGNORECASE,
)
_DOTTED_RE = re.compile(r"^(?:[A-Za-z]\.){1,5}[A-Za-z]$")   # W.S.C  A.S.U.I  p.m
_ROMAN_RE = re.compile(r"^(?=[MDCLXVI])M{0,3}(?:C[MD]|D?C{0,3})(?:X[CL]|L?X{0,3})(?:I[XV]|V?I{0,3})$")
_TRIPLE_RE = re.compile(r"([A-Za-z])\1\1")
_VOWEL_RE = re.compile(r"[aeiouy]", re.IGNORECASE)
_CONSONANT_RUN_RE = re.compile(r"[b-df-hj-np-tv-xz]{%d,}" % (MAX_CONSONANT_RUN + 1), re.IGNORECASE)


# ── Token helpers ──────────────────────────────────────────────────────────────

def _clean_token(token: str) -> str:
    """Strip leading/trailing punctuation, preserving internal structure."""
    return _EDGE_PUNCT.sub("", token)


def _has_normal_case(token: str) -> bool:
    """lower, ALL-CAPS or Title-case — not "soMe", "IJetters", "reGardin"."""
    alpha = [c for c in token if c.isalpha()]
    if not alpha:
        return True
    all_lower  = all(c.islower() for c in alpha)
    all_upper  = all(c.isupper() for c in alpha)
    title_case = alpha[0].isupper() and all(c.islower() for c in alpha[1:])
    return all_lower or all_upper or title_case


def _plausible_shape(word: str) -> bool:
    """Letter patterns real English words and names have."""
    return (bool(_VOWEL_RE.search(word))
            and not _TRIPLE_RE.search(word)
            and not _CONSONANT_RUN_RE.search(word))


def _is_dictionary_word(part: str) -> bool:
    """One letters-only part (no hyphen) is a real English or local word."""
    w = part.lower()
    if w in LEXICON:
        return True
    if len(w) <= 2:
        return w in SHORT_WORDS
    if w not in _spell:
        return False
    if len(w) == 3:
        return _freq[w] >= MIN_FREQ_3 or w in ABBREVIATIONS
    return True


HYPHEN_PREFIXES = {"co", "re", "pre", "non", "ex", "self", "anti", "semi", "vice", "mid", "post", "pro"}


def _is_known_word(token: str) -> bool:
    """A word token (possibly hyphenated / with an apostrophe) whose parts are all real words."""
    t = token.replace("\u2019", "'")
    if "'" not in t and "-" not in t:
        return _is_dictionary_word(t)
    if _is_dictionary_word(t.replace("'", "").replace("-", "")):      # can't, lead-er, sec-retary
        return True
    base = re.sub(r"'(s|d|ll|re|ve|t|m)$", "", t, flags=re.IGNORECASE)  # possessives, contractions
    if "-" not in base and len(base) > 3 and base.lower() in _spell:  # o'clock
        return True
    parts = [p for p in re.split(r"[-']", base) if p]
    return bool(parts) and all(_is_dictionary_word(p) or (i == 0 and p.lower() in HYPHEN_PREFIXES)
                               for i, p in enumerate(parts))


def _rule_sliver(token: str) -> bool:
    """"Iowing", "Ibandages", "lmonth": a column-rule stroke read as I/l, glued to a word."""
    return (len(token) >= 5 and token[0] in "Il" and token[1:2].islower()
            and _is_dictionary_word(token[1:]) and len(token) - 1 >= 4)


# ── Token classification ───────────────────────────────────────────────────────
# KNOWN    — counts wherever it appears on a non-garbage line
# CANDIDATE— an unknown name/acronym: counts only with support (context, recurrence, lexicon)
# REJECT   — never counts

# ACRONYM  — an unknown ALL-CAPS token: counts only if it recurs or is in the lexicon
#            (headline lines are mostly real words, so line context proves little)
KNOWN, CANDIDATE, ACRONYM, REJECT = "known", "candidate", "acronym", "reject"


def classify(raw_token: str) -> Tuple[str, str]:
    """(class, cleaned token)."""
    token = _clean_token(raw_token)
    if not token:
        return REJECT, token
    if re.fullmatch(r"[A-Z]\.", raw_token.strip("\"'(),;:")):
        return CANDIDATE, token                      # an initial ("J. R. Simplot"): needs a real-text line

    if _NUM_RE.match(token):
        digits = re.sub(r"\D", "", token)
        if set(digits) == {"0"} or (len(digits) >= 4 and len(set(digits)) == 1):
            return REJECT, token                     # "00", "0.00", "1111"
        return KNOWN, token

    if "-" in token and _NUM_RE.match(token.split("-")[0]) and _WORD_RE.match(token.split("-", 1)[1]) \
            and _is_known_word(token.split("-", 1)[1]):
        return KNOWN, token                          # 2,000-degree  19-year-old

    if _DOTTED_RE.match(token):
        if token.isupper() or token.lower() in DOTTED_LOWER:
            return KNOWN, token                      # W.S.C.  A.S.U.I.  p.m.
        return REJECT, token                         # "p.r.s", "b.e" — fragments

    if not _WORD_RE.match(token) or not _has_normal_case(token):
        return REJECT, token

    if token.lower() in ABBREVIATIONS and (token[0].isupper() or token in LOWER_ABBREVIATIONS):
        return KNOWN, token                          # Mr  Mrs  Dr  Rev  Aug
    if _is_known_word(token):
        return KNOWN, token
    if len(token) >= 2 and token.isupper() and _ROMAN_RE.match(token):
        return KNOWN, token                          # II  III  XIV

    # Not in any dictionary: could still be a name or an acronym.
    letters = token.replace("'", "").replace("’", "").replace("-", "")
    if token[0].islower() or not _plausible_shape(letters) or _rule_sliver(token):
        return REJECT, token
    if token.isupper() and len(letters) > 1:
        return (ACRONYM if len(letters) <= MAX_ACRONYM_LEN else REJECT), token
    return (CANDIDATE if len(letters) >= MIN_PROPER_NOUN_LEN else REJECT), token


# ── PDF word count ─────────────────────────────────────────────────────────────

def _lines(doc: "fitz.Document") -> List[List[str]]:
    """
    Text-layer words grouped into their printed lines (PyMuPDF block/line
    numbers), with words hyphenated across a line break joined back
    together ("prac-" + "tically" → "practically") so they can be checked.
    """
    lines: List[List[str]] = []
    for page in doc:
        cur_key, cur = None, []
        for w in page.get_text("words"):
            key = (w[5], w[6])                       # block_no, line_no
            if key != cur_key and cur:
                lines.append(cur)
                cur = []
            cur_key = key
            cur.append(w[4])
        if cur:
            lines.append(cur)
    for i in range(len(lines) - 1):
        a, b = lines[i], lines[i + 1]
        if a and b and a[-1].endswith("-") and len(a[-1]) > 2 and b[0][:1].islower():
            a[-1] = a[-1][:-1] + b[0]
            b.pop(0)
    return [ln for ln in lines if ln]


def count_words(pdf_path: Path) -> Dict[str, int]:
    """
    {"words": true words, "tokens": all tokens, "garbage_lines": lines rejected}.

    Works for born-digital PDFs, PDFs with an invisible OCR text layer, and
    files with NO text layer at all (all zeros), so an un-OCR'd file in A/
    is still comparable to its OCR'd counterpart in B/.
    """
    try:
        with fitz.open(str(pdf_path)) as doc:
            lines = _lines(doc)
    except Exception as e:
        logger.error(f"Could not read text layer from '{pdf_path.name}': {e}")
        return {"words": 0, "tokens": 0, "garbage_lines": 0}

    classified = [[classify(t) for t in ln] for ln in lines]
    recurring = Counter(tok for ln in classified for c, tok in ln if c in (CANDIDATE, ACRONYM))

    words = tokens = garbage = 0
    for ln in classified:
        tokens += len(ln)
        alpha = [(c, tok) for c, tok in ln if len(tok) >= 3 and tok.replace("'", "").replace("-", "").isalpha()]
        known_alpha = sum(1 for c, _ in alpha if c == KNOWN)
        if len(alpha) >= LINE_MIN_TOKENS and known_alpha / len(alpha) < LINE_MIN_REAL:
            garbage += 1                              # chance hits on a garbage line don't count
            continue
        line_reads_as_text = known_alpha >= 2 or (alpha and known_alpha / len(alpha) >= LINE_MIN_REAL)
        for c, tok in ln:
            if c == KNOWN:
                words += 1
            elif c == CANDIDATE and (line_reads_as_text or tok.lower() in LEXICON
                                     or recurring[tok] >= PROPER_NOUN_MIN_OCCURRENCES):
                words += 1
            elif c == ACRONYM and (tok.lower() in LEXICON
                                   or recurring[tok] >= PROPER_NOUN_MIN_OCCURRENCES):
                words += 1
    return {"words": words, "tokens": tokens, "garbage_lines": garbage}


def percent_searchability(words_a: int, words_b: int) -> float:
    """Percent change in true-word count from A to B (file rows and TOTAL row alike)."""
    if words_a > 0:
        return ((words_b - words_a) / words_a) * 100
    # A had no searchable text at all (e.g. image-only, never OCR'd) —
    # B's entire count is new coverage, not a "percent increase" of zero.
    return 100.0 if words_b > 0 else 0.0


def _rate(words: int, tokens: int) -> str:
    return f"{100 * words / tokens:.1f}%" if tokens else "n/a"


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    a_folder = Path(INPUT_DIR)
    b_folder = Path(OUTPUT_DIR)
    d_folder = Path(REPORT_DIR)

    for folder in (a_folder, b_folder):
        if not folder.exists():
            logger.error(f"Required folder '{folder}' not found.")
            sys.exit(1)

    d_folder.mkdir(parents=True, exist_ok=True)

    a_pdfs = sorted(a_folder.glob("*.pdf"))
    if not a_pdfs:
        logger.error(f"No PDF files found in '{INPUT_DIR}'.")
        sys.exit(1)

    now = datetime.datetime.now()
    report_datetime_str = now.strftime("%Y-%m-%d %H:%M:%S")

    report_rows = []
    tot = {"A": Counter(), "B": Counter()}
    files_processed = 0

    for a_pdf in a_pdfs:
        # Matched purely by filename, so a file with NO text layer in A/ is
        # still compared against its OCR'd version in B/.
        b_pdf = b_folder / a_pdf.name
        if not b_pdf.exists():
            logger.warning(f"No processed counterpart for '{a_pdf.name}' in B/ — skipping.")
            continue

        ca = count_words(a_pdf)   # all zeros if A/ has no text layer at all
        cb = count_words(b_pdf)
        file_pct = percent_searchability(ca["words"], cb["words"])

        logger.info(
            f"{a_pdf.name}: A={ca['words']:,} true words of {ca['tokens']:,} tokens "
            f"({_rate(ca['words'], ca['tokens'])}, {ca['garbage_lines']} garbage lines)  "
            f"B={cb['words']:,} of {cb['tokens']:,} ({_rate(cb['words'], cb['tokens'])}, "
            f"{cb['garbage_lines']} garbage lines)  ({file_pct:+.2f}%)"
        )

        report_rows.append({
            "tool_name":             TOOL_NAME,
            "tool_version":          TOOL_VERSION,
            "report_datetime":       report_datetime_str,
            "file_name":             a_pdf.name,
            "word_count_A":          ca["words"],
            "word_count_B":          cb["words"],
            "percent_searchability": f"{file_pct:.2f}%",
            "tokens_A":              ca["tokens"],
            "tokens_B":              cb["tokens"],
            "true_word_rate_A":      _rate(ca["words"], ca["tokens"]),
            "true_word_rate_B":      _rate(cb["words"], cb["tokens"]),
            "garbage_lines_A":       ca["garbage_lines"],
            "garbage_lines_B":       cb["garbage_lines"],
        })
        tot["A"].update(ca)
        tot["B"].update(cb)
        files_processed += 1

    if files_processed == 0:
        logger.error(
            "No matched PDF pairs found between A/ and B/. "
            "Ensure B/ contains files with the same names as A/."
        )
        sys.exit(1)

    a, b = tot["A"], tot["B"]
    pct = percent_searchability(a["words"], b["words"])
    report_rows.append({
        "tool_name":             TOOL_NAME,
        "tool_version":          TOOL_VERSION,
        "report_datetime":       report_datetime_str,
        "file_name":             f"TOTAL ({files_processed} files)",
        "word_count_A":          a["words"],
        "word_count_B":          b["words"],
        "percent_searchability": f"{pct:.2f}%",
        "tokens_A":              a["tokens"],
        "tokens_B":              b["tokens"],
        "true_word_rate_A":      _rate(a["words"], a["tokens"]),
        "true_word_rate_B":      _rate(b["words"], b["tokens"]),
        "garbage_lines_A":       a["garbage_lines"],
        "garbage_lines_B":       b["garbage_lines"],
    })

    csv_path = d_folder / f"opticolumn_report_{now.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(report_rows)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Report written       → {csv_path}")
    logger.info(f"Files processed      : {files_processed}")
    logger.info(f"True words  A        : {a['words']:,} of {a['tokens']:,} tokens ({_rate(a['words'], a['tokens'])})")
    logger.info(f"True words  B        : {b['words']:,} of {b['tokens']:,} tokens ({_rate(b['words'], b['tokens'])})")
    logger.info(f"Searchability change : {pct:.2f}%")


if __name__ == "__main__":
    main()
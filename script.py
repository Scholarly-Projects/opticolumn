#!/usr/bin/env python3
"""
Opticolumn 
"""

import sys
import os
import tempfile
from pathlib import Path
import fitz  
from kraken import blla
from kraken.lib.vgsl import TorchVGSLModel
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import re
import platform
import datetime
import shutil
import pikepdf
import hashlib
import uuid
from xml.sax.saxutils import escape as xml_escape

# ---------------- Configuration ----------------
INPUT_DIR  = "A"
OUTPUT_DIR = "B"
MODELS_DIR = "mlmodels"
POPPLER_PATH = None
DPI = 200
TROCR_MODELS = {
    "handwritten":       "microsoft/trocr-base-handwritten",
    "printed":           "microsoft/trocr-base-printed",
    "large_handwritten": "microsoft/trocr-large-handwritten",
    "large_printed":     "microsoft/trocr-large-printed",
}
TROCR_MODEL_NAME               = TROCR_MODELS["large_handwritten"]
ENABLE_PREPROCESSING           = True   
CONFIDENCE_THRESHOLD           = 0.25
SINGLE_CHAR_CONFIDENCE_THRESHOLD = 0.5
MIN_SEGMENT_HEIGHT             = 10
FONT_NAME  = "helv"                  
FONT_PATH  = "fonts/FreeSans.ttf"   
SRGB_ICC_PATH = "srgb.icc"
DEBUG_OCR_LAYER         = False
DEBUG_TEXT_POSITIONS    = False
DEBUG_SAVE_INTERMEDIATE = False

# ---- Branding / attribution / document metadata ----
# Written into every output PDF (document properties + XMP) so a derivative
# can always be traced to the tool, its author, the run that produced it and
# the exact source file.
APP_NAME        = "Opticolumn"
APP_VERSION     = "2026"
APP_SCRIPT      = Path(__file__).name
APP_URL         = "https://github.com/Scholarly-Projects/opticolumn"
APP_AUTHOR      = "Andrew Weymouth"
APP_INSTITUTION = "University of Idaho"       # the author's affiliation
APP_LICENSE     = "MIT"
APP_CREATOR     = f"{APP_NAME} {APP_VERSION} ({APP_URL})"
APP_CITATION    = f"{APP_AUTHOR}. {APP_NAME} ({APP_VERSION}). {APP_INSTITUTION}. {APP_URL}"
DOC_SUBJECT     = "OCR processed document"
DOC_KEYWORDS    = "OCR; historic newspaper; searchable PDF; Opticolumn; Kraken; TrOCR"
DOC_LANGUAGE    = "en-US"
OPT_NAMESPACE   = "https://github.com/Scholarly-Projects/opticolumn/ns/"
KEEP_SOURCE_TITLE_AUTHOR = True   # keep the input PDF's own Title/Author when it has them
SEGMENTATION_MODEL = "Kraken blla (blla.mlmodel)"

# One id per run: every PDF written by the same invocation shares it.
RUN_ID = str(uuid.uuid4())

# ---------------- Logging Setup ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ── Date helpers ──────────────────────────────────────────────────────────────
def get_pdf_date_string(dt=None):
    if dt is None:
        dt = datetime.datetime.now()
    return dt.strftime("D:%Y%m%d%H%M%S")

def get_xmp_date_string(dt=None):
    if dt is None:
        dt = datetime.datetime.now()
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


# ---------------- Font and ICC Profile Setup ----------------
def setup_pdfa_resources():
    try:
        font_dir = Path("fonts")
        font_dir.mkdir(exist_ok=True)
        font_path = Path(FONT_PATH)
        if not font_path.exists():
            logger.info("Downloading FreeSans font for embedding...")
            import urllib.request
            urllib.request.urlretrieve(
                "https://github.com/opensourcedesign/fonts/raw/master/gnu-freefont_freesans/FreeSans.ttf",
                str(font_path),
            )
        srgb_path = Path(SRGB_ICC_PATH)
        if not srgb_path.exists():
            logger.info("Downloading sRGB ICC profile...")
            try:
                import urllib.request
                if platform.system() == "Darwin":
                    system_profile = "/System/Library/ColorSync/Profiles/sRGB Profile.icc"
                elif platform.system() == "Windows":
                    system_profile = os.path.join(
                        os.environ.get("WINDIR", "C:\\Windows"),
                        "System32", "spool", "drivers", "color",
                        "sRGB Color Space Profile.icm",
                    )
                elif platform.system() == "Linux":
                    system_profile = "/usr/share/color/icc/sRGB.icc"
                else:
                    system_profile = None
                if system_profile and Path(system_profile).exists():
                    shutil.copy2(system_profile, str(srgb_path))
                else:
                    urllib.request.urlretrieve("https://www.color.org/srgb.xalter", str(srgb_path))
            except Exception as e:
                logger.warning(f"Could not obtain sRGB ICC profile: {e}")
        return True
    except Exception as e:
        logger.error(f"Failed to setup PDF/A resources: {e}")
        return False


# ---------------- XMP Metadata ----------------
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

_OPT_PROPERTIES = [
    ("ToolName",          "Text",    "Name of the tool that produced the OCR text layer"),
    ("Version",           "Text",    "Version of the tool that produced the OCR text layer"),
    ("Script",            "Text",    "Script file that produced the OCR text layer"),
    ("ProjectURL",        "Text",    "Project site of the tool"),
    ("ToolAuthor",        "Text",    "Author of the tool"),
    ("ToolAffiliation",   "Text",    "Institutional affiliation of the tool author"),
    ("ToolLicense",       "Text",    "Licence of the tool"),
    ("Citation",          "Text",    "Recommended citation for the tool"),
    ("RunID",             "Text",    "Identifier of the processing run that produced this file"),
    ("SourceFile",        "Text",    "File name of the source PDF"),
    ("SourceSHA256",      "Text",    "SHA-256 checksum of the source PDF"),
    ("RenderDPI",         "Integer", "Resolution at which pages were rendered for OCR"),
    ("SegmentationModel", "Text",    "Line segmentation model used for OCR"),
    ("OCRModel",          "Text",    "Text recognition model used for OCR"),
    ("PagesTotal",        "Integer", "Pages in the document"),
]


def create_xmp_metadata(title, author, subject, keywords, creator, producer,
                        creation_date, modify_date, language=DOC_LANGUAGE,
                        opt_values: Optional[Dict[str, str]] = None, source_file: str = "",
                        doc_id: str = "", instance_id: str = "", event_params: str = ""):
    """
    XMP packet for PDF/A-1b.  Standard schemas carry what any repository
    system reads (title, creator tool, keywords, source, identifiers, a
    processing-history event); the custom opt: schema carries attribution
    and run provenance, each property declared in the PDF/A extension
    schema.  Title/Author/Subject/Keywords/Creator/Producer mirror the
    document Info dictionary exactly, as PDF/A requires.
    """
    try:
        e = lambda v: xml_escape(str(v))
        ns = e(OPT_NAMESPACE)
        values = opt_values or {}
        creator_xml = (f"\n      <dc:creator><rdf:Seq><rdf:li>{e(author)}</rdf:li></rdf:Seq></dc:creator>"
                       if author else "")
        source_xml = f"\n      <dc:source>{e(source_file)}</dc:source>" if source_file else ""
        opt_xml = "\n".join(f"      <opt:{n}>{e(values[n])}</opt:{n}>"
                            for n, _t, _d in _OPT_PROPERTIES if values.get(n) not in (None, ""))
        decl_xml = "\n".join(f"""                <rdf:li rdf:parseType="Resource">
                  <pdfaProperty:name>{n}</pdfaProperty:name>
                  <pdfaProperty:valueType>{t}</pdfaProperty:valueType>
                  <pdfaProperty:category>internal</pdfaProperty:category>
                  <pdfaProperty:description>{e(d)}</pdfaProperty:description>
                </rdf:li>""" for n, t, d in _OPT_PROPERTIES)
        return f"""<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about="" xmlns:pdf="http://ns.adobe.com/pdf/1.3/">
      <pdf:Producer>{e(producer)}</pdf:Producer>
      <pdf:Keywords>{e(keywords)}</pdf:Keywords>
    </rdf:Description>
    <rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">
      <dc:format>application/pdf</dc:format>
      <dc:title><rdf:Alt><rdf:li xml:lang="x-default">{e(title)}</rdf:li></rdf:Alt></dc:title>{creator_xml}
      <dc:description><rdf:Alt><rdf:li xml:lang="x-default">{e(subject)}</rdf:li></rdf:Alt></dc:description>
      <dc:language><rdf:Bag><rdf:li>{e(language)}</rdf:li></rdf:Bag></dc:language>{source_xml}
    </rdf:Description>
    <rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/">
      <xmp:CreatorTool>{e(creator)}</xmp:CreatorTool>
      <xmp:CreateDate>{creation_date}</xmp:CreateDate>
      <xmp:ModifyDate>{modify_date}</xmp:ModifyDate>
      <xmp:MetadataDate>{modify_date}</xmp:MetadataDate>
    </rdf:Description>
    <rdf:Description rdf:about=""
        xmlns:xmpMM="http://ns.adobe.com/xap/1.0/mm/"
        xmlns:stEvt="http://ns.adobe.com/xap/1.0/sType/ResourceEvent#">
      <xmpMM:DocumentID>{e(doc_id)}</xmpMM:DocumentID>
      <xmpMM:InstanceID>{e(instance_id)}</xmpMM:InstanceID>
      <xmpMM:History>
        <rdf:Seq>
          <rdf:li rdf:parseType="Resource">
            <stEvt:action>converted</stEvt:action>
            <stEvt:instanceID>{e(instance_id)}</stEvt:instanceID>
            <stEvt:parameters>{e(event_params)}</stEvt:parameters>
            <stEvt:softwareAgent>{e(creator)}</stEvt:softwareAgent>
            <stEvt:when>{modify_date}</stEvt:when>
          </rdf:li>
        </rdf:Seq>
      </xmpMM:History>
    </rdf:Description>
    <rdf:Description rdf:about="" xmlns:pdfaid="http://www.aiim.org/pdfa/ns/id/">
      <pdfaid:part>1</pdfaid:part>
      <pdfaid:conformance>B</pdfaid:conformance>
    </rdf:Description>
    <rdf:Description rdf:about="" xmlns:opt="{ns}">
{opt_xml}
    </rdf:Description>
    <rdf:Description rdf:about=""
        xmlns:pdfaExtension="http://www.aiim.org/pdfa/ns/extension/"
        xmlns:pdfaSchema="http://www.aiim.org/pdfa/ns/schema#"
        xmlns:pdfaProperty="http://www.aiim.org/pdfa/ns/property#">
      <pdfaExtension:schemas>
        <rdf:Bag>
          <rdf:li rdf:parseType="Resource">
            <pdfaSchema:schema>{e(APP_NAME)} processing and attribution metadata</pdfaSchema:schema>
            <pdfaSchema:namespaceURI>{ns}</pdfaSchema:namespaceURI>
            <pdfaSchema:prefix>opt</pdfaSchema:prefix>
            <pdfaSchema:property>
              <rdf:Seq>
{decl_xml}
              </rdf:Seq>
            </pdfaSchema:property>
          </rdf:li>
        </rdf:Bag>
      </pdfaExtension:schemas>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""
    except Exception as e:
        logger.error(f"Failed to create XMP metadata: {e}")
        return None


def stamp_document_metadata(doc: fitz.Document, filename: str, source_path: Optional[str] = None) -> None:
    """
    Document Info dictionary + XMP + viewer prefs.

    Title and Author describe the newspaper, not the tool: the source PDF's
    own Title/Author are kept when present (KEEP_SOURCE_TITLE_AUTHOR),
    otherwise Title is the file name and Author is left empty.  The tool,
    its author, the run and the source checksum are recorded in Creator,
    Keywords and the XMP opt: properties.
    """
    now = datetime.datetime.now()
    creation_date = get_pdf_date_string(now)
    xmp_date = get_xmp_date_string(now)
    producer = "PyMuPDF"
    src = doc.metadata or {}
    title = (src.get("title") or "").strip() if KEEP_SOURCE_TITLE_AUTHOR else ""
    author = (src.get("author") or "").strip() if KEEP_SOURCE_TITLE_AUTHOR else ""
    title = title or filename
    doc_id = f"uuid:{uuid.uuid4()}"
    instance_id = f"uuid:{uuid.uuid4()}"

    sha = ""
    if source_path:
        try:
            sha = _sha256(source_path)
        except OSError as exc:
            logger.warning(f"Could not checksum {source_path}: {exc}")
    opt_values = {
        "ToolName": APP_NAME, "Version": APP_VERSION, "Script": APP_SCRIPT,
        "ProjectURL": APP_URL, "ToolAuthor": APP_AUTHOR, "ToolAffiliation": APP_INSTITUTION,
        "ToolLicense": APP_LICENSE, "Citation": APP_CITATION, "RunID": RUN_ID,
        "SourceFile": filename, "SourceSHA256": sha,
        "RenderDPI": str(DPI), "SegmentationModel": SEGMENTATION_MODEL,
        "OCRModel": TROCR_MODEL_NAME, "PagesTotal": str(len(doc)),
    }
    params = f"{APP_SCRIPT}; DPI {DPI}; {SEGMENTATION_MODEL}; OCR {TROCR_MODEL_NAME}; run {RUN_ID}"

    doc.set_metadata({
        "title":        title,
        "author":       author,
        "subject":      DOC_SUBJECT,
        "keywords":     DOC_KEYWORDS,
        "creator":      APP_CREATOR,
        "producer":     producer,
        "creationDate": creation_date,
        "modDate":      creation_date,
    })
    xmp = create_xmp_metadata(
        title=title,
        author=author,
        subject=DOC_SUBJECT,
        keywords=DOC_KEYWORDS,
        creator=APP_CREATOR,
        producer=producer,
        creation_date=xmp_date,
        modify_date=xmp_date,
        language=DOC_LANGUAGE,
        opt_values=opt_values,
        source_file=filename,
        doc_id=doc_id,
        instance_id=instance_id,
        event_params=params,
    )
    if xmp:
        doc.set_xml_metadata(xmp)
    cat = doc.pdf_catalog()
    doc.xref_set_key(cat, "ViewerPreferences", "<</DisplayDocTitle true>>")
    doc.xref_set_key(cat, "Lang", f"({DOC_LANGUAGE})")
    logger.info(f"Metadata: {APP_CREATOR}, run {RUN_ID}" + (f", source SHA-256 {sha}" if sha else ""))


# ---------------- Model Loading ----------------
def load_models():
    try:
        if not setup_pdfa_resources():
            logger.warning("PDF/A resources setup incomplete.")
        seg_model_path = Path(MODELS_DIR) / "blla.mlmodel"
        logger.info(f"Loading segmentation model: {seg_model_path}")
        seg_model = TorchVGSLModel.load_model(str(seg_model_path))
        seg_model.eval()
        logger.info(f"Loading TrOCR model: {TROCR_MODEL_NAME}")
        processor   = TrOCRProcessor.from_pretrained(TROCR_MODEL_NAME)
        trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_NAME)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trocr_model.to(device)
        logger.info(f"Using device: {device}")
        return seg_model, processor, trocr_model
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

try:
    seg_model, processor, trocr_model = load_models()
except Exception as e:
    logger.error("Model loading failed. Exiting.")
    sys.exit(1)


# ---------------- Image Preprocessing (OCR copy only) ----------------
def preprocess_for_ocr(pil_image: Image.Image) -> Image.Image:
    """
    Return a preprocessed COPY of pil_image suitable for the segmentation
    model and TrOCR.  The original pil_image is never modified and is NOT
    stored in the output PDF.
    """
    if not ENABLE_PREPROCESSING:
        return pil_image.copy()
    try:
        gray = pil_image.convert("L")
        gray = ImageOps.autocontrast(gray, cutoff=2)
        processed = gray.convert("RGB")
        processed = processed.filter(ImageFilter.SHARPEN)
        return processed
    except Exception as e:
        logger.error(f"Error preprocessing image for OCR: {e}")
        return pil_image.copy()


# ── Render a PDF page to a PIL image (original quality, used for the PDF) ──
def page_to_pil(page: fitz.Page, dpi: int = DPI) -> Image.Image:
    """Render *page* at *dpi* and return an RGB PIL Image."""
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


# ---------------- Text Recognition with TrOCR ----------------
def recognize_text_with_trocr(image: Image.Image, processor, model) -> tuple[str, float]:
    try:
        pixel_values = processor(image, return_tensors="pt").pixel_values
        device = next(model.parameters()).device
        pixel_values = pixel_values.to(device)
        with torch.no_grad():
            out = model.generate(
                pixel_values, output_scores=True, return_dict_in_generate=True
            )
            generated_text = processor.batch_decode(
                out.sequences, skip_special_tokens=True
            )[0]
            if out.scores:
                probs     = [torch.softmax(s, dim=-1) for s in out.scores]
                max_probs = [torch.max(p).item() for p in probs]
                confidence = sum(max_probs) / len(max_probs)
            else:
                confidence = 0.0
        return generated_text.strip(), confidence
    except Exception as e:
        logger.error(f"Error recognising text with TrOCR: {e}")
        return "", 0.0


# ---------------- Noise Detection ----------------
def is_likely_noise(text: str, confidence: float, seg_h: int, seg_w: int) -> bool:
    if not text:
        return True
    if seg_h < MIN_SEGMENT_HEIGHT or seg_w < 15:
        return True
    ar = seg_w / seg_h
    if ar < 0.1 or ar > 100:
        return True
    tc = text.strip()
    tl = len(tc)
    if tl == 1:
        return confidence < SINGLE_CHAR_CONFIDENCE_THRESHOLD
    if confidence < CONFIDENCE_THRESHOLD:
        return True
    if len(set(tc)) == 1 and tl > 2:
        return True
    noise_patterns = [r"^[oOlI\.\|]+$", r"^[0-9\.\,]+$", r"^[^a-zA-Z0-9\s]+$"]
    for pat in noise_patterns:
        if re.match(pat, tc) and confidence < SINGLE_CHAR_CONFIDENCE_THRESHOLD:
            return True
    if tl > 3 and not any(c.lower() in "aeiou" for c in tc) and confidence < 0.7:
        return True
    return False


# ---------------- Column Detection and Sorting ----------------
def geometric_column_sort(lines: List) -> List:
    """
    Pure-geometry fallback column sorter using a horizontal projection
    histogram.  Used only when Kraken has not provided reading-order
    metadata.  Detects column gaps as valleys in the histogram, assigns
    each line to a column by its centre-x, then sorts left→right across
    columns and top→bottom within each column.
    """
    if len(lines) <= 1:
        return lines

    bboxes = []
    for line in lines:
        if hasattr(line, "boundary") and len(line.boundary) >= 3:
            xs = [p[0] for p in line.boundary]
            ys = [p[1] for p in line.boundary]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        elif hasattr(line, "bbox"):
            x0, y0, x1, y1 = line.bbox
        else:
            continue
        bboxes.append((x0, y0, x1, y1, line))

    if not bboxes:
        return lines

    page_width = max(b[2] for b in bboxes)
    resolution = 10
    hist_w = int(page_width / resolution) + 1
    hist   = [0] * hist_w
    for x0, y0, x1, y1, _ in bboxes:
        for bi in range(int(x0 / resolution), min(int(x1 / resolution) + 1, hist_w)):
            hist[bi] += (y1 - y0)

    smoothed = hist.copy()
    for i in range(1, len(hist) - 1):
        smoothed[i] = (hist[i - 1] + 2 * hist[i] + hist[i + 1]) / 4

    valleys = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] < smoothed[i - 1] and smoothed[i] < smoothed[i + 1]:
            nbr_max = max(smoothed[i - 1], smoothed[i + 1])
            if nbr_max > 0 and smoothed[i] / nbr_max < 0.3:
                valleys.append(i * resolution)

    if not valleys:
        widths = [x1 - x0 for x0, y0, x1, y1, _ in bboxes]
        avg_w  = sum(widths) / len(widths) if widths else 100
        n_cols = max(1, min(int(page_width / (avg_w * 1.5)), 5))
        col_w  = page_width / n_cols
        valleys = [int((i + 1) * col_w) for i in range(n_cols - 1)]

    valleys  = sorted(v for v in valleys if 0 < v < page_width)
    columns  = [[] for _ in range(len(valleys) + 1)]
    for box in bboxes:
        x0, y0, x1, y1, line = box
        cx    = (x0 + x1) / 2
        col_i = sum(1 for v in valleys if cx > v)
        columns[col_i].append(box)

    sorted_lines = []
    for col in columns:
        for box in sorted(col, key=lambda b: b[1]):
            sorted_lines.append(box[4])
    return sorted_lines


def order_lines(segmentation) -> List:
    """
    Return segmentation lines in the best available reading order.

    Strategy:
      1. Kraken blla already runs its own reading-order algorithm before
         returning (the "Compute reading order / topological sort" log lines
         come from inside blla).  `segmentation.lines` is therefore already
         in reading order — use it directly whenever it has content.
      2. Only fall back to geometric_column_sort when the segmentation
         result is empty or carries no lines at all.

    This prevents our histogram sort from undoing the ordering Kraken spent
    time computing, which was the root cause of column mis-ordering on
    multi-column newspaper pages.
    """
    lines = getattr(segmentation, "lines", [])
    if not lines:
        return []

    # Kraken has provided an ordered list — trust it.
    logger.debug(f"Using Kraken reading order for {len(lines)} lines.")
    return list(lines)


# ---------------- OCR Text Element Extraction ----------------
def create_ocr_text_elements(
    pil_images: List[Image.Image],
    filename: str,
) -> List[List[dict]]:
    """
    Run segmentation + TrOCR on each PIL image (which may be a preprocessed
    copy).  Returns per-page lists of dicts with pixel-space coordinates.

    Keys: x0, x1, y_baseline, font_size, text
    x1 is stored so the insertion step can clamp text width to the segment
    boundary, preventing OCR text from bleeding past the right edge of the
    page.
    """
    font_path = Path(FONT_PATH)
    if not font_path.exists():
        raise FileNotFoundError(f"Required font {font_path} is missing.")

    all_pages: List[List[dict]] = []
    total_elements = 0

    for idx, pil_image in enumerate(pil_images):
        page_num = idx + 1
        logger.info(f"Processing page {page_num}/{len(pil_images)} of {filename}")
        page_elements: List[dict] = []
        filtered = 0

        try:
            ocr_image  = preprocess_for_ocr(pil_image)   # OCR copy — preprocessed
            # pil_image is the original render; ocr_image is used only here
            segmentation = blla.segment(ocr_image, model=seg_model)
            logger.info(f"Found {len(segmentation.lines)} text lines on page {page_num}")

            if not segmentation.lines:
                logger.warning("No text lines detected. Saving debug image...")
                debug_dir = Path("debug_images")
                debug_dir.mkdir(exist_ok=True)
                ocr_image.save(debug_dir / f"{filename}_page{page_num}_preprocessed.png")

            sorted_lines = order_lines(segmentation)
            logger.info(f"Sorted {len(sorted_lines)} lines into reading order.")

            for i, line in enumerate(sorted_lines):
                try:
                    if hasattr(line, "boundary") and len(line.boundary) >= 3:
                        xs = [p[0] for p in line.boundary]
                        ys = [p[1] for p in line.boundary]
                        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                    elif hasattr(line, "bbox"):
                        x0, y0, x1, y1 = line.bbox
                    else:
                        continue

                    sh, sw = y1 - y0, x1 - x0
                    if sh < 5 or sw < 5:
                        filtered += 1
                        continue

                    # Crop from the OCR copy (same pixel space)
                    line_img = ocr_image.crop((x0, y0, x1, y1))
                    text, confidence = recognize_text_with_trocr(
                        line_img, processor, trocr_model
                    )

                    if is_likely_noise(text, confidence, sh, sw):
                        filtered += 1
                        continue

                    page_elements.append({
                        "x0":         x0,
                        "x1":         x1,   # segment right edge (pixel space)
                        "y_baseline": y1,
                        "font_size":  max(6, min(sh * 0.9, 72)),
                        "text":       text,
                    })

                except Exception as e:
                    logger.error(f"Error on text line {i + 1}: {e}")

            logger.info(f"Page {page_num}: {len(page_elements)} elements, {filtered} filtered.")
            total_elements += len(page_elements)

        except Exception as e:
            logger.error(f"OCR failed for page {page_num}: {e}")

        all_pages.append(page_elements)

    logger.info(
        f"OCR extraction complete: {total_elements} total text elements "
        f"across {len(pil_images)} pages"
    )
    return all_pages


# ---------------- PDF/A Compliance ----------------
def setup_pdfa_compliance(pdf_path: str):
    """Embed sRGB OutputIntent into an already-saved PDF using pikepdf."""
    try:
        srgb_path = Path(SRGB_ICC_PATH)
        if not srgb_path.exists():
            logger.error("sRGB ICC profile not found; skipping PDF/A OutputIntent.")
            return
        with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
            # Use explicit '/Key' string syntax — safest across pikepdf versions.
            if "/OutputIntents" not in pdf.Root:
                pdf.Root["/OutputIntents"] = pikepdf.Array()

            # Build the ICC profile stream
            icc_data = srgb_path.read_bytes()
            icc_stream = pdf.make_stream(icc_data)
            icc_stream.stream_dict["/N"]         = pikepdf.Integer(3)
            icc_stream.stream_dict["/Alternate"] = pikepdf.Name("/DeviceRGB")

            # Build the OutputIntent dictionary
            output_intent = pikepdf.Dictionary({
                "/Type":                     pikepdf.Name("/OutputIntent"),
                "/S":                        pikepdf.Name("/GTS_PDFA1"),
                "/Info":                     pikepdf.String("sRGB IEC61966-2.1"),
                "/OutputConditionIdentifier": pikepdf.String("sRGB"),
                "/DestOutputProfile":         pdf.make_indirect(icc_stream),
            })
            pdf.Root["/OutputIntents"].append(pdf.make_indirect(output_intent))
            pdf.save(pdf_path)
        logger.info("PDF/A OutputIntent embedded successfully.")
    except Exception as e:
        logger.error(f"Failed to set up PDF/A compliance: {e}")


# ---------------- PDF Processing (OCR) ----------------
def process_single_pdf_ocr(input_path: str, output_path: str) -> bool:
    """
    Add an invisible OCR text layer to *input_path* and write the result to
    *output_path*.

    Approach (no flatten step):
      1. Open the ORIGINAL PDF with fitz.
      2. Render each page to a PIL image (in memory) for OCR only.
      3. Run segmentation + TrOCR on a preprocessed copy of each render.
      4. Insert invisible text elements directly into the original page.
      5. Save once with deflate compression.

    Because the original image streams are never re-encoded, the output file
    is typically only 3-8% larger than the input.
    """
    filename = os.path.basename(input_path)
    logger.info(f"Starting OCR for: {filename}")

    try:
        with fitz.open(input_path) as doc:
            # ── Render all pages to PIL images for OCR ──────────────────────
            logger.info(f"Rendering {len(doc)} pages at {DPI} DPI for OCR…")
            pil_images: List[Image.Image] = []
            for page in doc:
                pil_images.append(page_to_pil(page, dpi=DPI))

            # ── Run OCR on rendered images ───────────────────────────────────
            ocr_pages = create_ocr_text_elements(pil_images, filename)

            # ── Set document metadata ────────────────────────────────────────
            stamp_document_metadata(doc, filename, input_path)

            # ── Insert invisible text into original pages ────────────────────
            page_count = min(len(doc), len(ocr_pages))
            logger.info(f"Inserting text into {page_count} pages…")

            for page_num in range(page_count):
                page     = doc[page_num]
                elements = ocr_pages[page_num]
                pil_img  = pil_images[page_num]

                # ── Strip any pre-existing text layer ────────────────────────
                # Scans that were previously OCR'd (or partially searchable)
                # would otherwise produce a doubled text layer.  We cover the
                # entire page with a redaction annotation and apply it with
                # PDF_REDACT_IMAGE_NONE so that image streams are untouched —
                # only content-stream text operators are removed.
                existing_text = page.get_text().strip()
                if existing_text:
                    logger.info(
                        f"Page {page_num+1}: removing existing text layer "
                        f"({len(existing_text)} chars) before inserting OCR."
                    )
                    page.add_redact_annot(page.rect)
                    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

                # Scale: pixel-space → PDF point-space
                img_w, img_h = pil_img.size
                page_w = page.rect.width
                page_h = page.rect.height
                sx     = page_w / img_w
                sy     = page_h / img_h

                logger.debug(
                    f"Page {page_num+1}: {img_w}×{img_h}px → "
                    f"{page_w:.1f}×{page_h:.1f}pt  (sx={sx:.4f}, sy={sy:.4f})"
                )

                inserted = 0
                for elem in elements:
                    try:
                        x0_pt       = elem["x0"] * sx
                        x1_pt       = elem["x1"] * sx
                        baseline_pt = elem["y_baseline"] * sy
                        fontsize    = max(4, elem["font_size"] * sy)

                        # ── Clamp font size to segment width ─────────────────
                        # The segment right edge (x1_pt) is the authoritative
                        # boundary for this line of text.  If the rendered
                        # string would exceed that boundary, scale the font
                        # down proportionally so no characters are clipped.
                        # We also cap at page_w to guard against any segment
                        # that itself extends marginally past the crop box.
                        available_w = min(x1_pt, page_w) - x0_pt
                        if available_w > 0:
                            text_w = fitz.get_text_length(
                                elem["text"], fontname=FONT_NAME, fontsize=fontsize
                            )
                            if text_w > available_w:
                                fontsize = max(4, fontsize * available_w / text_w)

                        page.insert_text(
                            fitz.Point(x0_pt, baseline_pt),
                            elem["text"],
                            fontsize=fontsize,
                            fontname=FONT_NAME,
                            render_mode=3,      
                            color=(0, 0, 0),
                        )
                        inserted += 1
                    except Exception as e:
                        logger.error(f"Failed to insert text on page {page_num+1}: {e}")

                logger.info(f"Page {page_num+1}: inserted {inserted}/{len(elements)} text elements")

            # ── Save with deflate; do NOT re-encode images ───────────────────
            doc.save(
                output_path,
                deflate=True,          # compress streams (text, metadata, etc.)
                garbage=4,             # remove unused objects
                clean=True,
                deflate_images=False,  # leave original image streams untouched
                encryption=fitz.PDF_ENCRYPT_KEEP,
            )
            logger.info(f"OCR-enhanced PDF saved: {output_path}")

        # ── PDF/A compliance — MUST be called after the file is on disk ──────
        srgb_path = Path(SRGB_ICC_PATH)
        if srgb_path.exists():
            logger.info("Applying PDF/A OutputIntent…")
            setup_pdfa_compliance(output_path)
        else:
            logger.warning("sRGB ICC profile not found; PDF/A OutputIntent skipped.")

        # ── Verify OCR layer ─────────────────────────────────────────────────
        logger.info("Verifying OCR layer in final output…")
        try:
            with fitz.open(output_path) as final_pdf:
                total_chars = 0
                for i, pg in enumerate(final_pdf):
                    chars = len(pg.get_text().strip())
                    total_chars += chars
                    logger.info(f"Final PDF page {i+1} extractable text length: {chars}")
                if total_chars > 0:
                    logger.info(f"SUCCESS: Final PDF contains {total_chars} characters of searchable text")
                else:
                    logger.error("PROBLEM: Final PDF has no extractable text!")
        except Exception as e:
            logger.error(f"Failed to verify final output: {e}")

        return True

    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------- Compression (Size Targeting) ----------------
def compress_to_target_size(input_pdf: Path, output_pdf: Path, original_size: int) -> Path:
    """
    Try to keep the output within 15% of the original size using
    PDF-native deflate compression only.  Because the original image
    streams are never re-encoded, this target is almost always achievable
    without touching images at all.

    No JPEG recompression is attempted — that causes generation loss and
    is incompatible with archival quality requirements.
    """
    max_target = int(original_size * 1.15)
    logger.info(
        f"Targeting maximum size: {max_target // 1024} KB "
        f"(15% increase from original {original_size // 1024} KB)"
    )

    current_size = input_pdf.stat().st_size
    logger.info(f"OCR file size before compression: {current_size // 1024} KB")

    if current_size <= max_target:
        shutil.copy2(input_pdf, output_pdf)
        logger.info("File already within target size. No additional compression needed.")
        return output_pdf

    # Progressive deflate options — images left untouched throughout
    compression_options = [
        {"deflate": True, "garbage": 4, "clean": True, "deflate_images": False},
        {"deflate": True, "garbage": 3, "clean": True, "deflate_images": False},
        {"deflate": True, "garbage": 2, "clean": True, "deflate_images": False},
    ]

    for i, opts in enumerate(compression_options):
        temp_out = output_pdf.with_suffix(f".temp_{i}.pdf")
        try:
            with fitz.open(str(input_pdf)) as doc:
                doc.save(str(temp_out), **opts, encryption=fitz.PDF_ENCRYPT_KEEP)

            compressed_size = temp_out.stat().st_size
            pct = (compressed_size - original_size) / original_size * 100
            logger.info(f"Compression option {i+1}: {compressed_size // 1024} KB ({pct:+.1f}% from original)")

            if compressed_size <= max_target:
                # Verify OCR was not lost
                try:
                    with fitz.open(str(temp_out)) as chk:
                        total_chars = sum(len(pg.get_text().strip()) for pg in chk)
                except Exception:
                    total_chars = -1

                if total_chars > 0:
                    shutil.move(str(temp_out), str(output_pdf))
                    logger.info(f"Compression option {i+1} accepted; OCR preserved ({total_chars} chars).")
                    return output_pdf
                else:
                    logger.error("OCR lost after compression attempt — falling back to uncompressed OCR file.")
                    temp_out.unlink(missing_ok=True)
                    shutil.copy2(input_pdf, output_pdf)
                    return output_pdf
            else:
                temp_out.unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Compression option {i+1} failed: {e}")
            temp_out.unlink(missing_ok=True)

    logger.warning(
        "All deflate options exceeded the 15% size budget. "
        "Returning OCR file as-is (images untouched, quality preserved). "
        "Consider reducing DPI or using a lighter segmentation model if "
        "strict size limits are mandatory."
    )
    shutil.copy2(input_pdf, output_pdf)
    return output_pdf


# ---------------- Main ----------------
def main():
    logger.info(f"{APP_NAME} {APP_VERSION} ({APP_SCRIPT}) by {APP_AUTHOR}, {APP_INSTITUTION} — "
                f"{APP_LICENSE} licence — {APP_URL}")
    logger.info(f"Run {RUN_ID}")
    input_folder  = Path(INPUT_DIR)
    output_folder = Path(OUTPUT_DIR)

    if not input_folder.exists():
        logger.error(f"Input folder '{INPUT_DIR}' not found.")
        sys.exit(1)
    output_folder.mkdir(exist_ok=True)

    pdf_files = list(input_folder.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files in '{INPUT_DIR}'")
        sys.exit(1)

    # Identify already-processed files and skip them
    skipped = []
    pending = []
    for pdf_path in pdf_files:
        final_path = output_folder / f"{pdf_path.stem}.pdf"
        if final_path.exists():
            skipped.append(pdf_path.name)
        else:
            pending.append(pdf_path)

    if skipped:
        logger.info(f"Skipping {len(skipped)} already-processed file(s): {', '.join(skipped)}")

    if not pending:
        logger.info("All files have already been processed. Nothing to do.")
        return

    logger.info(f"Processing {len(pending)} file(s) with TrOCR: {TROCR_MODEL_NAME}")
    logger.info("Target: Final size ≤ original + 15% (OCR text layer only; images untouched)")

    for pdf_path in pending:
        original_size = pdf_path.stat().st_size
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing {pdf_path.name} | Original: {original_size // 1024} KB")

        ocr_temp_path = output_folder / f"{pdf_path.stem}_ocr_temp.pdf"

        if not process_single_pdf_ocr(str(pdf_path), str(ocr_temp_path)):
            logger.error(f"Skipping {pdf_path.name} due to OCR failure.")
            continue

        final_path  = output_folder / f"{pdf_path.stem}.pdf"
        result_path = compress_to_target_size(ocr_temp_path, final_path, original_size)

        if result_path.exists():
            final_size    = result_path.stat().st_size
            size_increase = (final_size - original_size) / original_size * 100
            logger.info(
                f"SUCCESS: {result_path.name} | {final_size // 1024} KB "
                f"({size_increase:+.1f}% increase from original)"
            )
        else:
            logger.error(f"Failed to generate final output for {pdf_path.name}")

        try:
            ocr_temp_path.unlink()
        except Exception as e:
            logger.warning(f"Could not delete temp file: {e}")

    logger.info(f"\nAll done! Output files in '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
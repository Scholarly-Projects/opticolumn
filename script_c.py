#!/usr/bin/env python3
"""
opticolumn book scan
"""

import sys
import os
import gc
import math
from dataclasses import dataclass, field
from pathlib import Path
import fitz
from kraken import blla
from kraken.lib.vgsl import TorchVGSLModel
from PIL import Image, ImageDraw, ImageFilter, ImageOps
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
DEBUG_SAVE_INTERMEDIATE = False   # save each preprocessed (split + deskewed) page to debug_images/

# ---- Two-page book spreads ----
SPREAD_MODE            = "auto"        # "auto": split pages wider than SPREAD_MIN_ASPECT; "always"; "never"
SPREAD_MIN_ASPECT      = 1.15          # width / height at or above which a page is treated as a spread
GUTTER_SEARCH_BAND     = (0.35, 0.65)  # fraction of the spread width searched for the spine
SINGLE_PAGE_SPINE_SIDE = "auto"        # unsplit pages: "auto" (darker edge), "left", "right" or None

# ---- Per-page deskew ----
DESKEW         = True
MAX_SKEW_DEG   = 5.0    # search range, each direction
MIN_SKEW_DEG   = 0.15   # smaller angles are left alone (avoids a needless resample)
SKEW_MIN_GAIN  = 0.03   # best angle must beat 0° by 3% on the projection score to be used

# ---- Spine-side cleanup ----
FLATTEN_ILLUMINATION  = True   # remove gutter shadow / uneven lighting before OCR
SPINE_ZONE_FRAC       = 0.25   # fraction of each page's width (measured from the spine) that gets extra help
SPINE_SHARPEN_PERCENT = 180    # unsharp-mask strength at the spine edge, tapering to 0 across the zone
SPINE_SHARPEN_RADIUS  = 2.0
SPINE_CONF_RELAX      = 0.20   # confidence thresholds lowered by up to 20% at the spine edge

# ---- Line crops ----
LINE_POLYGON_MASK = True   # whiten everything outside the line's polygon before TrOCR
LINE_PAD_PX       = 4      # padding around each line crop (pixels, OCR space)
SKIP_EXISTING_OUTPUT = True  # skip a PDF whose output already exists (checked before models load)
TROCR_BATCH_SIZE     = 8     # line crops per TrOCR generate() call; lower it if memory is tight
USE_MPS              = True  # Apple-silicon GPU if available (falls back to CPU on error)

# ---- Branding / attribution / document metadata ----
APP_NAME        = "Opticolumn"
APP_VERSION     = "2026"
APP_SCRIPT      = Path(__file__).name
APP_URL         = "https://github.com/Scholarly-Projects/opticolumn"
APP_AUTHOR      = "Andrew Weymouth"
APP_INSTITUTION = "University of Idaho"
APP_LICENSE     = "MIT"
APP_CREATOR     = f"{APP_NAME} {APP_VERSION} ({APP_URL})"
APP_CITATION    = f"{APP_AUTHOR}. {APP_NAME} ({APP_VERSION}). {APP_INSTITUTION}. {APP_URL}"
DOC_SUBJECT     = "OCR processed book scan"
DOC_KEYWORDS    = "OCR; book scan; two-page spread; searchable PDF; Opticolumn; Kraken; TrOCR"
DOC_LANGUAGE    = "en-US"
OPT_NAMESPACE   = "https://github.com/Scholarly-Projects/opticolumn/ns/"
KEEP_SOURCE_TITLE_AUTHOR = True
SEGMENTATION_MODEL = "Kraken blla (blla.mlmodel)"
PAGE_LAYOUT = (
    f"spreads {SPREAD_MODE} (aspect >= {SPREAD_MIN_ASPECT}); "
    f"deskew {'per page, max ' + str(MAX_SKEW_DEG) + ' deg' if DESKEW else 'off'}; "
    f"illumination flattening {'on' if FLATTEN_ILLUMINATION else 'off'}; "
    f"spine sharpening {SPINE_SHARPEN_PERCENT}% over {SPINE_ZONE_FRAC:.0%} of page width"
)

RUN_ID = str(uuid.uuid4())

# Set at start-up by _calibrate_morph_sign(): which sign of fitz.Matrix(angle)
# tilts inserted text downward-right in page coordinates.
MORPH_SIGN = 1

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
    ("PageLayout",        "Text",    "Spread splitting, deskew and spine preprocessing applied before OCR"),
    ("PagesTotal",        "Integer", "Pages in the document"),
]


def create_xmp_metadata(title, author, subject, keywords, creator, producer,
                        creation_date, modify_date, language=DOC_LANGUAGE,
                        opt_values: Optional[Dict[str, str]] = None, source_file: str = "",
                        doc_id: str = "", instance_id: str = "", event_params: str = ""):
    """
    XMP packet for PDF/A-1b.  Standard schemas carry what any repository
    system reads; the custom opt: schema carries attribution and run
    provenance, each property declared in the PDF/A extension schema.
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
        return f"""<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>
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
    Document Info dictionary + XMP + viewer prefs.  Title/Author describe
    the book (kept from the source PDF when present); the tool, run, source
    checksum and preprocessing are recorded in Creator, Keywords and XMP.
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
        "OCRModel": TROCR_MODEL_NAME, "PageLayout": PAGE_LAYOUT, "PagesTotal": str(len(doc)),
    }
    params = (f"{APP_SCRIPT}; DPI {DPI}; {SEGMENTATION_MODEL}; OCR {TROCR_MODEL_NAME}; "
              f"{PAGE_LAYOUT}; run {RUN_ID}")

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
        title=title, author=author, subject=DOC_SUBJECT, keywords=DOC_KEYWORDS,
        creator=APP_CREATOR, producer=producer,
        creation_date=xmp_date, modify_date=xmp_date, language=DOC_LANGUAGE,
        opt_values=opt_values, source_file=filename,
        doc_id=doc_id, instance_id=instance_id, event_params=params,
    )
    if xmp:
        doc.set_xml_metadata(xmp)
    cat = doc.pdf_catalog()
    doc.xref_set_key(cat, "ViewerPreferences", "<</DisplayDocTitle true>>")
    doc.xref_set_key(cat, "Lang", f"({DOC_LANGUAGE})")
    logger.info(f"Metadata: {APP_CREATOR}, run {RUN_ID}" + (f", source SHA-256 {sha}" if sha else ""))


# ---------------- Device and Memory Helpers ----------------
def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if USE_MPS and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def free_device_memory() -> None:
    """Release freed Python objects and cached accelerator memory."""
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


# ---------------- Model Loading ----------------
seg_model = None
processor = None
trocr_model = None


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
        trocr_model.eval()
        device = _pick_device()
        trocr_model.to(device)
        logger.info(f"Using device: {device}")
        return seg_model, processor, trocr_model
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


def _calibrate_morph_sign() -> int:
    """
    Insert a test string tilted by fitz.Matrix(+10) on a scratch page and
    read back where its characters landed.  Returns +1 if a positive angle
    tilts text downward-right in page coordinates (y down), else -1.  This
    removes any doubt about PyMuPDF's morph convention across versions.
    """
    try:
        with fitz.open() as scratch:
            pg = scratch.new_page(width=400, height=400)
            p = fitz.Point(50, 200)
            pg.insert_text(p, "MMMMMMMMMM", fontsize=12, fontname=FONT_NAME,
                           morph=(p, fitz.Matrix(10)))
            chars = [c for b in pg.get_text("rawdict")["blocks"]
                     for ln in b.get("lines", []) for s in ln["spans"] for c in s["chars"]]
            if len(chars) >= 2:
                return 1 if chars[-1]["origin"][1] > chars[0]["origin"][1] else -1
    except Exception as e:
        logger.warning(f"Could not calibrate text rotation ({e}); assuming default.")
    return 1


# ---------------- Image helpers (OCR copy only) ----------------
def _otsu(arr: np.ndarray) -> float:
    """Otsu threshold of a uint8 array."""
    hist = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128.0
    p = hist / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    denom = omega * (1.0 - omega)
    denom[denom == 0] = np.nan
    sigma_b = (mu[-1] * omega - mu) ** 2 / denom
    if np.all(np.isnan(sigma_b)):
        return 128.0
    return float(np.nanargmax(sigma_b))


def flatten_illumination(gray: Image.Image) -> Image.Image:
    """
    Remove gutter shadow and uneven lighting: estimate the paper
    brightness with a max filter (wipes out the text) plus a heavy blur on
    a small copy, then divide the image by it.  Text in the shadowed spine
    zone ends up with the same contrast as text in the open page.
    """
    w, h = gray.size
    scale = max(1, int(round(max(w, h) / 800)))
    small = gray.resize((max(1, w // scale), max(1, h // scale)), Image.BILINEAR)
    bg = small.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.GaussianBlur(radius=15))
    bg = bg.resize((w, h), Image.BILINEAR)
    g = np.asarray(gray, dtype=np.float32)
    b = np.maximum(np.asarray(bg, dtype=np.float32), 1.0)
    out = np.clip(g / b * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out, "L")


def spine_graded_sharpen(gray: Image.Image, spine_side: Optional[str]) -> Image.Image:
    """
    Blend in an unsharp-masked copy with a weight that is 1 at the spine
    edge and falls smoothly to 0 across SPINE_ZONE_FRAC of the page width.
    """
    if spine_side not in ("left", "right") or SPINE_SHARPEN_PERCENT <= 0:
        return gray
    w, h = gray.size
    zone = max(1.0, w * SPINE_ZONE_FRAC)
    dist = np.arange(w, dtype=np.float32)
    if spine_side == "right":
        dist = (w - 1) - dist
    ramp = np.clip(1.0 - dist / zone, 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)              # smoothstep
    if ramp.max() <= 0:
        return gray
    sharp = gray.filter(ImageFilter.UnsharpMask(
        radius=SPINE_SHARPEN_RADIUS, percent=SPINE_SHARPEN_PERCENT, threshold=2))
    mask = Image.fromarray(np.tile((ramp * 255).astype(np.uint8), (h, 1)), "L")
    return Image.composite(sharp, gray, mask)


def estimate_skew(gray: Image.Image) -> float:
    """
    Projection-profile deskew.  Returns the angle (degrees, PIL convention:
    positive = rotate counter-clockwise) that makes the text lines of this
    page horizontal, or 0.0 if no angle is clearly better than none.

    Only the central part of the page is measured, so the book edge, the
    scanner bed and the gutter don't dominate the profile.
    """
    w, h = gray.size
    region = gray.crop((int(w * 0.08), int(h * 0.05), int(w * 0.92), int(h * 0.95)))
    scale = max(1.0, region.width / 900.0)
    small = region.resize((max(1, int(region.width / scale)), max(1, int(region.height / scale))),
                          Image.BILINEAR)
    arr = np.asarray(small, dtype=np.uint8)
    ink = arr < _otsu(arr)
    if ink.mean() < 0.002 or ink.mean() > 0.5:
        return 0.0
    bin_img = Image.fromarray((ink * 255).astype(np.uint8), "L")

    def score(a: float) -> float:
        rows = np.asarray(bin_img.rotate(a, resample=Image.NEAREST, fillcolor=0),
                          dtype=np.float32).sum(axis=1)
        return float(np.sum(np.diff(rows) ** 2))

    coarse = np.arange(-MAX_SKEW_DEG, MAX_SKEW_DEG + 1e-6, 0.5)
    best = float(coarse[int(np.argmax([score(a) for a in coarse]))])
    fine = np.arange(best - 0.5, best + 0.5 + 1e-6, 0.05)
    fine_scores = [score(a) for a in fine]
    best = float(fine[int(np.argmax(fine_scores))])
    s0 = score(0.0)
    if s0 > 0 and (max(fine_scores) - s0) / s0 < SKEW_MIN_GAIN:
        return 0.0
    if abs(best) < MIN_SKEW_DEG:
        return 0.0
    return round(best, 2)


def rotate_with_map(img: Image.Image, angle_deg: float, fill) -> Tuple[Image.Image, Tuple[float, ...]]:
    """
    Rotate *img* counter-clockwise by angle_deg, expanding the canvas so
    nothing is cut off.  Returns the rotated image and the affine
    (A, B, C, D, E, F) mapping a point (u, v) in the rotated image back to
    (A*u + B*v + C, D*u + E*v + F) in the original.
    """
    if not angle_deg:
        return img, (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    W, H = img.size
    nW = int(math.ceil(W * abs(c) + H * abs(s)))
    nH = int(math.ceil(W * abs(s) + H * abs(c)))
    A, B, D, E = c, -s, s, c
    C = W / 2.0 - A * nW / 2.0 - B * nH / 2.0
    F = H / 2.0 - D * nW / 2.0 - E * nH / 2.0
    out = img.transform((nW, nH), Image.AFFINE, (A, B, C, D, E, F),
                        resample=Image.BICUBIC, fillcolor=fill)
    return out, (A, B, C, D, E, F)


# ---------------- Spread layout ----------------
@dataclass
class Region:
    """One book page inside a rendered PDF page, and how OCR space maps back to it."""
    label: str
    box: Tuple[int, int, int, int]          # in the full render, pixels
    spine_side: Optional[str]               # "left", "right" or None
    angle: float = 0.0                      # deskew applied (degrees, CCW)
    affine: Tuple[float, ...] = field(default=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
    width: int = 0                          # width of the deskewed OCR image

    def to_render(self, u: float, v: float) -> Tuple[float, float]:
        A, B, C, D, E, F = self.affine
        return (A * u + B * v + C + self.box[0], D * u + E * v + F + self.box[1])

    def spine_weight(self, x: float) -> float:
        """1.0 at the spine edge, falling to 0.0 at SPINE_ZONE_FRAC of the width."""
        if self.spine_side not in ("left", "right") or self.width <= 0:
            return 0.0
        zone = max(1.0, self.width * SPINE_ZONE_FRAC)
        d = x if self.spine_side == "left" else self.width - x
        return max(0.0, min(1.0, 1.0 - d / zone))


def find_gutter(render: Image.Image) -> int:
    """
    x-position (render pixels) of the spine in a two-page spread.

    Within GUTTER_SEARCH_BAND, find the widest text-free band around the
    column with the least ink (measured on an illumination-flattened copy,
    smoothed over ~3% of the width so a thin dark crease doesn't break it
    up).  Inside that band, take the darkest column of the raw scan (the
    gutter shadow or crease); if the band has no real shadow, its middle.
    """
    W, H = render.size
    scale = max(1, W // 1200)
    small = render.convert("L").resize((max(1, W // scale), max(1, H // scale)), Image.BILINEAR)
    raw = np.asarray(small, dtype=np.float32)
    flat = np.asarray(flatten_illumination(small), dtype=np.uint8)
    h, w = raw.shape
    y0, y1 = int(h * 0.05), max(int(h * 0.95), int(h * 0.05) + 1)
    lo, hi = int(w * GUTTER_SEARCH_BAND[0]), int(w * GUTTER_SEARCH_BAND[1])
    if hi - lo < 5:
        return W // 2

    # Text makes a column vary from top to bottom (lines and gaps); a
    # gutter shadow or crease is roughly uniform down the page, so the
    # per-column standard deviation separates them where an ink count
    # would mistake the shadow for text.
    ink = flat[y0:y1].astype(np.float32).std(axis=0)
    shade = raw[y0:y1].mean(axis=0)
    wide = max(3, int(w * 0.03)) | 1
    ink_w = np.convolve(ink, np.ones(wide) / wide, mode="same")[lo:hi]
    shade_s = np.convolve(shade, np.ones(5) / 5, mode="same")[lo:hi]

    i_min = int(np.argmin(ink_w))
    floor_, med = float(ink_w.min()), float(np.median(ink_w))
    limit = floor_ + 0.3 * max(med - floor_, 0.0)
    a = b = i_min
    while a > 0 and ink_w[a - 1] <= limit:
        a -= 1
    while b < len(ink_w) - 1 and ink_w[b + 1] <= limit:
        b += 1
    run = shade_s[a:b + 1]
    x = a + int(np.argmin(run)) if run.max() - run.min() >= 8 else (a + b) // 2
    return int(min(max((lo + x) * scale, 1), W - 1))


def guess_spine_side(render: Image.Image) -> Optional[str]:
    """For a single (unsplit) page: which edge is the spine, if it can be told."""
    if SINGLE_PAGE_SPINE_SIDE in ("left", "right"):
        return SINGLE_PAGE_SPINE_SIDE
    if SINGLE_PAGE_SPINE_SIDE != "auto":
        return None
    W, H = render.size
    g = np.asarray(render.convert("L").resize((200, max(1, int(200 * H / W)))), dtype=np.float32)
    strip = max(2, g.shape[1] // 16)
    left, right = float(g[:, :strip].mean()), float(g[:, -strip:].mean())
    if abs(left - right) < 15:
        return None
    return "left" if left < right else "right"


def plan_regions(render: Image.Image, page_num: int) -> List[Region]:
    """Split a render into book pages (left page first)."""
    W, H = render.size
    is_spread = SPREAD_MODE == "always" or (SPREAD_MODE == "auto" and W / H >= SPREAD_MIN_ASPECT)
    if is_spread:
        gx = find_gutter(render)
        logger.info(f"Page {page_num}: two-page spread, spine at x={gx}px ({gx / W:.1%} of width)")
        return [
            Region("left page",  (0, 0, gx, H), "right"),
            Region("right page", (gx, 0, W, H), "left"),
        ]
    side = guess_spine_side(render)
    logger.info(f"Page {page_num}: single page" + (f", spine on the {side}" if side else ""))
    return [Region("page", (0, 0, W, H), side)]


def prepare_region(render: Image.Image, region: Region) -> Image.Image:
    """
    Build the OCR copy of one book page: crop it from the render, flatten
    the lighting, deskew it, then contrast + spine-graded sharpening.
    Fills in region.angle / region.affine / region.width.
    """
    part = render.crop(region.box)
    if ENABLE_PREPROCESSING or FLATTEN_ILLUMINATION:
        img = part.convert("L")
        if FLATTEN_ILLUMINATION:
            img = flatten_illumination(img)
        fill = 255
    else:
        img = part.copy()
        fill = (255, 255, 255)
    del part

    region.angle = estimate_skew(img.convert("L")) if DESKEW else 0.0
    img, region.affine = rotate_with_map(img, region.angle, fill)
    region.width = img.width

    if ENABLE_PREPROCESSING:
        img = ImageOps.autocontrast(img, cutoff=1)
        img = img.filter(ImageFilter.SHARPEN)
        img = spine_graded_sharpen(img, region.spine_side)
    return img.convert("RGB")


def crop_line(img: Image.Image, boundary: List[Tuple[float, float]],
              bbox: Tuple[float, float, float, float]) -> Image.Image:
    """Crop one line, padded, with everything outside its polygon whitened."""
    x0, y0, x1, y1 = bbox
    bx0, by0 = max(0, int(x0) - LINE_PAD_PX), max(0, int(y0) - LINE_PAD_PX)
    bx1, by1 = min(img.width, int(math.ceil(x1)) + LINE_PAD_PX), min(img.height, int(math.ceil(y1)) + LINE_PAD_PX)
    crop = img.crop((bx0, by0, bx1, by1))
    if LINE_POLYGON_MASK and len(boundary) >= 3:
        mask = Image.new("L", crop.size, 0)
        ImageDraw.Draw(mask).polygon([(px - bx0, py - by0) for px, py in boundary], fill=255)
        mask = mask.filter(ImageFilter.MaxFilter(5))      # don't shave ascenders/descenders
        crop = Image.composite(crop, Image.new("RGB", crop.size, (255, 255, 255)), mask)
    return crop


# ── Render a PDF page to a PIL image ──
def page_to_pil(page: fitz.Page, dpi: int = DPI) -> Image.Image:
    """Render *page* at *dpi* and return an RGB PIL Image."""
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


# ---------------- Text Recognition with TrOCR ----------------
def _eos_id(model, processor) -> Optional[int]:
    gcfg = getattr(model, "generation_config", None)
    eos = getattr(gcfg, "eos_token_id", None) if gcfg is not None else None
    if eos is None:
        eos = processor.tokenizer.eos_token_id
    if isinstance(eos, (list, tuple)):
        eos = eos[0] if eos else None
    return eos


def _batch_confidences(out, n: int, eos_id: Optional[int]) -> List[float]:
    """
    Mean over decoding steps of the top softmax probability, per sample;
    steps after a sample's own end-of-sequence token are excluded.
    """
    if not out.scores:
        return [0.0] * n
    steps = len(out.scores)
    rows = out.scores[0].shape[0]
    beams = max(1, rows // n)
    step_max = torch.stack(
        [torch.softmax(s.float(), dim=-1).max(dim=-1).values for s in out.scores]
    ).cpu()
    seqs = out.sequences[:, 1:].cpu()
    confs: List[float] = []
    for i in range(n):
        length = steps
        if eos_id is not None:
            hits = (seqs[i] == eos_id).nonzero(as_tuple=True)[0]
            if hits.numel():
                length = min(steps, int(hits[0]) + 1)
        if length <= 0:
            confs.append(0.0)
            continue
        vals = step_max[:length, i * beams:(i + 1) * beams].max(dim=1).values
        confs.append(float(vals.mean()))
    return confs


def _read_trocr_batch(images: List[Image.Image], processor, model) -> List[Tuple[str, float]]:
    pixel_values = processor(images=images, return_tensors="pt").pixel_values
    device = next(model.parameters()).device
    pixel_values = pixel_values.to(device)
    with torch.inference_mode():
        out = model.generate(pixel_values, output_scores=True, return_dict_in_generate=True)
    texts = processor.batch_decode(out.sequences, skip_special_tokens=True)
    confs = _batch_confidences(out, len(images), _eos_id(model, processor))
    del out, pixel_values
    return [(t.strip(), c) for t, c in zip(texts, confs)]


def recognize_lines_with_trocr(images: List[Image.Image], processor, model) -> List[Tuple[str, float]]:
    """
    Read line crops TROCR_BATCH_SIZE at a time.  A batch that fails on the
    GPU moves the model to the CPU and is retried; a batch that still fails
    returns empty reads instead of ending the run.
    """
    results: List[Tuple[str, float]] = []
    for i in range(0, len(images), TROCR_BATCH_SIZE):
        batch = images[i:i + TROCR_BATCH_SIZE]
        try:
            results.extend(_read_trocr_batch(batch, processor, model))
            continue
        except Exception as e:
            exc = e
        device = next(model.parameters()).device
        if device.type != "cpu":
            logger.warning(f"TrOCR batch failed on {device} ({str(exc)[:120]}); falling back to CPU.")
            free_device_memory()
            model.to(torch.device("cpu"))
            try:
                results.extend(_read_trocr_batch(batch, processor, model))
                continue
            except Exception as e2:
                exc = e2
        logger.error(f"Error recognising text with TrOCR: {exc}")
        results.extend([("", 0.0)] * len(batch))
    return results


def recognize_text_with_trocr(image: Image.Image, processor, model) -> tuple[str, float]:
    """Single-crop wrapper kept for compatibility."""
    return recognize_lines_with_trocr([image], processor, model)[0]


# ---------------- Noise Detection ----------------
def is_likely_noise(text: str, confidence: float, seg_h: int, seg_w: int, relax: float = 0.0) -> bool:
    """
    relax (0..SPINE_CONF_RELAX) lowers the confidence thresholds for lines
    near the spine, where correct reads come back with lower confidence.
    """
    k = 1.0 - max(0.0, min(relax, 0.9))
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
        return confidence < SINGLE_CHAR_CONFIDENCE_THRESHOLD * k
    if confidence < CONFIDENCE_THRESHOLD * k:
        return True
    if len(set(tc)) == 1 and tl > 2:
        return True
    noise_patterns = [r"^[oOlI\.\|]+$", r"^[0-9\.\,]+$", r"^[^a-zA-Z0-9\s]+$"]
    for pat in noise_patterns:
        if re.match(pat, tc) and confidence < SINGLE_CHAR_CONFIDENCE_THRESHOLD * k:
            return True
    if tl > 3 and not any(c.lower() in "aeiou" for c in tc) and confidence < 0.7 * k:
        return True
    return False


# ---------------- Reading order ----------------
def geometric_column_sort(lines: List) -> List:
    """
    Pure-geometry fallback column sorter (horizontal projection histogram).
    Only used if Kraken returns no reading order.
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
    Kraken blla returns lines already in reading order; use that.  Each
    book page is segmented on its own, so the order never jumps across
    the spine.
    """
    lines = getattr(segmentation, "lines", [])
    if not lines:
        return []
    return list(lines)


# ---------------- OCR Text Element Extraction ----------------
def ocr_region(render: Image.Image, region: Region, page_num: int, filename: str) -> List[dict]:
    """
    Segment + read one book page.  Returned elements are in full-render
    pixel space: start/end are the left and right ends of the baseline, so
    they carry the page's original skew.
    """
    out: List[dict] = []
    filtered = 0
    ocr_image = prepare_region(render, region)

    if DEBUG_SAVE_INTERMEDIATE:
        debug_dir = Path("debug_images")
        debug_dir.mkdir(exist_ok=True)
        ocr_image.save(debug_dir / f"{filename}_page{page_num}_{region.label.replace(' ', '_')}.png")

    segmentation = blla.segment(ocr_image, model=seg_model)
    lines = order_lines(segmentation)
    del segmentation
    logger.info(f"Page {page_num} {region.label}: deskew {region.angle:+.2f}°, {len(lines)} text lines")

    if not lines:
        logger.warning("No text lines detected. Saving debug image...")
        debug_dir = Path("debug_images")
        debug_dir.mkdir(exist_ok=True)
        ocr_image.save(debug_dir / f"{filename}_page{page_num}_{region.label.replace(' ', '_')}_preprocessed.png")
        return out

    jobs: List[dict] = []
    crops: List[Image.Image] = []
    for i, line in enumerate(lines):
        try:
            raw_b = getattr(line, "boundary", None)
            boundary = [(float(p[0]), float(p[1])) for p in raw_b] if raw_b is not None else []
            if len(boundary) >= 3:
                xs = [p[0] for p in boundary]
                ys = [p[1] for p in boundary]
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            elif hasattr(line, "bbox"):
                x0, y0, x1, y1 = line.bbox
                boundary = []
            else:
                continue

            sh, sw = y1 - y0, x1 - x0
            if sh < 5 or sw < 5:
                filtered += 1
                continue

            # Baseline ends give the text position; falls back to the
            # bottom of the line box when Kraken gives no baseline.
            raw_bl = getattr(line, "baseline", None)
            baseline = [(float(p[0]), float(p[1])) for p in raw_bl] if raw_bl is not None else []
            if len(baseline) >= 2:
                y_start = min(baseline, key=lambda p: p[0])[1]
                y_end   = max(baseline, key=lambda p: p[0])[1]
            else:
                y_start = y_end = y1

            crops.append(crop_line(ocr_image, boundary, (x0, y0, x1, y1)))
            jobs.append({
                "start": region.to_render(x0, y_start),
                "end":   region.to_render(x1, y_end),
                "sh": sh, "sw": sw,
                "relax": SPINE_CONF_RELAX * region.spine_weight((x0 + x1) / 2),
            })
        except Exception as e:
            logger.error(f"Error on text line {i + 1}: {e}")
    del lines, ocr_image

    reads = recognize_lines_with_trocr(crops, processor, trocr_model)
    del crops

    for job, (text, confidence) in zip(jobs, reads):
        if is_likely_noise(text, confidence, job["sh"], job["sw"], job["relax"]):
            filtered += 1
            continue
        out.append({
            "start":     job["start"],
            "end":       job["end"],
            "font_size": max(6, min(job["sh"] * 0.9, 72)),
            "text":      text,
        })

    logger.info(f"Page {page_num} {region.label}: {len(out)} elements, {filtered} filtered.")
    return out


def create_page_ocr_elements(
    pil_image: Image.Image,
    page_num: int,
    total_pages: int,
    filename: str,
) -> List[dict]:
    """
    OCR one rendered PDF page: split it into book pages if it is a spread,
    then process each (left first).  Element keys: start, end (render-pixel
    baseline ends), font_size (render pixels), text.
    """
    logger.info(f"Processing page {page_num}/{total_pages} of {filename}")
    elements: List[dict] = []
    try:
        for region in plan_regions(pil_image, page_num):
            try:
                elements.extend(ocr_region(pil_image, region, page_num, filename))
            except Exception as e:
                logger.error(f"OCR failed for page {page_num} {region.label}: {e}")
            free_device_memory()
    except Exception as e:
        logger.error(f"OCR failed for page {page_num}: {e}")
    return elements


def create_ocr_text_elements(pil_images: List[Image.Image], filename: str) -> List[List[dict]]:
    """Compatibility wrapper: OCR a list of already-rendered pages."""
    font_path = Path(FONT_PATH)
    if not font_path.exists():
        raise FileNotFoundError(f"Required font {font_path} is missing.")
    all_pages: List[List[dict]] = []
    for idx, pil_image in enumerate(pil_images):
        all_pages.append(create_page_ocr_elements(pil_image, idx + 1, len(pil_images), filename))
        free_device_memory()
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
            if "/OutputIntents" not in pdf.Root:
                pdf.Root["/OutputIntents"] = pikepdf.Array()
            icc_data = srgb_path.read_bytes()
            icc_stream = pdf.make_stream(icc_data)
            icc_stream.stream_dict["/N"]         = pikepdf.Integer(3)
            icc_stream.stream_dict["/Alternate"] = pikepdf.Name("/DeviceRGB")
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
    Add an invisible OCR text layer to *input_path*, one page at a time,
    and write the result to *output_path*.  Text for a skewed book page is
    inserted at that page's angle, so it lines up with the image.
    """
    filename = os.path.basename(input_path)
    logger.info(f"Starting OCR for: {filename}")

    try:
        font_path = Path(FONT_PATH)
        if not font_path.exists():
            raise FileNotFoundError(f"Required font {font_path} is missing.")

        with fitz.open(input_path) as doc:
            n_pages = len(doc)
            stamp_document_metadata(doc, filename, input_path)

            logger.info(f"Rendering {n_pages} pages at {DPI} DPI for OCR, one page at a time…")
            total_elements = 0

            for page_num in range(n_pages):
                page = doc[page_num]

                pil_img = page_to_pil(page, dpi=DPI)
                img_w, img_h = pil_img.size
                elements = create_page_ocr_elements(pil_img, page_num + 1, n_pages, filename)
                del pil_img
                free_device_memory()
                total_elements += len(elements)

                # ── Strip any pre-existing text layer ────────────────────
                existing_text = page.get_text().strip()
                if existing_text:
                    logger.info(
                        f"Page {page_num+1}: removing existing text layer "
                        f"({len(existing_text)} chars) before inserting OCR."
                    )
                    page.add_redact_annot(page.rect)
                    redact_kwargs = {"images": fitz.PDF_REDACT_IMAGE_NONE}
                    if hasattr(fitz, "PDF_REDACT_LINE_ART_NONE"):
                        redact_kwargs["graphics"] = fitz.PDF_REDACT_LINE_ART_NONE
                    page.apply_redactions(**redact_kwargs)

                page_w = page.rect.width
                page_h = page.rect.height
                sx     = page_w / img_w
                sy     = page_h / img_h

                inserted = 0
                for elem in elements:
                    try:
                        p0 = fitz.Point(elem["start"][0] * sx, elem["start"][1] * sy)
                        p1 = fitz.Point(elem["end"][0] * sx, elem["end"][1] * sy)
                        dx, dy = p1.x - p0.x, p1.y - p0.y
                        length = math.hypot(dx, dy)
                        angle = math.degrees(math.atan2(dy, dx))   # page coords, y down
                        fontsize = max(4, elem["font_size"] * sy)

                        # Keep the string within the line's own length and
                        # inside the page.
                        if dx > 0 and p1.x > page_w:
                            length *= max(0.0, (page_w - p0.x) / dx)
                        if length > 0:
                            text_w = fitz.get_text_length(elem["text"], fontname=FONT_NAME, fontsize=fontsize)
                            if text_w > length:
                                fontsize = max(4, fontsize * length / text_w)

                        kwargs = {}
                        if abs(angle) >= 0.05:
                            kwargs["morph"] = (p0, fitz.Matrix(MORPH_SIGN * angle))

                        page.insert_text(
                            p0,
                            elem["text"],
                            fontsize=fontsize,
                            fontname=FONT_NAME,
                            render_mode=3,
                            color=(0, 0, 0),
                            **kwargs,
                        )
                        inserted += 1
                    except Exception as e:
                        logger.error(f"Failed to insert text on page {page_num+1}: {e}")

                logger.info(f"Page {page_num+1}: inserted {inserted}/{len(elements)} text elements")
                del elements

            logger.info(f"OCR extraction complete: {total_elements} total text elements across {n_pages} pages")

            doc.save(
                output_path,
                deflate=True,
                garbage=4,
                clean=True,
                deflate_images=False,  # leave original image streams untouched
                encryption=fitz.PDF_ENCRYPT_KEEP,
            )
            logger.info(f"OCR-enhanced PDF saved: {output_path}")

        srgb_path = Path(SRGB_ICC_PATH)
        if srgb_path.exists():
            logger.info("Applying PDF/A OutputIntent…")
            setup_pdfa_compliance(output_path)
        else:
            logger.warning("sRGB ICC profile not found; PDF/A OutputIntent skipped.")

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
    finally:
        free_device_memory()


# ---------------- Compression (Size Targeting) ----------------
def compress_to_target_size(input_pdf: Path, output_pdf: Path, original_size: int) -> Path:
    """
    Keep the output within 15% of the original size using PDF-native
    deflate only.  No image recompression (archival quality).
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
        "Returning OCR file as-is (images untouched, quality preserved)."
    )
    shutil.copy2(input_pdf, output_pdf)
    return output_pdf


# ---------------- Main ----------------
def main():
    global seg_model, processor, trocr_model, MORPH_SIGN

    logger.info(f"{APP_NAME} {APP_VERSION} ({APP_SCRIPT}) by {APP_AUTHOR}, {APP_INSTITUTION} — "
                f"{APP_LICENSE} licence — {APP_URL}")
    logger.info(f"Run {RUN_ID}")
    logger.info(f"Layout: {PAGE_LAYOUT}")
    input_folder  = Path(INPUT_DIR)
    output_folder = Path(OUTPUT_DIR)

    if not input_folder.exists():
        logger.error(f"Input folder '{INPUT_DIR}' not found.")
        sys.exit(1)
    output_folder.mkdir(exist_ok=True)

    pdf_files = sorted(input_folder.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files in '{INPUT_DIR}'")
        sys.exit(1)

    skipped, pending, summary = [], [], []
    for pdf_path in pdf_files:
        final_path = output_folder / f"{pdf_path.stem}.pdf"
        if SKIP_EXISTING_OUTPUT and final_path.exists():
            skipped.append(pdf_path.name)
            summary.append((pdf_path.name, "SKIPPED", final_path.stat().st_size))
        else:
            pending.append(pdf_path)

    if skipped:
        logger.info(f"Skipping {len(skipped)} already-processed file(s): {', '.join(skipped)}")

    if not pending:
        logger.info("All files have already been processed. Nothing to do.")
        return

    try:
        seg_model, processor, trocr_model = load_models()
    except Exception:
        logger.error("Model loading failed. Exiting.")
        sys.exit(1)

    MORPH_SIGN = _calibrate_morph_sign()
    logger.debug(f"Text rotation sign: {MORPH_SIGN:+d}")

    logger.info(f"Processing {len(pending)} file(s) with TrOCR: {TROCR_MODEL_NAME}")
    logger.info("Target: Final size ≤ original + 15% (OCR text layer only; images untouched)")

    for pdf_path in pending:
        original_size = pdf_path.stat().st_size
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing {pdf_path.name} | Original: {original_size // 1024} KB")

        ocr_temp_path = output_folder / f"{pdf_path.stem}_ocr_temp.pdf"

        if not process_single_pdf_ocr(str(pdf_path), str(ocr_temp_path)):
            logger.error(f"Skipping {pdf_path.name} due to OCR failure.")
            summary.append((pdf_path.name, "FAILED", 0))
            ocr_temp_path.unlink(missing_ok=True)
            free_device_memory()
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
            summary.append((pdf_path.name, "OK", final_size))
        else:
            logger.error(f"Failed to generate final output for {pdf_path.name}")
            summary.append((pdf_path.name, "FAILED", 0))

        try:
            ocr_temp_path.unlink()
        except Exception as e:
            logger.warning(f"Could not delete temp file: {e}")

        free_device_memory()

    logger.info(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    for name, status, sz in summary:
        logger.info(f"  {status:<8}  {name}  ({sz // 1024} KB)")
    logger.info(f"\nAll done! Output files in '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
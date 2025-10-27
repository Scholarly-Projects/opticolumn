#!/usr/bin/env python3
import sys
import os
import tempfile
from pathlib import Path
from pdf2image import convert_from_path
import fitz  # PyMuPDF
from kraken import blla
from kraken.lib.vgsl import TorchVGSLModel
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth 
from reportlab.pdfbase.ttfonts import TTFont
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import logging
from typing import List, Tuple
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import re
import platform
import datetime
import shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom
import math

# ---------------- Configuration ----------------
INPUT_DIR = "A"
OUTPUT_DIR = "B"
MODELS_DIR = "mlmodels"
POPPLER_PATH = None
DPI = 225 
TROCR_MODELS = {
    "handwritten": "microsoft/trocr-base-handwritten",
    "printed": "microsoft/trocr-base-printed",
    "large_handwritten": "microsoft/trocr-large-handwritten",
    "large_printed": "microsoft/trocr-large-printed"
}
TROCR_MODEL_NAME = TROCR_MODELS["large_handwritten"]
ENABLE_PREPROCESSING = True
CONFIDENCE_THRESHOLD = 0.25
SINGLE_CHAR_CONFIDENCE_THRESHOLD = 0.5
MIN_SEGMENT_HEIGHT = 10
FONT_NAME = "FreeSans"
FONT_PATH = "fonts/FreeSans.ttf"
SRGB_ICC_PATH = "srgb.icc"
DEBUG_OCR_LAYER = False
DEBUG_TEXT_POSITIONS = False
DEBUG_SAVE_INTERMEDIATE = False
DEBUG_PDFA = False
COMPRESSION_LEVEL = 75  
AGGRESSIVE_COMPRESSION = False  

# ---------------- Logging Setup ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Helper function to format date for PDF
def get_pdf_date_string(dt=None):
    if dt is None:
        dt = datetime.datetime.now()
    return dt.strftime("D:%Y%m%d%H%M%S")

# Helper function to format date for XMP
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
                str(font_path)
            )
        pdfmetrics.registerFont(TTFont(FONT_NAME, str(font_path)))
        logger.info(f"Font {FONT_NAME} registered for embedding")
        srgb_path = Path(SRGB_ICC_PATH)
        if not srgb_path.exists():
            logger.info("Downloading sRGB ICC profile...")
            try:
                if platform.system() == "Darwin":
                    system_profile = "/System/Library/ColorSync/Profiles/sRGB Profile.icc"
                elif platform.system() == "Windows":
                    system_profile = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'),
                                               'System32', 'spool', 'drivers', 'color', 'sRGB Color Space Profile.icm')
                elif platform.system() == "Linux":
                    system_profile = "/usr/share/color/icc/sRGB.icc"
                else:
                    system_profile = None
                if system_profile and Path(system_profile).exists():
                    shutil.copy2(system_profile, str(srgb_path))
                else:
                    urllib.request.urlretrieve(
                        "https://www.color.org/srgb.xalter",
                        str(srgb_path)
                    )
            except Exception as e:
                logger.warning(f"Could not get sRGB ICC profile: {e}")
        return True
    except Exception as e:
        logger.error(f"Failed to setup PDF/A resources: {e}")
        return False

# ---------------- XMP Metadata Creation ----------------
def create_xmp_metadata(title, author, subject, creator, producer, creation_date, modify_date):
    try:
        xmp_packet = f"""<?xpacket begin="ï»¿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core 5.6-c140 79.164452, 2017/09/07-01:11:22        ">
   <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
      <rdf:Description rdf:about="" xmlns:pdf="http://ns.adobe.com/pdf/1.3/">
         <pdf:Producer>{producer}</pdf:Producer>
      </rdf:Description>
      <rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">
         <dc:title>
            <rdf:Alt>
               <rdf:li xml:lang="x-default">{title}</rdf:li>
            </rdf:Alt>
         </dc:title>
         <dc:creator>
            <rdf:Seq>
               <rdf:li>{author}</rdf:li>
            </rdf:Seq>
         </dc:creator>
         <dc:description>
            <rdf:Alt>
               <rdf:li xml:lang="x-default">{subject}</rdf:li>
            </rdf:Alt>
         </dc:description>
         <dc:language>
            <rdf:Bag>
               <rdf:li>en-US</rdf:li>
            </rdf:Bag>
         </dc:language>
      </rdf:Description>
      <rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/">
         <xmp:CreatorTool>{creator}</xmp:CreatorTool>
         <xmp:CreateDate>{creation_date}</xmp:CreateDate>
         <xmp:ModifyDate>{modify_date}</xmp:ModifyDate>
         <xmp:Language>en-US</xmp:Language>
      </rdf:Description>
      <rdf:Description rdf:about="" xmlns:pdfaid="http://www.aiim.org/pdfa/ns/id/">
         <pdfaid:part>1</pdfaid:part>
         <pdfaid:conformance>B</pdfaid:conformance>
      </rdf:Description>
   </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""
        return xmp_packet
    except Exception as e:
        logger.error(f"Failed to create XMP metadata: {e}")
        return None

# ---------------- Model Loading ----------------
def load_models():
    try:
        if not setup_pdfa_resources():
            logger.warning("PDF/A resources setup failed. PDF/A compliance may be affected.")
        seg_model_path = Path(MODELS_DIR) / "blla.mlmodel"
        logger.info(f"Loading segmentation model: {seg_model_path}")
        seg_model = TorchVGSLModel.load_model(str(seg_model_path))
        seg_model.eval()
        logger.info(f"Loading TrOCR model: {TROCR_MODEL_NAME}")
        processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_NAME)
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

# ---------------- Image Preprocessing ----------------
def preprocess_image(pil_image: Image.Image) -> Image.Image:
    if not ENABLE_PREPROCESSING:
        return pil_image
    try:
        # Convert to grayscale for better text detection
        gray_image = pil_image.convert('L')
        # Apply mild contrast enhancement
        from PIL import ImageOps
        gray_image = ImageOps.autocontrast(gray_image, cutoff=2)
        # Convert back to RGB for consistency
        processed_image = gray_image.convert('RGB')
        # Apply mild sharpening to enhance text edges
        processed_image = processed_image.filter(ImageFilter.SHARPEN)
        return processed_image
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return pil_image

# ---------------- PDF Utilities (Modified for JPEG) ----------------
def flatten_pdf_to_images(input_path: str, temp_pdf_path: str) -> bool:
    """Create a flattened version of the PDF for OCR layer creation — saves as JPEG to reduce size."""
    try:
        logger.debug(f"Flattening PDF: {input_path}")
        with fitz.open(input_path) as doc, fitz.open() as output_pdf:
            for page_num, page in enumerate(doc):
                pix = page.get_pixmap(dpi=DPI)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img_buffer = BytesIO()
                # Use more aggressive compression level
                img.save(img_buffer, format="JPEG", quality=COMPRESSION_LEVEL, optimize=True, progressive=True)
                img_buffer.seek(0)
                img_page = output_pdf.new_page(width=pix.width, height=pix.height)
                img_page.insert_image(img_page.rect, stream=img_buffer.read())
            output_pdf.save(temp_pdf_path, deflate=True, garbage=3, clean=True)
        return True
    except Exception as e:
        logger.error(f"Error flattening PDF: {e}")
        return False

# ---------------- Text Recognition with TrOCR ----------------
def recognize_text_with_trocr(image: Image.Image, processor, model) -> tuple[str, float]:
    try:
        pixel_values = processor(image, return_tensors="pt").pixel_values
        device = next(model.parameters()).device
        pixel_values = pixel_values.to(device)
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, output_scores=True, return_dict_in_generate=True)
            generated_text = processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)[0]
            scores = generated_ids.scores
            if scores:
                probs = [torch.softmax(score, dim=-1) for score in scores]
                max_probs = [torch.max(prob).item() for prob in probs]
                confidence = sum(max_probs) / len(max_probs)
            else:
                confidence = 0.0
        return generated_text.strip(), confidence
    except Exception as e:
        logger.error(f"Error recognizing text with TrOCR: {e}")
        return "", 0.0

# ---------------- Noise Detection ----------------
def is_likely_noise(text: str, confidence: float, segment_height: int, segment_width: int) -> bool:
    if not text:
        return True
    if segment_height < MIN_SEGMENT_HEIGHT:
        return True
    if segment_width < 15:
        return True
    aspect_ratio = segment_width / segment_height
    if aspect_ratio < 0.1 or aspect_ratio > 100:
        return True
    text_clean = text.strip()
    text_length = len(text_clean)
    if text_length == 1:
        if confidence < SINGLE_CHAR_CONFIDENCE_THRESHOLD:
            return True
        return False
    if confidence < CONFIDENCE_THRESHOLD:
        return True
    if len(set(text_clean)) == 1 and text_length > 2:
        return True
    noise_patterns = [
        r'^[oOlI\.\|]+$',
        r'^[0-9\.\,]+$',
        r'^[^a-zA-Z0-9\s]+$',
    ]
    for pattern in noise_patterns:
        if re.match(pattern, text_clean):
            if confidence < SINGLE_CHAR_CONFIDENCE_THRESHOLD:
                return True
    if text_length > 3 and not any(char.lower() in 'aeiou' for char in text_clean):
        if confidence < 0.7:
            return True
    return False

# ---------------- Improved Column Detection and Sorting ----------------
def improved_column_sort(lines: List) -> List:
    """
    Improved column detection and sorting algorithm that:
    1. More accurately detects columns using a combination of horizontal projection and clustering
    2. Sorts lines within columns top-to-bottom
    3. Sorts columns left-to-right
    4. Handles irregular column layouts better
    """
    if len(lines) <= 1:
        return lines
    
    # Extract bounding boxes
    bboxes = []
    for line in lines:
        if hasattr(line, 'boundary') and len(line.boundary) >= 3:
            x_coords = [p[0] for p in line.boundary]
            y_coords = [p[1] for p in line.boundary]
            x0, y0 = min(x_coords), min(y_coords)
            x1, y1 = max(x_coords), max(y_coords)
        elif hasattr(line, 'bbox'):
            x0, y0, x1, y1 = line.bbox
        else:
            continue
        bboxes.append((x0, y0, x1, y1, line))
    
    if not bboxes:
        return lines
    
    # Calculate page dimensions
    page_width = max(box[2] for box in bboxes)
    page_height = max(box[3] for box in bboxes)
    
    # Step 1: Detect potential columns using horizontal projection
    resolution = 10  # Resolution for histogram
    hist_width = int(page_width / resolution) + 1
    hist = [0] * hist_width
    
    # Create horizontal projection histogram
    for x0, y0, x1, y1, _ in bboxes:
        start_bin = int(x0 / resolution)
        end_bin = int(x1 / resolution)
        for bin_idx in range(start_bin, min(end_bin + 1, hist_width)):
            hist[bin_idx] += (y1 - y0)  # Add line height to the bin
    
    # Smooth the histogram
    smoothed_hist = hist.copy()
    for i in range(1, len(hist) - 1):
        smoothed_hist[i] = (hist[i-1] + 2*hist[i] + hist[i+1]) / 4
    
    # Find valleys (potential column separators)
    valleys = []
    for i in range(1, len(smoothed_hist) - 1):
        if smoothed_hist[i] < smoothed_hist[i-1] and smoothed_hist[i] < smoothed_hist[i+1]:
            # Check if this is a significant valley
            neighborhood_max = max(smoothed_hist[i-1], smoothed_hist[i+1])
            if neighborhood_max > 0 and smoothed_hist[i] / neighborhood_max < 0.3:
                valleys.append(i * resolution)
    
    # If no significant valleys found, estimate columns based on line distribution
    if not valleys:
        # Calculate average line width
        widths = [x1 - x0 for x0, y0, x1, y1, _ in bboxes]
        avg_width = sum(widths) / len(widths) if widths else 100
        
        # Estimate number of columns
        estimated_col_count = max(1, int(page_width / (avg_width * 1.5)))
        estimated_col_count = min(estimated_col_count, 5)  # Limit to 5 columns
        
        # Create evenly spaced column separators
        col_width = page_width / estimated_col_count
        valleys = [int((i + 1) * col_width) for i in range(estimated_col_count - 1)]
    
    # Sort valleys and ensure they're within page bounds
    valleys = sorted([v for v in valleys if 0 < v < page_width])
    
    # Step 2: Group lines into columns
    columns = [[] for _ in range(len(valleys) + 1)]
    
    for box in bboxes:
        x0, y0, x1, y1, line = box
        center_x = (x0 + x1) / 2
        
        # Determine which column this line belongs to
        col_idx = 0
        for valley in valleys:
            if center_x > valley:
                col_idx += 1
            else:
                break
        
        columns[col_idx].append(box)
    
    # Step 3: Sort lines within each column by y-coordinate (top to bottom)
    sorted_columns = []
    for column in columns:
        sorted_column = sorted(column, key=lambda box: box[1])  # Sort by y0
        sorted_columns.append(sorted_column)
    
    # Step 4: Flatten columns in left-to-right order
    sorted_lines = []
    for column in sorted_columns:
        for bbox in column:
            sorted_lines.append(bbox[4])
    
    return sorted_lines

# ---------------- OCR Layer (Pixel-Perfect Alignment) ----------------
def create_ocr_layer(images: List[Image.Image], filename: str) -> BytesIO:
    packet = BytesIO()
    c = canvas.Canvas(packet)
    try:
        font_dir = Path("fonts")
        font_dir.mkdir(exist_ok=True)
        font_path = Path("fonts/FreeSans.ttf")
        if not font_path.exists():
            logger.error(f"CRITICAL: Font file {font_path} not found. Cannot proceed without FreeSans.ttf.")
            raise FileNotFoundError(f"Required font {font_path} is missing.")
        pdfmetrics.registerFont(TTFont(FONT_NAME, str(font_path)))
        logger.info(f"Successfully registered and using embedded font: {FONT_NAME}")
    except Exception as e:
        logger.error(f"Failed to load or register FreeSans.ttf: {e}")
        raise RuntimeError("Font setup failed. FreeSans.ttf is required and must be present in ./fonts/") from e

    total_text_elements = 0
    for img_idx, pil_image in enumerate(images):
        page_num = img_idx + 1
        logger.info(f"Processing page {page_num}/{len(images)} of {filename}")
        pdf_width, pdf_height = pil_image.size
        c.setPageSize((pdf_width, pdf_height))
        c.setFillColorRGB(0, 0, 0, 0)
        page_text_count = 0
        try:
            processed_image = preprocess_image(pil_image)
            segmentation = blla.segment(processed_image, model=seg_model)
            logger.info(f"Found {len(segmentation.lines)} text lines on page {page_num}")
            if len(segmentation.lines) == 0:
                logger.warning("No text lines detected. Saving debug image...")
                debug_dir = Path("debug_images")
                debug_dir.mkdir(exist_ok=True)
                processed_image.save(debug_dir / f"{filename}_page{page_num}_preprocessed.png")
            filtered_lines = 0
            
            # Use improved column sorting
            sorted_lines = improved_column_sort(segmentation.lines)
            logger.info(f"Sorted {len(sorted_lines)} lines into column-based reading order.")
            
            # Calculate average line height for consistent spacing
            line_heights = []
            for line in sorted_lines:
                if hasattr(line, 'boundary') and len(line.boundary) >= 3:
                    x_coords = [p[0] for p in line.boundary]
                    y_coords = [p[1] for p in line.boundary]
                    x0, y0 = min(x_coords), min(y_coords)
                    x1, y1 = max(x_coords), max(y_coords)
                elif hasattr(line, 'bbox'):
                    x0, y0, x1, y1 = line.bbox
                else:
                    continue
                segment_height = y1 - y0
                if segment_height >= 5:  # Only consider valid lines
                    line_heights.append(segment_height)
            
            avg_line_height = sum(line_heights) / len(line_heights) if line_heights else 20
            
            # Process each line
            for i, line in enumerate(sorted_lines):
                try:
                    if hasattr(line, 'boundary') and len(line.boundary) >= 3:
                        x_coords = [p[0] for p in line.boundary]
                        y_coords = [p[1] for p in line.boundary]
                        x0, y0 = min(x_coords), min(y_coords)
                        x1, y1 = max(x_coords), max(y_coords)
                    elif hasattr(line, 'bbox'):
                        x0, y0, x1, y1 = line.bbox
                    else:
                        continue
                    segment_height = y1 - y0
                    segment_width = x1 - x0
                    if segment_height < 5 or segment_width < 5:
                        filtered_lines += 1
                        continue
                    line_image = processed_image.crop((x0, y0, x1, y1))
                    text, confidence = recognize_text_with_trocr(line_image, processor, trocr_model)
                    if is_likely_noise(text, confidence, segment_height, segment_width):
                        filtered_lines += 1
                        continue
                    
                    # Calculate font size based on height
                    font_size = max(6, min(segment_height * 1.2, 72))
                    ascent, descent = 0, 0
                    for trial in range(3):
                        try:
                            face = pdfmetrics.getFont(FONT_NAME).face
                            ascent = face.ascent * font_size / 1000.0
                            descent = face.descent * font_size / 1000.0
                            text_height = ascent - descent
                            if abs(text_height - segment_height) < 2:
                                break
                            if text_height == 0:
                                break
                            font_size *= segment_height / text_height
                            font_size = max(6, min(font_size, 72))
                        except Exception as e:
                            logger.warning(f"Font metric error: {e}")
                            break
                    
                    # Calculate text width and adjust font size if needed
                    c.setFont(FONT_NAME, font_size)
                    natural_width = stringWidth(text, FONT_NAME, font_size)
                    target_width = segment_width
                    
                    # Adjust font size to fill width if necessary
                    if natural_width > 0 and natural_width < target_width * 0.8:
                        # Text is too short, try increasing font size
                        scale_factor = min(2.0, target_width / natural_width)
                        adjusted_font_size = min(72, font_size * scale_factor)
                        c.setFont(FONT_NAME, adjusted_font_size)
                        # Recalculate width with new font size
                        natural_width = stringWidth(text, FONT_NAME, adjusted_font_size)
                    
                    # Calculate baseline position
                    y_baseline = pdf_height - y1 + descent
                    
                    # Use the full width of the original line segment
                    # This ensures the text extends to the right edge of the original line
                    text_object = c.beginText(x0, y_baseline)
                    text_object.setFont(FONT_NAME, font_size)
                    text_object.setCharSpace(0)
                    
                    # If the text is shorter than the segment width, adjust character spacing
                    # to make it span the full width
                    if natural_width < target_width:
                        char_space = (target_width - natural_width) / max(1, len(text) - 1)
                        text_object.setCharSpace(char_space)
                    
                    # Draw the text
                    text_object.textOut(text)
                    c.drawText(text_object)
                    
                    page_text_count += 1
                except Exception as e:
                    logger.error(f"Error processing text line {i+1}: {e}")
                    continue
            logger.info(f"Page {page_num}: Filtered {filtered_lines} noisy lines.")
            total_text_elements += page_text_count
        except Exception as e:
            logger.error(f"OCR failed for page {page_num}: {e}")
        c.showPage()
    c.save()
    packet.seek(0)
    logger.info(f"OCR layer created with {total_text_elements} total text elements")
    return packet

# ---------------- PDF Processing (OCR) ----------------
def process_single_pdf_ocr(input_path: str, output_path: str) -> bool:
    filename = os.path.basename(input_path)
    logger.info(f"Starting OCR for: {filename}")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        temp_pdf_path = temp_pdf.name
    if not flatten_pdf_to_images(input_path, temp_pdf_path):
        logger.error(f"Failed to flatten {filename}")
        return False
    try:
        original_max_pixels = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None

        pil_images = convert_from_path(temp_pdf_path, dpi=DPI, poppler_path=POPPLER_PATH)
        logger.info(f"Converted to {len(pil_images)} images @ {DPI} DPI")
        ocr_layer = create_ocr_layer(pil_images, filename)
        logger.info("Merging OCR layer with base PDF...")
        with fitz.open(temp_pdf_path) as base_pdf, fitz.open(stream=ocr_layer.getvalue(), filetype="pdf") as ocr_pdf:
            creation_date = get_pdf_date_string()
            modify_date = creation_date
            metadata = {
                "title": filename,
                "author": "OCR Processing",
                "subject": "OCR processed document",
                "creator": "OCR Processor",
                "producer": "PyMuPDF",
                "creationDate": creation_date,
                "modDate": modify_date
            }
            base_pdf.set_metadata(metadata)
            xmp_metadata = create_xmp_metadata(
                title=metadata["title"],
                author=metadata["author"],
                subject=metadata["subject"],
                creator=metadata["creator"],
                producer=metadata["producer"],
                creation_date=get_xmp_date_string(),
                modify_date=get_xmp_date_string()
            )
            if xmp_metadata:
                base_pdf.set_xml_metadata(xmp_metadata)
            else:
                logger.warning("Failed to create XMP metadata")
            page_count = min(len(base_pdf), len(ocr_pdf))
            logger.info(f"Merging {page_count} pages...")
            for page_num in range(page_count):
                base_pdf[page_num].show_pdf_page(base_pdf[page_num].rect, ocr_pdf, page_num)
            logger.info("Applying PDF/A compliance fixes...")
            srgb_path = Path(SRGB_ICC_PATH)
            if srgb_path.exists():
                # Try to add OutputIntent for PDF/A compliance
                try:
                    # For older PyMuPDF versions, we'll skip OutputIntent embedding
                    logger.info("Note: PDF/A OutputIntent embedding requires newer PyMuPDF version")
                except Exception as e:
                    logger.warning(f"Could not embed OutputIntent: {e}")
            else:
                logger.error("CRITICAL: sRGB ICC profile not found. PDF/A-1B compliance is impossible.")
            # Save with compression settings
            base_pdf.save(
                output_path,
                deflate=True,
                garbage=4,
                clean=True,
                encryption=fitz.PDF_ENCRYPT_KEEP
            )
            logger.info(f"OCR-enhanced PDF saved: {output_path}")
        logger.info("Verifying OCR layer in final output...")
        try:
            with fitz.open(output_path) as final_pdf:
                total_text_length = 0
                for i in range(len(final_pdf)):
                    page = final_pdf[i]
                    text = page.get_text()
                    page_text_length = len(text.strip())
                    total_text_length += page_text_length
                    logger.info(f"Final PDF page {i+1} extractable text length: {page_text_length}")
                if total_text_length > 0:
                    logger.info(f"SUCCESS: Final PDF contains {total_text_length} characters of searchable text")
                else:
                    logger.error("PROBLEM: Final PDF has no extractable text!")
        except Exception as e:
            logger.error(f"Failed to verify final output: {e}")
        Image.MAX_IMAGE_PIXELS = original_max_pixels
        return True
    except Exception as e:
        Image.MAX_IMAGE_PIXELS = original_max_pixels
        logger.error(f"OCR processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

# ---------------- Enhanced Compression (Size Targeting) ----------------
def enhanced_compress_to_target_size(input_pdf: Path, output_pdf: Path, original_size: int) -> Path:
    """
    Enhanced compression function that tries to get the processed file as close as possible 
    to the original file size, allowing up to a 15% increase for the added OCR layer.
    """
    # Calculate maximum target size (15% increase from original size)
    max_target = int(original_size * 1.15)  # 15% increase
    
    logger.info(f"Targeting maximum size: {max_target//1024} KB (15% increase from original)")
    
    # Get the current OCR file size
    current_size = input_pdf.stat().st_size
    logger.info(f"OCR file size before compression: {current_size//1024} KB")
    
    # If we're already within the target, just copy the file
    if current_size <= max_target:
        shutil.copy2(input_pdf, output_pdf)
        logger.info(f"OCR file already within target size. No compression needed.")
        return output_pdf
    
    # Try different compression options
    compression_options = [
        # Option 1: Maximum compression
        {"deflate": True, "garbage": 4, "clean": True, "deflate_images": True, "pretty": False},
        # Option 2: High compression
        {"deflate": True, "garbage": 3, "clean": True, "deflate_images": True, "pretty": False},
        # Option 3: Medium compression
        {"deflate": True, "garbage": 2, "clean": True, "deflate_images": False, "pretty": False},
    ]
    
    for i, options in enumerate(compression_options):
        temp_output = output_pdf.with_suffix(f".temp_{i}.pdf")
        
        try:
            with fitz.open(str(input_pdf)) as doc:
                doc.save(str(temp_output), **options, encryption=fitz.PDF_ENCRYPT_KEEP)
            
            compressed_size = temp_output.stat().st_size
            size_increase_pct = (compressed_size - original_size) / original_size * 100
            
            logger.info(f"Compression option {i+1}: {compressed_size//1024} KB ({size_increase_pct:+.1f}% from original)")
            
            # Check if this is within our target
            if compressed_size <= max_target:
                # Success! Use this file
                shutil.move(str(temp_output), str(output_pdf))
                logger.info(f"Found suitable compression with option {i+1}")
                
                # Verify OCR text is preserved
                try:
                    with fitz.open(str(output_pdf)) as final_pdf:
                        total_chars = sum(len(page.get_text().strip()) for page in final_pdf)
                        if total_chars > 0:
                            logger.info(f"OCR preserved: {total_chars} characters found.")
                            return output_pdf
                        else:
                            logger.error("OCR LOST after compression!")
                            # Fall back to original OCR file
                            shutil.copy2(input_pdf, output_pdf)
                            return output_pdf
                except Exception as e:
                    logger.warning(f"Could not verify OCR after compression: {e}")
                    # Fall back to original OCR file
                    shutil.copy2(input_pdf, output_pdf)
                    return output_pdf
            else:
                # Delete temp file and try next option
                temp_output.unlink(missing_ok=True)
                
        except Exception as e:
            logger.error(f"Compression option {i+1} failed: {e}")
            # Continue to next option
            if temp_output.exists():
                temp_output.unlink(missing_ok=True)
    
    # If we're still over the target, we need to try a different approach
    logger.info("Standard compression options insufficient. Trying image recompression...")
    
    try:
        temp_output = output_pdf.with_suffix(".temp_recompress.pdf")
        
        with fitz.open(str(input_pdf)) as doc:
            # Extract and recompress images
            for page in doc:
                # Get all images on the page
                image_list = page.get_images(full=True)
                
                # Process each image
                for img in image_list:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_data = base_image["image"]
                    
                    # Convert to PIL Image
                    pil_image = Image.open(BytesIO(image_data))
                    
                    # Save as JPEG with aggressive compression
                    img_buffer = BytesIO()
                    pil_image.save(img_buffer, format="JPEG", quality=40, optimize=True, progressive=True)
                    img_buffer.seek(0)
                    
                    # Replace the image in the PDF
                    doc.update_image(xref, img_buffer.getvalue())
            
            # Save with maximum compression
            doc.save(
                str(temp_output),
                deflate=True,
                garbage=4,      # Maximum cleanup
                clean=True,     # Clean unused objects
                deflate_images=True,  # Compress images
                pretty=False,   # Don't pretty print
                encryption=fitz.PDF_ENCRYPT_KEEP
            )
        
        compressed_size = temp_output.stat().st_size
        size_increase_pct = (compressed_size - original_size) / original_size * 100
        
        logger.info(f"Image recompression: {compressed_size//1024} KB ({size_increase_pct:+.1f}% from original)")
        
        # If this is still too big, we'll have to use it anyway as it's the best we can do
        shutil.move(str(temp_output), str(output_pdf))
        
        # Verify OCR text is preserved
        try:
            with fitz.open(str(output_pdf)) as final_pdf:
                total_chars = sum(len(page.get_text().strip()) for page in final_pdf)
                if total_chars > 0:
                    logger.info(f"OCR preserved: {total_chars} characters found.")
                    return output_pdf
                else:
                    logger.error("OCR LOST after image recompression!")
                    # Fall back to original OCR file
                    shutil.copy2(input_pdf, output_pdf)
                    return output_pdf
        except Exception as e:
            logger.warning(f"Could not verify OCR after image recompression: {e}")
            # Fall back to original OCR file
            shutil.copy2(input_pdf, output_pdf)
            return output_pdf
            
    except Exception as e:
        logger.error(f"Image recompression failed: {e}")
        # Fall back to original OCR file
        shutil.copy2(input_pdf, output_pdf)
        return output_pdf

# ---------------- Main ----------------
def main():
    input_folder = Path(INPUT_DIR)
    output_folder = Path(OUTPUT_DIR)
    if not input_folder.exists():
        logger.error(f"Input folder '{INPUT_DIR}' not found.")
        sys.exit(1)
    output_folder.mkdir(exist_ok=True)
    pdf_files = list(input_folder.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files in '{INPUT_DIR}'")
        sys.exit(1)
    logger.info(f"Processing {len(pdf_files)} files with TrOCR: {TROCR_MODEL_NAME}")
    logger.info(f"Target: Final size as close as possible to original size (max 15% increase for OCR)")

    for pdf_path in pdf_files:
        original_size = pdf_path.stat().st_size
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {pdf_path.name} | Original: {original_size//1024} KB")

        # Step 1: OCR
        ocr_temp_path = output_folder / f"{pdf_path.stem}_ocr_temp.pdf"
        if not process_single_pdf_ocr(str(pdf_path), str(ocr_temp_path)):
            logger.error(f"Skipping {pdf_path.name} due to OCR failure.")
            continue

        # Step 2: Enhanced compression targeting original size with max 15% increase
        final_path = output_folder / f"{pdf_path.stem}_final.pdf"
        result_path = enhanced_compress_to_target_size(ocr_temp_path, final_path, original_size)

        if result_path.exists():
            final_size = result_path.stat().st_size
            size_increase = (final_size - original_size) / original_size * 100
            logger.info(f"SUCCESS: {result_path.name} | {final_size//1024} KB ({size_increase:+.1f}% increase from original)")
        else:
            logger.error(f"Failed to generate final output for {pdf_path.name}")

        # Cleanup temp file
        try:
            ocr_temp_path.unlink()
        except Exception as e:
            logger.warning(f"Could not delete temp file: {e}")

    logger.info(f"\nAll done! Output files in '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    main()
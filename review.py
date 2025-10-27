import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF

# Configuration
INPUT_DIR = "B"  # Directory with OCR-enhanced PDFs
OUTPUT_DIR = "C"  # Output directory for review images

# Position adjustment for OCR text layer
X_OFFSET = -10  # Pixels to shift text to the right
Y_OFFSET = 0  # Pixels to shift text up (negative value moves it up)

# Padding around text in OCR image
PADDING = 40  # Increased extra space around text

# Font size scaling factor
FONT_SCALE = 1.2  # Scale factor for font size to ensure readability

# Font cache to avoid repeated lookups
_font_cache = {}

def get_font(size, font_name=None):
    """Get a font of the given size with fallbacks."""
    key = (size, font_name)
    if key in _font_cache:
        return _font_cache[key]
    
    # Scale the font size
    scaled_size = int(size * FONT_SCALE)
    
    # Try to use the specified font first
    if font_name:
        try:
            font = ImageFont.truetype(font_name, scaled_size)
            _font_cache[key] = font
            return font
        except (OSError, FileNotFoundError):
            pass
    
    # Try FreeSans font (used in the OCR script)
    possible_fonts = [
        "fonts/FreeSans.ttf",  # Try local fonts directory first
        "FreeSans.ttf",
        "arial.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for font_path in possible_fonts:
        try:
            font = ImageFont.truetype(font_path, scaled_size)
         1 python review.py[?2004l
Found 1 OCR-enhanced PDF(s). Creating review images...

Using rawdict method for mg670_b1-001p001-015_test_copy_final.pdf
Using words method for mg670_b1-001p001-015_test_copy_final.pdf
Found 1471 text elements in mg670_b1-001p001-015_test_copy_final.pdf
Created review image: C/mg670_b1-001p001-015_test_copy_final_review.jpg

All done! Check the 'C/' folder for review images.
[1m[7m%[27m[1m[0m                                                                                                                                                                                                                           [0m[27m[24m[Jaweymouth@uidaho.edu@uofi-HN2J5TJCW6 opti-doc % [K[?2004hs)
            
            # Try different text extraction methods
            text_data = []
            
            # Method 1: Try rawdict first
            try:
                text_dict = page.get_text("rawdict")
                print(f"Using rawdict method for {Path(pdf_path).name}")
                
                for block in text_dict.get("blocks", []):
                    if "lines" not in block:
                        continue
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if not text:
                                continue
                            bbox = span.get("bbox", [0, 0, 0, 0])
                            font_size_pts = span.get("size", 12)
                            font_name = span.get("font", "FreeSans")
                            x0, y0, x1, y1 = bbox
                            
                            # Get the font for this span
                            font = get_font(font_size_pts, font_name)
                            
                            # Calculate text dimensions for positioning
                            temp_img = Image.new("RGB", (1, 1))
                            temp_draw = ImageDraw.Draw(temp_img)
                            text_bbox = temp_draw.textbbox((0, 0), text, font=font)
                            text_width = text_bbox[2] - text_bbox[0]
                            text_height = text_bbox[3] - text_bbox[1]
                            
                            # Get font metrics for baseline calculation
                            try:
                                font_metrics = font.getmetrics()
                                ascent = font_metrics[0] if font_metrics[0] else int(font_size_pts * 0.8)
                                descent = font_metrics[1] if font_metrics[1] else int(font_size_pts * 0.2)
                            except (AttributeError, IndexError):
                                ascent = int(font_size_pts * 0.8)
                                descent = int(font_size_pts * 0.2)
                            
                            # Calculate baseline position using the same formula as in the OCR script
                            y_baseline = pix.height - y1 + descent
                            
                            # Calculate adjusted position
                            adj_x0 = x0 + X_OFFSET
                            adj_y0 = y_baseline + Y_OFFSET
                            
                            text_data.append((text, adj_x0, adj_y0, font))
                            
            except Exception as e:
                print(f"Rawdict method failed for {Path(pdf_path).name}: {e}")
                text_data = []
                
                # Method 2: Try dict method
                try:
                    text_dict = page.get_text("dict")
                    print(f"Using dict method for {Path(pdf_path).name}")
                    
                    for block in text_dict.get("blocks", []):
                        if "lines" not in block:
                            continue
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                text = span.get("text", "").strip()
                                if not text:
                                    continue
                                bbox = span.get("bbox", [0, 0, 0, 0])
                                font_size_pts = span.get("size", 12)
                                font_name = span.get("font", "FreeSans")
                                x0, y0, x1, y1 = bbox
                                
                                # Get the font for this span
                                font = get_font(font_size_pts, font_name)
                                
                                # Calculate text dimensions for positioning
                                temp_img = Image.new("RGB", (1, 1))
                                temp_draw = ImageDraw.Draw(temp_img)
                                text_bbox = temp_draw.textbbox((0, 0), text, font=font)
                                text_width = text_bbox[2] - text_bbox[0]
                                text_height = text_bbox[3] - text_bbox[1]
                                
                                # Get font metrics for baseline calculation
                                try:
                                    font_metrics = font.getmetrics()
                                    ascent = font_metrics[0] if font_metrics[0] else int(font_size_pts * 0.8)
                                    descent = font_metrics[1] if font_metrics[1] else int(font_size_pts * 0.2)
                                except (AttributeError, IndexError):
                                    ascent = int(font_size_pts * 0.8)
                                    descent = int(font_size_pts * 0.2)
                                
                                # Calculate baseline position using the same formula as in the OCR script
                                y_baseline = pix.height - y1 + descent
                                
                                # Calculate adjusted position
                                adj_x0 = x0 + X_OFFSET
                                adj_y0 = y_baseline + Y_OFFSET
                                
                                text_data.append((text, adj_x0, adj_y0, font))
                                
                except Exception as e2:
                    print(f"Dict method failed for {Path(pdf_path).name}: {e2}")
                    text_data = []
                    
                    # Method 3: Try simple text extraction
                    try:
                        text_instances = page.get_text("blocks")
                        print(f"Using blocks method for {Path(pdf_path).name}")
                        
                        for block in text_instances:
                            if len(block) < 5:
                                continue
                            x0, y0, x1, y1, text = block[:5]
                            text = text.strip()
                            if not text:
                                continue
                            
                            # Estimate font size based on block height
                            font_size_pts = max(12, int(y1 - y0))
                            font = get_font(font_size_pts)
                            
                            # Calculate adjusted position
                            adj_x0 = x0 + X_OFFSET
                            adj_y0 = y0 + Y_OFFSET
                            
                            text_data.append((text, adj_x0, adj_y0, font))
                            
                    except Exception as e3:
                        print(f"Blocks method failed for {Path(pdf_path).name}: {e3}")
                        text_data = []
            
            # If we still don't have text data, try one more method
            if not text_data:
                try:
                    # Method 4: Get all text with positions
                    text_instances = page.get_text("words")
                    print(f"Using words method for {Path(pdf_path).name}")
                    
                    for word in text_instances:
                        if len(word) < 5:
                            continue
                        x0, y0, x1, y1, text = word[:5]
                        text = text.strip()
                        if not text:
                            continue
                        
                        # Estimate font size based on word height
                        font_size_pts = max(12, int(y1 - y0))
                        font = get_font(font_size_pts)
                        
                        # Calculate adjusted position
                        adj_x0 = x0 + X_OFFSET
                        adj_y0 = y0 + Y_OFFSET
                        
                        text_data.append((text, adj_x0, adj_y0, font))
                except Exception as e4:
                    print(f"Words method failed for {Path(pdf_path).name}: {e4}")
            
            # Calculate bounding box for all text with offsets applied
            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = float('-inf'), float('-inf')
            
            for text, x, y, font in text_data:
                # Get text dimensions
                temp_img = Image.new("RGB", (1, 1))
                temp_draw = ImageDraw.Draw(temp_img)
                text_bbox = temp_draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # Update bounding box
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + text_width)
                max_y = max(max_y, y + text_height)
            
            # If no text found, create empty OCR image
            if not text_data:
                print(f"No text found in {Path(pdf_path).name}")
                ocr_img = Image.new("RGB", (pix.width, pix.height), (255, 255, 255))
            else:
                print(f"Found {len(text_data)} text elements in {Path(pdf_path).name}")
                
                # Calculate OCR image dimensions with padding
                ocr_width = int(max_x - min_x + 2 * PADDING)
                ocr_height = int(max_y - min_y + 2 * PADDING)
                
                # Ensure minimum dimensions
                ocr_width = max(ocr_width, pix.width)
                ocr_height = max(ocr_height, pix.height)
                
                # Add extra buffer to ensure text doesn't get cut off
                ocr_width += int(PADDING * 0.5)
                ocr_height += int(PADDING * 0.5)
                
                # Create OCR image with calculated dimensions
                ocr_img = Image.new("RGB", (ocr_width, ocr_height), (255, 255, 255))
                ocr_draw = ImageDraw.Draw(ocr_img)
                
                # Draw text at adjusted positions relative to the new canvas
                for text, x, y, font in text_data:
                    # Adjust coordinates relative to the new canvas
                    draw_x = x - min_x + PADDING
                    draw_y = y - min_y + PADDING
                    
                    # Draw the text in black
                    ocr_draw.text(
                        (draw_x, draw_y),
                        text,
                        fill=(0, 0, 0),  # Black text
                        font=font
                    )
            
            # Determine dimensions for the combined image
            combined_width = pix.width + ocr_img.width
            combined_height = max(pix.height, ocr_img.height)
            
            # Create a new image to hold both images side by side
            combined_img = Image.new("RGB", (combined_width, combined_height), (255, 255, 255))
            
            # Paste the original image on the left
            combined_img.paste(img, (0, 0))
            
            # Paste the OCR text image on the right
            combined_img.paste(ocr_img, (pix.width, 0))
            
            # Add a dividing line between the two images
            draw_line = ImageDraw.Draw(combined_img)
            draw_line.line([(pix.width, 0), (pix.width, combined_height)], fill=(128, 128, 128), width=2)
            
            # Save the combined image
            combined_img.save(output_path, "JPEG", quality=95)
            print(f"Created review image: {output_path}")
            
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        import traceback
        traceback.print_exc()

def main():
    input_folder = Path(INPUT_DIR)
    output_folder = Path(OUTPUT_DIR)
    
    if not input_folder.exists():
        print(f"Input folder '{INPUT_DIR}' not found. Please run the OCR script first.")
        sys.exit(1)
    
    output_folder.mkdir(exist_ok=True)
    
    pdf_files = list(input_folder.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{INPUT_DIR}/'. Please run the OCR script first.")
        sys.exit(1)
    
    print(f"Found {len(pdf_files)} OCR-enhanced PDF(s). Creating review images...\n")
    
    for pdf_path in pdf_files:
        output_path = output_folder / f"{pdf_path.stem}_review.jpg"
        create_review_image(str(pdf_path), str(output_path))
    
    print("\nAll done! Check the 'C/' folder for review images.")

if __name__ == "__main__":
    main()
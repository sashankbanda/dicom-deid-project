import pydicom
import numpy as np
from PIL import Image, ImageDraw
import pytesseract
import re

# ⚠️ NOTE: You may need to specify the path to your tesseract executable.
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' 

def get_image_from_dicom(ds: pydicom.Dataset) -> Image.Image:
    """Converts DICOM pixel data to a Pillow Image object."""
    # Scale pixel data if necessary (especially for 16-bit data)
    pixel_array = ds.pixel_array
    
    # Simple check for 16-bit data (often found in medical images)
    if pixel_array.dtype in [np.int16, np.uint16]:
        # Normalize the pixel data to 8-bit for better OCR performance
        normalized_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255
        img_data = normalized_array.astype(np.uint8)
    else:
        img_data = pixel_array

    # Handle multi-frame or color images
    if img_data.ndim == 3:
        if ds.SamplesPerPixel == 3:
            # Color image
            img = Image.fromarray(img_data)
        else:
            # Multi-frame image (take the first frame)
            img = Image.fromarray(img_data[0])
    else:
        # Grayscale single-frame image
        img = Image.fromarray(img_data)
    
    return img.convert('L') # Convert to grayscale

def run_ocr_and_mask(ds: pydicom.Dataset, phis_to_mask: dict, output_path: str):
    """
    Runs OCR, identifies PHI, masks it, and saves the modified DICOM file.

    Args:
        ds: The pydicom Dataset.
        phis_to_mask: A dictionary of actual PHI values removed from metadata 
                      (e.g., {'PatientName': 'DOE^JOHN'}).
        output_path: Path to save the new de-identified DICOM file.
    """
    try:
        img = get_image_from_dicom(ds)
    except Exception as e:
        print(f"  Could not process pixel data: {e}")
        ds.save_as(output_path)
        return

    # Use Tesseract's image_to_data for bounding box information
    ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    
    # Convert image back to numpy array for masking
    img_array = np.array(img.convert('RGB')) 
    
    # 1. PHI Direct Match List (values from metadata)
    metadata_phis = [v for k, v in phis_to_mask.items() if v]

    # 2. PHI Pattern List (keywords that indicate PHI)
    # The list of fields you provided (StudyDate, PatientID, etc.) are the best
    # source for these keywords. We'll check for them appearing before data.
    phi_keywords = ['ID', 'NAME', 'DOB', 'DATE', 'TIME', 'ACCESSION']

    masked_count = 0
    
    for i in range(len(ocr_data['text'])):
        text = ocr_data['text'][i].strip().upper()
        conf = float(ocr_data['conf'][i])
        
        if not text or conf < 70: # Ignore low-confidence text
            continue

        is_phi = False
        
        # --- Identification Logic ---
        
        # A) Check against actual PHI values removed from metadata
        if any(phi_val.upper() in text for phi_val in metadata_phis):
            is_phi = True
            
        # B) Check for PHI keywords or common patterns
        if any(keyword in text for keyword in phi_keywords):
             is_phi = True

        # C) General check for strong identifiers (numbers/dates)
        if re.match(r'^\d{4}-\d{2}-\d{2}$', text) or re.match(r'^\d{6,}', text): # Simple date or long number
             is_phi = True

        # --- Masking Logic ---
        if is_phi:
            x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
            
            # Draw a black rectangle over the detected text area (add a small buffer)
            buffer = 5
            img_array[y-buffer:y+h+buffer, x-buffer:x+w+buffer] = [0, 0, 0] # Black color
            masked_count += 1
            
    if masked_count > 0:
        print(f"  Masked {masked_count} text blocks in the image.")
        
        # Convert masked array back to DICOM pixel data format
        final_img = Image.fromarray(img_array).convert('L')
        final_pixel_array = np.array(final_img).astype(ds.pixel_array.dtype)
        ds.PixelData = final_pixel_array.tobytes()
        ds.Rows, ds.Columns = final_pixel_array.shape
        
    # Save the final DICOM file
    ds.save_as(output_path)
    print(f"  Saved de-identified file to {output_path}")

import pydicom
import pydicom.uid
import os
from ocr_utils import run_ocr_and_mask

# --- Configuration ---
# Use r-strings for Windows paths to handle backslashes correctly
INPUT_DIR = r"D:\0000 study spacd\06.1 SEM8\00 internship\01 go zeal\02 projects\02 dicom-deid-project\input_dcm"
OUTPUT_DIR = r"D:\0000 study spacd\06.1 SEM8\00 internship\01 go zeal\02 projects\02 dicom-deid-project\output_dcm"

# DICOM Tags from your COMPLETE list, labeled for clarity (Hexadecimal Tag, Keyword)
PHI_TAGS = [
    # Patient Identifying Information
    (0x0010, 0x0010), # PatientName
    (0x0010, 0x0020), # PatientID
    (0x0010, 0x0030), # PatientBirthDate
    (0x0010, 0x0032), # PatientBirthTime
    (0x0010, 0x0040), # PatientSex
    (0x0010, 0x1010), # PatientAge
    (0x0010, 0x1000), # OtherPatientIDs
    (0x0010, 0x1001), # OtherPatientNames
    (0x0010, 0x1040), # PatientAddress
    (0x0010, 0x2150), # PatientTelephoneNumbers
    
    # Institution and Location Information
    (0x0008, 0x0080), # InstitutionName
    (0x0008, 0x0081), # InstitutionAddress
    (0x0008, 0x1040), # InstitutionalDepartmentName
    (0x0008, 0x1010), # StationName
    
    # Physician and Operator Names
    (0x0008, 0x0090), # ReferringPhysicianName
    (0x0008, 0x0094), # ReferringPhysicianTelephoneNumbers
    (0x0008, 0x0096), # RequestingPhysician
    (0x0008, 0x9000), # PhysiciansOfRecord
    (0x0008, 0x1050), # PerformingPhysicianName
    (0x0008, 0x1084), # NameOfPhysiciansReadingStudy
    (0x0008, 0x1070), # OperatorsName
    (0x0040, 0x0006), # ScheduledPerformingPhysicianName
    (0x0040, 0xA075), # VerifyingObserverName
    (0x0040, 0xA078), # VerifyingObserverIdentificationCodeSequence
    
    # Dates, Times, and Study Identifiers
    (0x0008, 0x0020), # StudyDate
    (0x0008, 0x0030), # StudyTime
    (0x0008, 0x0050), # AccessionNumber
    (0x0020, 0x0010), # StudyID
    (0x0020, 0x0011), # SeriesNumber
    
    # Device Identifiers
    (0x0018, 0x1000), # DeviceSerialNumber
    (0x0018, 0x1002), # DeviceSeriesNumber
    
    # Unique Identifiers (UIDs - MUST be replaced, not deleted)
    (0x0020, 0x000D), # StudyInstanceUID
    (0x0020, 0x000E)  # SeriesInstanceUID
]

# DICOM SOP Class UID for Secondary Capture (High-risk images)
SECONDARY_CAPTURE_UID = '1.2.840.10008.5.1.4.1.1.7' 

# --- Main Functions ---

def remove_metadata_phi(ds: pydicom.Dataset) -> dict:
    """Removes PHI from metadata and returns the original PHI values."""
    original_phis = {}
    
    for tag in PHI_TAGS:
        if tag in ds:
            # Get the tag keyword and value before modification
            tag_name = ds[tag].keyword
            original_phis[tag_name] = str(ds[tag].value) 
            
            # Rule: Replace UIDs/Numbers/IDs with new unique values or '0'
            if 'UID' in tag_name:
                 ds[tag].value = pydicom.uid.generate_uid()
            elif 'Number' in tag_name or 'ID' in tag_name:
                 ds[tag].value = '0' # Placeholder value
            # Rule: Delete other PHI tags (Names, Dates, Addresses, etc.)
            else:
                 del ds[tag]
                 
    # Final safety check: ensure critical UIDs are new
    if 'StudyInstanceUID' in ds:
        ds.StudyInstanceUID = pydicom.uid.generate_uid()
    if 'SeriesInstanceUID' in ds:
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        
    return original_phis

def should_run_ocr(ds: pydicom.Dataset) -> bool:
    """Implements the expanded, targeted filtering logic."""
    
    modality = ds.get('Modality', '').upper()
    study_desc = ds.get('StudyDescription', '').upper()
    series_desc = ds.get('SeriesDescription', '').upper()
    
    # 1. Check Modality (US, XA, CR, DX are highest risk)
    if modality in ['US', 'XA', 'CR', 'DX']: 
        return True
        
    # 2. Check for Secondary Capture Flag
    if ds.get('SOPClassUID') == SECONDARY_CAPTURE_UID:
        return True
        
    # 3. Check for 'PORTABLE' or 'SC' in description 
    if 'PORTABLE' in study_desc or 'PORTABLE' in series_desc or \
       'SCANNED' in study_desc or 'SCANNED' in series_desc:
        return True

    # 4. Check for Burned-In Annotation Flag (if present)
    if ds.get('BurnedInAnnotation', '').upper() == 'YES':
        return True

    return False

def process_dicom_files(input_dir: str, output_dir: str):
    """Iterates through files and applies de-identification."""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_list = [f for f in os.listdir(input_dir) if f.endswith('.dcm')]
    total_files = len(file_list)
    ocr_run_count = 0
    
    print(f"Processing {total_files} DICOM files...")
    
    for i, filename in enumerate(file_list):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        print(f"\n[{i+1}/{total_files}] Processing {filename}:")

        try:
            ds = pydicom.dcmread(input_path)
        except Exception as e:
            print(f"  Error reading DICOM file: {e}. Skipping.")
            continue
            
        # 1. Metadata De-identification
        original_phis = remove_metadata_phi(ds)
        
        # 2. Targeted OCR Filter Check
        if hasattr(ds, 'PixelData') and should_run_ocr(ds):
            print(f"  - Running OCR: High-risk file (Modality: {ds.get('Modality', 'N/A')}).")
            ocr_run_count += 1
            # 3. Identify and Mask PHI Text
            run_ocr_and_mask(ds, original_phis, output_path)
        else:
            # If no OCR needed, just save the metadata-cleaned file
            print("  - Skipping OCR: Low risk of burned-in PHI.")
            ds.save_as(output_path)

    print(f"\n--- Summary ---")
    print(f"Total Files Processed: {total_files}")
    print(f"OCR Engine Run Count: {ocr_run_count} ({ocr_run_count/total_files*100:.2f}%)")

if __name__ == '__main__':
    if not os.path.exists(INPUT_DIR):
        print(f"Input directory not found: {INPUT_DIR}")
        print("Please ensure the input directory is correctly set and exists.")
    else:
        process_dicom_files(INPUT_DIR, OUTPUT_DIR)

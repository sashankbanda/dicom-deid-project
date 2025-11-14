# DICOM PHI Redaction Project 🏥

This project implements a method for de-identifying DICOM medical images by removing Protected Health Information (PHI) from both the **metadata (header)** and the **pixel data (burned-in text)**. It uses a targeted approach to run the time-consuming Optical Character Recognition (OCR) only on high-risk images, maximizing efficiency.

## 🚀 Getting Started

Instructions on how to get the DICOM PHI Redaction Project up and running on your local machine.

### Prerequisites

Before running the scripts, you must install the following:

*   **Python 3.x**
*   **Tesseract OCR Engine:** Tesseract must be installed on your operating system and accessible via your system's PATH.

### Installation

Navigate to your project folder (e.g., `dicom-deid-project`) and install the required Python libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### ⚙️ Configuration

You must set the absolute paths for your input and output directories in the `deidentify_runner.py` file.

1.  Open `deidentify_runner.py`.
2.  Locate the Configuration section and update the `INPUT_DIR` and `OUTPUT_DIR` paths using Python's raw string format (`r"..."`):

```python
# --- Configuration ---
# Update these paths to match your system's location
INPUT_DIR = r"D:\0000 study spacd\06.1 SEM8\00 internship\01 go zeal\02 projects\02 dicom-deid-project\input_dcm"
OUTPUT_DIR = r"D:\0000 study spacd\06.1 SEM8\00 internship\01 go zeal\02 projects\02 dicom-deid-project\output_dcm"
# ... (rest of the file)
```

### 🏁 Execution

Run the de-identification process using the main runner script:

```bash
python deidentify_runner.py
```

## 📁 Project Structure

Ensure your file structure is organized as follows:

```
dicom-deid-project/
├── input_dcm/         <-- Place your original DICOM (.dcm) files here
├── output_dcm/        <-- De-identified files will be saved here
├── requirements.txt
├── deidentify_runner.py  # Main script to execute the pipeline
└── ocr_utils.py          # Utility functions for image processing and masking
```
# Minneapolis City Directory OCR Project

A WIP Python pipeline for digitizing historical Minneapolis city directories from 1900, converting scanned images into structured, searchable data.

## 🎯 Project Overview

This project automates the extraction and digitization of historical city directory data, transforming scanned directory pages into machine-readable text and structured data.

### What This Project Does

<table>
<tr>
<th>Original Scanned Document</th>
<th>OCR Transcribed Text</th>
</tr>
<tr>
<td width="50%">
<img src="example_images/scanned_document.jpg" alt="Scanned Document" width="100%">
</td>
<td width="50%" style="font-family: monospace; white-space: pre-line; font-size: 10px;">
Bergman Emma, cook 416 14th av s e, rms 418 same <br>
Brick lab H C Akeley Lbr Co, rms 3322 n 3d. <br>
Brika, dom 2705 Hennepin av. <br>
Fred J, painter, r 905 13th av s. <br>
Prederick, cooper Minn L O Paint Co r 243 lith av s. <br>
Freda, smstrs Madame Boyd, b flat 2 60 Island av w. <br>
Hannah, dom 2600 3d av gs. <br>
Harris (Rosen: & Bergman), r 716 Lyn dale av n. <br>
Jennie, dom 1928 Stevens av. <br>
John, dairy 2810 n e Johnson, r same. <br>
John, lab H C: Akeley Lbr Co, rms 3322 <br>
John A, car repr C M & St.P Ry. <br>
John A, foreman Ankeny Blev, r 1911 1th av s. <br>
John P, cabtmkr Larson & Nelander, r 436 ne Jefferson. <br>
Julia, smstrs Madame Boyd, b flat 2 60 Island av w.<br>
Louis, harnessmkr L Laramee & Co, b 234 14th av s. <br>
Magnus, mach opr Am Bridge Co, r 1417 s 2d. <br>
Margaret, smstrs M & I Moeser, b 929 Nicollet av. <br>
Martin, mach, r 1417 s 2d. <br>
Mary, dom, rms 515 s 4th. <br>
Mary E (wid John), r 1051 n 3d. <br>
Mathilda, waiter 138% s 4th. <br>
Maud H, clk, b 2115 Bloomington av. <br>
Nels A, harnessmkr S B Loye & Sons rms 1007 Washn av s. <br>
Nils, lab, b 2017 Washn av s. <br>
Oscar F (P O Hammer & Co), rms 414 n e Jefferson. <br>
Wm, lab, r 1911 Bluff. <br>
Wm, cooper Frank, Tupa, r 2201 n e Marshall. <br>
Bergmark Adolph, clk C 8S Brackett, b 34 <br>
Olga A (wid Charles O), mach opr E R <br>
Ely, b 509 n 4th <br>
Sylvia, student, b 509 n 4th. <br>
Bergquist, see also Barquist and Bjurquist. <br>
Albert, lab, b 521 18th av n. <br>
Alma A, mach opr Sterling Mnfg Co, b 700 5th av n. <br>
Amelia (wid Emil), r 309 Plymouth av. <br>
...
</td>
</tr>
</table>

1. **Smart Column Detection**: Automatically identifies and separates text columns from scanned directory pages
2. **Advanced OCR Processing**: Extracts text using Tesseract with custom preprocessing for historical documents
3. **Quality Evaluation**: Provides detailed accuracy metrics and error analysis
4. **Structured Output**: Converts raw text into organized, searchable formats

## 📊 Current Performance
- Comparing 832 lines, to their ground truth, yields:
   - **Character Accuracy**: 96.4%
   - **Word Accuracy**: 83.5%
- Successfully processes both left and right page formats
- Handles advertisements and irregular layouts 

## 🚀 Quick Start

### Prerequisites

```bash
# Install Python dependencies
pip install opencv-python numpy matplotlib scipy pathlib pytesseract Pillow jiwer

# Install Tesseract OCR
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
# macOS: brew install tesseract
# Ubuntu: sudo apt install tesseract-ocr
```

### Basic Usage

1. **Place your directory images** in a `test_images/` folder
2. **Extract columns** from scanned pages:
   ```bash
   python Extract_Columns.py
   ```
3. **Run OCR** on extracted columns:
   ```bash
   python OCR_Col_to_Text.py
   ```
4. **Evaluate results** (optional):
   ```bash
   python OCR_Evaluation.py ground_truth_dir/ ocr_output_dir/
   ```

## 📁 Project Structure

```
├── Extract_Columns.py          # Main column detection and extraction
├── OCR_Col_to_Text.py         # OCR processing with noise reduction
├── OCR_Evaluation.py          # Accuracy evaluation and error analysis
├── test_images/               # Input directory images
├── extracted_columns/         # Output: separated column images
├── ocr_text_files/           # Output: extracted text files
└── Documentation.docx         # Detailed progress and findings
```

## 🔧 Core Components

### 1. Column Extraction (`Extract_Columns.py`)

**What it does**: Like a digital librarian that knows how to read old book layouts, this component automatically identifies where text columns begin and end on each page.

- **Structural Line Detection**: Uses computer vision to find column separators and page boundaries
- **Page Type Recognition**: Distinguishes between left pages (with ads) and right pages
- **Smart Cropping**: Extracts clean column images for optimal OCR processing

### 2. OCR Processing (`OCR_Col_to_Text.py`)

**What it does**: The digital transcriptionist that converts images to text, with special handling for 1900s typography and common scanning artifacts.

Key features:
- **Preprocessing**: Noise reduction specifically tuned for historical documents
- **Text Cleaning**: Removes scanning artifacts while preserving genuine punctuation
- **Line Continuation Handling**: Properly combines multi-line directory entries
- **Page Combination**: Merges left and right columns into complete page files

### 3. Quality Evaluation (`OCR_Evaluation.py`)

**What it does**: The quality inspector that tells you exactly how well the OCR performed and what needs improvement.

Provides:
- **Detailed Error Analysis**: Character and word-level accuracy metrics
- **Common Error Patterns**: Identifies systematic OCR mistakes
- **Improvement Suggestions**: Specific fixes for the most frequent errors
   - ex: OCR Mistake: 'r'  -> ground truth: 'b' (15 times)

## 🎨 Example Output

### Input: Scanned Directory Page
Raw scanned image with multiple columns, advertisements, and 1900s typography.

### Output: Clean Text
```
Anderson Carl, lab, h 1247 Van Buren
Anderson Charles, carp, h 1626 4th st ne
Anderson Christina Mrs, h 1324 5th st se
Anderson Edward, lab Washburn Crosby Co, h 1435 University av se
```

## 📈 Technical Approach

The pipeline uses a three-stage approach analogous to how a human would process these documents:

1. **Visual Analysis** (Column Extraction)
   - Detects structural lines using Hough transforms
   - Identifies page layout patterns
   - Separates content from advertisements

2. **Text Recognition** (OCR Processing)
   - Applies noise reduction and contrast enhancement
   - Uses Tesseract with optimized settings for historical text
   - Implements smart text cleaning algorithms

3. **Quality Assurance** (Evaluation)
   - Compares against ground truth using edit distance metrics
   - Identifies systematic error patterns
   - Suggests targeted improvements

## 🛠 Configuration Options

### Column Extraction Parameters
```python
detector = StructuralLineDetector(
    blur_kernel=3,          # Noise reduction strength
    canny_low=100,          # Edge detection sensitivity
    canny_high=200,
    hough_threshold=1200,   # Line detection threshold
    max_line_gap=10,        # Maximum gap in detected lines
    line_thickness=4        # Visualization line thickness
)
```

### OCR Processing Options
```python
ocr = SimpleOCR(
    output_dir="ocr_text_files",
    show_preprocessing=False  # Set True to visualize processing steps
)
```

## 📊 Performance Metrics

Based on testing with 1900 Minneapolis directory pages:

| Metric | Value | Notes |
|--------|-------|-------|
| Character Error Rate (CER) | 3.6% | |
| Word Error Rate (WER) | 16.5% | |
| Average Word Accuracy | 83.5% | Most OCR on historical docs acheive 70-85% |
| Average Character Accuracy | 96.4 % | Industry standard 85-95% for historical material |
## 🔍 Common Issues & Solutions

### Low OCR Accuracy?
- **Check image quality**: Ensure 300+ DPI resolution
- **Verify preprocessing**: Enable `show_preprocessing=True` to debug
- **Review error patterns**: Use evaluation script to identify systematic issues

### Column Detection Problems?
- **Adjust thresholds**: Modify `hough_threshold` in StructuralLineDetector
- **Check page type detection**: Verify left/right page classification
- **Manual inspection**: Enable line visualization for debugging

### Text Cleaning Issues?
- **Review cleaning rules**: Check `clean_text()` function in OCR_Col_to_Text.py
- **Add custom fixes**: Use evaluation output to identify needed corrections
- **Adjust continuation logic**: Modify `is_continuation_line()` for different formats

## 🚧 Current Limitations

- **Single directory format**: Optimized for 1900 Minneapolis directories
- **English text only**: No multi-language support
- **Manual quality review**: High-accuracy applications may need human verification
- **Fixed page layouts**: Assumes two-column directory format

## 📚 Documentation

- `Documentation.docx`: Detailed daily progress log with technical insights
- `Approach_to_Phase_1.docx`: Original project planning and goals
- Code comments: Extensive inline documentation in all Python files

## 🤝 Contributing

This project welcomes contributions! Areas where help is especially valuable:

- **Historical expertise**: Knowledge of directory formats from different eras
- **OCR optimization**: Experience with Tesseract configuration
- **Computer vision**: Improvements to column detection algorithms
- **Quality assurance**: Testing with directories from other cities/years

## 📄 License

    MIT License

## 🙏 Acknowledgments

- Historical images sourced from [specify source]
- Built using Tesseract OCR, OpenCV, and other open-source libraries
- Inspired by digital humanities and genealogical research communities

---

*This project transforms static historical documents into dynamic, searchable data—bridging the gap between past and present through the power of modern OCR technology.*
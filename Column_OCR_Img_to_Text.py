"""
Simple OCR Script for City Directory Images

Basic script that:
1. Preprocesses images for better OCR
2. Runs Tesseract OCR
3. Saves results to text files
"""

import cv2
import numpy as np
import pytesseract
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Uncomment and adjust this line if Tesseract isn't in your PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class SimpleOCR:
    def __init__(self, output_dir="ocr_text_files", show_preprocessing=False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.show_preprocessing = show_preprocessing
    
    def preprocess_image(self, image_path):
        """Improved denoising to remove salt and pepper artifacts with visualization."""
        # Load image in grayscale
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        original = img.copy()
        
        # Apply median filter to remove salt and pepper noise
        denoised = cv2.medianBlur(img, 5)
        
        # Apply bilateral filter to further smooth while preserving edges
        denoised2 = cv2.bilateralFilter(denoised, 9, 75, 75)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised2)
        
        # Apply threshold to get clean black/white image
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological opening to remove small noise specks
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Show preprocessing steps if requested
        if self.show_preprocessing:
            self._show_preprocessing_steps(
                image_path, original, denoised, denoised2, enhanced, binary, cleaned
            )
        
        return cleaned
    
    def _show_preprocessing_steps(self, image_path, original, denoised, denoised2, enhanced, binary, cleaned):
        """Display preprocessing steps for debugging."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Preprocessing Steps: {Path(image_path).name}', fontsize=14)
        
        # Show each step
        steps = [
            (original, 'Original'),
            (denoised, 'Median Filter (Remove Salt/Pepper)'),
            (denoised2, 'Bilateral Filter (Smooth)'),
            (enhanced, 'CLAHE (Enhance Contrast)'),
            (binary, 'Binary Threshold'),
            (cleaned, 'Morphological Opening (Final)')
        ]
        
        for i, (img, title) in enumerate(steps):
            row, col = i // 3, i % 3
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].set_title(title)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Ask user if they want to continue
        input("\nPress Enter to continue to next image...")
        plt.close()
        
    def clean_text(self, text):
        """Clean OCR text by removing noise symbols while preserving indentation."""
        if not text:
            return text
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            if not line.strip():  # Skip empty lines
                continue
            
            # First, clean the entire line of noise characters
            cleaned_line = line
            allowed_chars_leading = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

            while cleaned_line and cleaned_line[0] not in allowed_chars_leading:
                cleaned_line = cleaned_line[1:]

            # Clean from the end, but only keep periods that follow letters/numbers
            while cleaned_line:
                last_char = cleaned_line[-1]
                
                # If it's a letter or number, keep it and stop
                if last_char.isalnum():
                    break
                
                # If it's a period, only keep it if the previous character is a letter/number
                elif last_char == '.' and len(cleaned_line) > 1 and cleaned_line[-2].isalnum():
                    break
                
                # Otherwise, remove it
                else:
                    cleaned_line = cleaned_line[:-1]
            
            # Now get the clean content
            content = cleaned_line.strip()
            
            if content:
                cleaned_lines.append(content)
        
        # Combine continuation lines with previous entries (OUTSIDE the loop)
        combined_lines = []
        for line in cleaned_lines:
            if combined_lines and self.is_continuation_line(line):
                # Combine with previous line
                combined_lines[-1] = combined_lines[-1] + ' ' + line.strip()
            else:
                # Start new line
                combined_lines.append(line)

        return '\n'.join(combined_lines)  # Return combined_lines, not cleaned_lines

    def is_continuation_line(self, line):
        """Check if a line is a continuation of the previous entry."""
        first_char = line.strip()[0] if line.strip() else ''

        return (
            first_char.isdigit() or          # Starts with number (address)
            first_char.islower() or          # Starts with lowercase letter (continuation)
            len(line.strip()) < 16           # Very short lines are likely continuations
        )

    def extract_text(self, image_path):
        """Extract text using Tesseract OCR and clean artifacts."""
        # Preprocess the image
        processed_img = self.preprocess_image(image_path)
        
        # Convert to PIL Image
        pil_img = Image.fromarray(processed_img)
        
        # Run OCR with basic configuration
        text = pytesseract.image_to_string(pil_img, config= '--psm 6 --oem 3')
        
        # Clean the text to remove noise symbols
        cleaned_text = self.clean_text(text)
        
        return cleaned_text 
    
    def save_text_file(self, image_path, text):
        """Save OCR text to a .txt file."""
        image_name = Path(image_path).stem
        output_file = self.output_dir / f"{image_name}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Saved: {output_file.name}")
    
    def process_directory(self, input_dir, pattern="*_col.jpg"):
        """Process all images in a directory and combine by page."""
        input_path = Path(input_dir)
        image_files = list(input_path.glob(pattern))
        
        if not image_files:
            print(f"No files found matching '{pattern}' in {input_dir}")
            return
        
        print(f"Found {len(image_files)} images to process")
        print(f"Output directory: {self.output_dir}")
        print("-" * 50)
        
        # Group files by page (first two parts before underscores)
        pages = {}
        for image_file in image_files:
            # Extract page identifier (e.g., "1900_0209" from "1900_0209_left_col.jpg")
            name_parts = image_file.stem.split('_')
            if len(name_parts) >= 3:
                page_id = '_'.join(name_parts[:2])  # First two parts
                column_type = name_parts[2]  # "left" or "right"
                
                if page_id not in pages:
                    pages[page_id] = {}
                pages[page_id][column_type] = image_file
        
        print(f"Found {len(pages)} pages to process:")
        for page_id, columns in pages.items():
            col_types = list(columns.keys())
            print(f"  {page_id}: {col_types}")
        print("-" * 50)
        
        # Process each page
        successful_pages = 0
        for page_id, columns in sorted(pages.items()):
            print(f"\nProcessing page: {page_id}")
            
            # Extract text from left column
            left_text = ""
            if 'left' in columns:
                print(f"  Processing left column: {columns['left'].name}")
                left_text = self.extract_text(columns['left'])
                if left_text:
                    preview = left_text[:100].replace('\n', ' ')
                    print(f"    Left preview: {preview}...")
            
            # Extract text from right column  
            right_text = ""
            if 'right' in columns:
                print(f"  Processing right column: {columns['right'].name}")
                right_text = self.extract_text(columns['right'])
                if right_text:
                    preview = right_text[:100].replace('\n', ' ')
                    print(f"    Right preview: {preview}...")
            
            # Combine texts and save
            if left_text or right_text:
                self.save_combined_page(page_id, left_text, right_text)
                successful_pages += 1
            else:
                print(f"    Warning: No text extracted for page {page_id}")
        
        print("-" * 50)
        print(f"Completed! Processed {successful_pages} pages")
        print(f"Combined page files saved in: {self.output_dir}")
        
        return pages
    
    def save_combined_page(self, page_id, left_text, right_text):
        """Save combined left and right column text to a single page file."""
        output_file = self.output_dir / f"{page_id}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            if left_text:
                f.write(left_text)
                f.write("\n")
            
            if right_text:
                f.write(right_text)
        
        print(f"    Saved combined page: {output_file.name}")
    
    def process_image(self, image_path):
        """Process a single image (used by the combined page processing)."""
        try:
            # Extract text
            text = self.extract_text(image_path)
            return text
            
        except Exception as e:
            print(f"    Error processing {image_path}: {e}")
            return None


def main():
    """Main function to run the OCR script."""
    # Initialize OCR processor with visualization option
    # Set show_preprocessing=True to see the image processing steps
    ocr = SimpleOCR(show_preprocessing=False)  # Change to False to disable visualization
    
    # Process all column images in the extracted_columns directory
    ocr.process_directory("extracted_columns")


if __name__ == "__main__":
    main()
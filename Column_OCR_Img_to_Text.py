"""
Simple OCR Script for City Directory Images

Basic script that:
1. Preprocesses images for better OCR
2. Runs Tesseract OCR
3. Saves results to text files

Start simple, improve incrementally.
"""

import cv2
import numpy as np
import pytesseract
from pathlib import Path
from PIL import Image

# Uncomment and adjust this line if Tesseract isn't in your PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class SimpleOCR:
    def __init__(self, output_dir="ocr_text_files"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def preprocess_image(self, image_path):
        """Basic image preprocessing for better OCR."""
        # Load image in grayscale
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img)
        
        # Apply threshold to get clean black/white image
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def extract_text(self, image_path):
        """Extract text using Tesseract OCR."""
        # Preprocess the image
        processed_img = self.preprocess_image(image_path)
        
        # Convert to PIL Image
        pil_img = Image.fromarray(processed_img)
        
        # Run OCR with basic configuration
        text = pytesseract.image_to_string(pil_img, config='--psm 6')
        
        return text
    
    def save_text_file(self, image_path, text):
        """Save OCR text to a .txt file."""
        image_name = Path(image_path).stem
        output_file = self.output_dir / f"{image_name}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Saved: {output_file.name}")
    
    def process_image(self, image_path):
        """Process a single image."""
        print(f"Processing: {Path(image_path).name}")
        
        try:
            # Extract text
            text = self.extract_text(image_path)
            
            # Save to file
            self.save_text_file(image_path, text)
            
            # Show preview of extracted text
            preview = text[:200].replace('\n', ' ')
            print(f"Preview: {preview}...")
            
            return text
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def process_directory(self, input_dir, pattern="*_col.jpg"):
        """Process all images in a directory."""
        input_path = Path(input_dir)
        image_files = list(input_path.glob(pattern))
        
        if not image_files:
            print(f"No files found matching '{pattern}' in {input_dir}")
            return
        
        print(f"Found {len(image_files)} images to process")
        print(f"Output directory: {self.output_dir}")
        print("-" * 50)
        
        results = []
        for image_file in sorted(image_files):
            text = self.process_image(image_file)
            if text:
                results.append((image_file.name, text))
        
        print("-" * 50)
        print(f"Completed! Processed {len(results)} images")
        print(f"Text files saved in: {self.output_dir}")
        
        return results


def main():
    """Main function to run the OCR script."""
    # Initialize OCR processor
    ocr = SimpleOCR()
    
    # Process all column images in the extracted_columns directory
    ocr.process_directory("extracted_columns")


if __name__ == "__main__":
    main()
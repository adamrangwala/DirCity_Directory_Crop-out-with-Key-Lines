"""
City Directory OCR Script with Format Preservation

This script processes extracted city directory column images using Tesseract OCR
while preserving the original formatting, indentation, and line structure.
"""

import cv2
import numpy as np
import pytesseract
from pathlib import Path
import json
import re
from PIL import Image
import argparse

class CityDirectoryOCR:
    """OCR processor for city directory columns with format preservation."""
    
    def __init__(self, output_dir="ocr_text_files"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Tesseract configuration for better text recognition
        self.tesseract_config = '--oem 3 --psm 6 -c preserve_interword_spaces=1'
        
        # Regular expressions for cleaning common OCR errors
        self.cleanup_patterns = [
            (r'\s+', ' '),  # Multiple spaces to single space
            (r'([a-z])\s+([A-Z])', r'\1\2'),  # Fix broken words like "J ohn" -> "John"
            (r'(\d)\s+(\d)', r'\1\2'),  # Fix broken numbers
            (r'\s*,\s*', ', '),  # Normalize comma spacing
            (r'\s*\.\s*', '. '),  # Normalize period spacing
        ]
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for better OCR results.
        Think of this like adjusting a photo before reading - we enhance contrast
        and clarity so the text recognition works better.
        """
        # Load image
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img)
        
        # Denoise the image
        denoised = cv2.medianBlur(enhanced, 3)
        
        # Apply threshold to get clean black/white image
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def extract_text_with_layout(self, image_path):
        """
        Extract text while preserving layout and indentation.
        Uses Tesseract's detailed output to maintain spatial relationships.
        """
        # Preprocess image
        processed_img = self.preprocess_image(image_path)
        
        # Convert to PIL Image for pytesseract
        pil_img = Image.fromarray(processed_img)
        
        # Get detailed OCR data with bounding boxes
        ocr_data = pytesseract.image_to_data(
            pil_img, 
            config=self.tesseract_config,
            output_type=pytesseract.Output.DICT
        )
        
        # Group text by lines based on y-coordinates
        lines = self._group_text_by_lines(ocr_data)
        
        # Process each line to preserve indentation
        formatted_lines = []
        for line_data in lines:
            formatted_line = self._format_line_with_indentation(line_data, processed_img.shape[1])
            if formatted_line.strip():  # Only add non-empty lines
                formatted_lines.append(formatted_line)
        
        return formatted_lines
    
    def _group_text_by_lines(self, ocr_data):
        """Group OCR words into lines based on y-coordinates."""
        words_with_positions = []
        
        for i in range(len(ocr_data['text'])):
            if int(ocr_data['conf'][i]) > 30:  # Only confident detections
                word_data = {
                    'text': ocr_data['text'][i],
                    'left': ocr_data['left'][i],
                    'top': ocr_data['top'][i],
                    'width': ocr_data['width'][i],
                    'height': ocr_data['height'][i],
                    'conf': ocr_data['conf'][i]
                }
                words_with_positions.append(word_data)
        
        # Sort by y-coordinate (top)
        words_with_positions.sort(key=lambda x: x['top'])
        
        # Group words into lines (words with similar y-coordinates)
        lines = []
        current_line = []
        line_tolerance = 15  # pixels
        
        for word in words_with_positions:
            if not current_line:
                current_line = [word]
            else:
                # Check if word is on the same line
                avg_top = sum(w['top'] for w in current_line) / len(current_line)
                if abs(word['top'] - avg_top) <= line_tolerance:
                    current_line.append(word)
                else:
                    # Start new line
                    if current_line:
                        # Sort current line by x-coordinate
                        current_line.sort(key=lambda x: x['left'])
                        lines.append(current_line)
                    current_line = [word]
        
        # Don't forget the last line
        if current_line:
            current_line.sort(key=lambda x: x['left'])
            lines.append(current_line)
        
        return lines
    
    def _format_line_with_indentation(self, line_words, image_width):
        """Format a line of words while preserving indentation."""
        if not line_words:
            return ""
        
        # Calculate indentation based on leftmost word position
        leftmost_pos = min(word['left'] for word in line_words)
        
        # Convert pixel position to approximate character indentation
        # Assume average character width of ~12 pixels
        char_width = 12
        indent_chars = max(0, leftmost_pos // char_width)
        
        # Build the line text
        line_text = ""
        last_right = 0
        
        for word in line_words:
            word_text = word['text'].strip()
            if not word_text:
                continue
                
            # Add spaces between words based on their positions
            if last_right > 0:
                gap = word['left'] - last_right
                spaces = max(1, gap // char_width)
                line_text += " " * min(spaces, 5)  # Limit excessive spacing
            
            line_text += word_text
            last_right = word['left'] + word['width']
        
        # Add indentation
        indentation = " " * min(indent_chars, 20)  # Limit excessive indentation
        
        return indentation + line_text.strip()
    
    def clean_ocr_text(self, text_lines):
        """Clean common OCR errors while preserving structure."""
        cleaned_lines = []
        
        for line in text_lines:
            cleaned_line = line
            
            # Apply cleanup patterns
            for pattern, replacement in self.cleanup_patterns:
                cleaned_line = re.sub(pattern, replacement, cleaned_line)
            
            # Preserve leading whitespace but clean the rest
            leading_spaces = len(line) - len(line.lstrip())
            cleaned_content = cleaned_line.strip()
            
            if cleaned_content:
                final_line = " " * leading_spaces + cleaned_content
                cleaned_lines.append(final_line)
        
        return cleaned_lines
    
    def process_single_image(self, image_path):
        """Process a single column image and return OCR results."""
        print(f"Processing: {Path(image_path).name}")
        
        try:
            # Extract text with layout preservation
            raw_lines = self.extract_text_with_layout(image_path)
            
            # Clean OCR errors
            cleaned_lines = self.clean_ocr_text(raw_lines)
            
            return {
                'image_path': str(image_path),
                'raw_lines': raw_lines,
                'cleaned_lines': cleaned_lines,
                'line_count': len(cleaned_lines)
            }
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def process_directory(self, input_dir, file_pattern="*_col.jpg"):
        """Process all column images in a directory."""
        input_path = Path(input_dir)
        image_files = list(input_path.glob(file_pattern))
        
        if not image_files:
            print(f"No files found matching pattern '{file_pattern}' in {input_dir}")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        all_results = {}
        
        for image_file in sorted(image_files):
            result = self.process_single_image(image_file)
            if result:
                all_results[image_file.name] = result
                
                # Save individual text file
                self._save_text_file(image_file, result['cleaned_lines'])
        
        # Save combined results
        self._save_combined_results(all_results)
        
        print(f"\nProcessing complete! Results saved in: {self.output_dir}")
        return all_results
    
    def _save_text_file(self, image_file, text_lines):
        """Save OCR results as a text file."""
        output_file = self.output_dir / f"{image_file.stem}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in text_lines:
                f.write(line + '\n')
    
    def _save_combined_results(self, all_results):
        """Save combined results in JSON and text formats."""
        # Save detailed JSON
        json_file = self.output_dir / "ocr_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # Save combined text file
        combined_file = self.output_dir / "combined_text.txt"
        with open(combined_file, 'w', encoding='utf-8') as f:
            for filename, result in all_results.items():
                f.write(f"\n{'='*60}\n")
                f.write(f"FILE: {filename}\n")
                f.write(f"{'='*60}\n")
                for line in result['cleaned_lines']:
                    f.write(line + '\n')


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='OCR City Directory Columns')
    parser.add_argument('input_dir', help='Directory containing column images')
    parser.add_argument('--output', '-o', default='ocr_text_files', help='Output directory')
    parser.add_argument('--pattern', '-p', default='*_col.jpg', help='File pattern to match')
    
    args = parser.parse_args()
    
    # Initialize OCR processor
    ocr_processor = CityDirectoryOCR(output_dir=args.output)
    
    # Process all images
    results = ocr_processor.process_directory(args.input_dir, args.pattern)
    
    if results:
        print(f"\nSummary:")
        total_lines = sum(result['line_count'] for result in results.values())
        print(f"- Processed {len(results)} images")
        print(f"- Extracted {total_lines} lines of text")
        print(f"- Results saved to: {args.output}")


if __name__ == "__main__":
    # If running without command line args, use current directory
    import sys
    if len(sys.argv) == 1:
        # Create OCR processor that saves to 'ocr_text_files' folder in current directory
        ocr_processor = CityDirectoryOCR(output_dir="ocr_text_files")
        ocr_processor.process_directory("extracted_columns")
    else:
        main()
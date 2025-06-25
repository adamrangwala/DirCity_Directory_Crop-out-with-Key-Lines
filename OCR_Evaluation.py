#!/usr/bin/env python3
"""
Simple OCR Evaluation using jiwer library
Install with: pip install jiwer
"""

import sys
import jiwer

def evaluate_ocr(ground_truth_file, ocr_output_file):
    """Evaluate OCR accuracy using jiwer library."""
    
    # Read files
    try:
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            ground_truth = f.read()
        with open(ocr_output_file, 'r', encoding='utf-8') as f:
            ocr_output = f.read()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Calculate metrics
    wer = jiwer.wer(ground_truth, ocr_output)
    cer = jiwer.cer(ground_truth, ocr_output)
    
    # Display results
    print("OCR Evaluation Results")
    print("=" * 30)
    print(f"Word Error Rate (WER):      {wer:.3f} ({wer*100:.1f}%)")
    print(f"Character Error Rate (CER): {cer:.3f} ({cer*100:.1f}%)")
    print(f"Word Accuracy:              {(1-wer)*100:.1f}%")
    print(f"Character Accuracy:         {(1-cer)*100:.1f}%")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python OCR_Evaluation.py ground_truth.txt ocr_output.txt")
        sys.exit(1)
    
    evaluate_ocr(sys.argv[1], sys.argv[2])
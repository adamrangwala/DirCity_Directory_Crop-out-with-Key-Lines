#!/usr/bin/env python3
"""
Detailed OCR Evaluation using jiwer library with error analysis
Install with: pip install jiwer
"""

import sys
import jiwer
import difflib
from collections import Counter

def detailed_error_analysis(ground_truth_file, ocr_output_file, max_lines_to_show=10):
    """Evaluate OCR accuracy with detailed error analysis using jiwer library."""
    
    # Read files
    try:
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            ground_truth = f.read()
        with open(ocr_output_file, 'r', encoding='utf-8') as f:
            ocr_output = f.read()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Calculate basic metrics
    wer = jiwer.wer(ground_truth, ocr_output)
    cer = jiwer.cer(ground_truth, ocr_output)
    
    # Display basic results
    print("OCR Evaluation Results")
    print("=" * 50)
    print(f"Word Error Rate (WER):      {wer:.3f} ({wer*100:.1f}%)")
    print(f"Character Error Rate (CER): {cer:.3f} ({cer*100:.1f}%)")
    print(f"Word Accuracy:              {(1-wer)*100:.1f}%")
    print(f"Character Accuracy:         {(1-cer)*100:.1f}%")
    
    # Detailed error analysis
    print(f"\nDetailed Error Analysis (showing first {max_lines_to_show} differing lines)")
    print("=" * 70)
    
    # Split into lines for comparison
    gt_lines = ground_truth.split('\n')
    ocr_lines = ocr_output.split('\n')
    
    # Track error patterns
    char_substitutions = Counter()
    word_substitutions = Counter()
    lines_with_errors = 0
    
    # Compare line by line
    for i, (gt_line, ocr_line) in enumerate(zip(gt_lines, ocr_lines)):
        if gt_line.strip() != ocr_line.strip():
            lines_with_errors += 1
            
            # Only show first max_lines_to_show errors for readability
            if lines_with_errors <= max_lines_to_show:
                print(f"\n--- Line {i+1} difference ---")
                print(f"TRUTH: '{gt_line}'")
                print(f"OCR:   '{ocr_line}'")
                
                # Show character-by-character differences
                differ = difflib.SequenceMatcher(None, gt_line, ocr_line)
                changes = []
                for tag, i1, i2, j1, j2 in differ.get_opcodes():
                    if tag == 'replace':
                        gt_part = gt_line[i1:i2]
                        ocr_part = ocr_line[j1:j2]
                        changes.append(f"'{gt_part}' → '{ocr_part}'")
                        char_substitutions[f"{gt_part}→{ocr_part}"] += 1
                    elif tag == 'delete':
                        deleted = gt_line[i1:i2]
                        changes.append(f"DELETED: '{deleted}'")
                        char_substitutions[f"{deleted}→(deleted)"] += 1
                    elif tag == 'insert':
                        inserted = ocr_line[j1:j2]
                        changes.append(f"INSERTED: '{inserted}'")
                        char_substitutions[f"(none)→{inserted}"] += 1
                
                if changes:
                    print(f"Changes: {', '.join(changes)}")
            
            # Track word-level substitutions
            gt_words = gt_line.split()
            ocr_words = ocr_line.split()
            word_differ = difflib.SequenceMatcher(None, gt_words, ocr_words)
            for tag, i1, i2, j1, j2 in word_differ.get_opcodes():
                if tag == 'replace' and i2-i1 == 1 and j2-j1 == 1:
                    # Single word substitution
                    word_substitutions[f"{gt_words[i1]}→{ocr_words[j1]}"] += 1
    
    # Summary statistics
    print(f"\n\nError Summary")
    print("=" * 30)
    print(f"Total lines compared: {len(gt_lines)}")
    print(f"Lines with errors: {lines_with_errors}")
    print(f"Line accuracy: {((len(gt_lines) - lines_with_errors) / len(gt_lines) * 100):.1f}%")
    
    # Show most common character errors
    if char_substitutions:
        print(f"\nMost Common Character/Sequence Errors:")
        print("-" * 40)
        for error, count in char_substitutions.most_common(10):
            print(f"  {error:<25} ({count} times)")
    
    # Show most common word errors
    if word_substitutions:
        print(f"\nMost Common Word Substitutions:")
        print("-" * 35)
        for error, count in word_substitutions.most_common(10):
            print(f"  {error:<25} ({count} times)")
    
    # Suggestions for improvement
    print(f"\nSuggestions for Improvement:")
    print("-" * 30)
    
    if char_substitutions:
        top_errors = char_substitutions.most_common(5)
        print("Consider adding these fixes to your cleaning function:")
        for error, count in top_errors:
            if '→' in error:
                wrong, right = error.split('→')
                if wrong != '(none)' and right != '(deleted)':
                    print(f"  ('{wrong}', '{right}'),  # occurs {count} times")
    
    print(f"\nTo improve further:")
    print("1. Add the common substitutions above to your text cleaning")
    print("2. Try different Tesseract PSM modes if WER > 10%")
    print("3. Improve image preprocessing if CER > 5%")
    print("4. Consider training custom models if accuracy plateaus")

def main():
    if len(sys.argv) not in [3, 4]:
        print("Usage: python detailed_ocr_eval.py ground_truth.txt ocr_output.txt [max_lines_to_show]")
        print("  max_lines_to_show: Number of error examples to display (default: 10)")
        sys.exit(1)
    
    max_lines = int(sys.argv[3]) if len(sys.argv) == 4 else 10
    detailed_error_analysis(sys.argv[1], sys.argv[2], max_lines)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Detailed OCR Evaluation using jiwer library with error analysis
Supports both single file comparison and directory comparison
Install with: pip install jiwer
"""

import sys
import jiwer
import difflib
from collections import Counter
from pathlib import Path

def detailed_error_analysis_single(ground_truth_file, ocr_output_file, max_lines_to_show=10):
    """Evaluate OCR accuracy with detailed error analysis for single file pair."""
    
    # Read files
    try:
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            ground_truth = f.read()
        with open(ocr_output_file, 'r', encoding='utf-8') as f:
            ocr_output = f.read()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
     
    # Calculate basic metrics
    wer = jiwer.wer(ground_truth, ocr_output)
    cer = jiwer.cer(ground_truth, ocr_output)
    
    # Track error patterns
    char_substitutions = Counter()
    word_substitutions = Counter()
    
    # Split into lines for comparison
    gt_lines = ground_truth.split('\n')
    ocr_lines = ocr_output.split('\n')
    
    lines_with_errors = 0
    
    # Compare line by line
    for i, (gt_line, ocr_line) in enumerate(zip(gt_lines, ocr_lines)):
        if gt_line.strip() != ocr_line.strip():
            lines_with_errors += 1
            
            # Track character-level substitutions
            differ = difflib.SequenceMatcher(None, gt_line, ocr_line)
            for tag, i1, i2, j1, j2 in differ.get_opcodes():
                if tag == 'replace':
                    gt_part = gt_line[i1:i2]
                    ocr_part = ocr_line[j1:j2]
                    char_substitutions[f"{gt_part}→{ocr_part}"] += 1
                elif tag == 'delete':
                    deleted = gt_line[i1:i2]
                    char_substitutions[f"{deleted}→(deleted)"] += 1
                elif tag == 'insert':
                    inserted = ocr_line[j1:j2]
                    char_substitutions[f"(none)→{inserted}"] += 1
            
            # Track word-level substitutions
            gt_words = gt_line.split()
            ocr_words = ocr_line.split()
            word_differ = difflib.SequenceMatcher(None, gt_words, ocr_words)
            for tag, i1, i2, j1, j2 in word_differ.get_opcodes():
                if tag == 'replace' and i2-i1 == 1 and j2-j1 == 1:
                    # Single word substitution
                    word_substitutions[f"{gt_words[i1]}→{ocr_words[j1]}"] += 1
    
    return {
        'wer': wer,
        'cer': cer,
        'total_lines': len(gt_lines),
        'lines_with_errors': lines_with_errors,
        'char_substitutions': char_substitutions,
        'word_substitutions': word_substitutions,
        'ground_truth': ground_truth,
        'ocr_output': ocr_output
    }

def detailed_error_analysis_directory(ground_truth_dir, ocr_output_dir, max_lines_to_show=10):
    """Evaluate OCR accuracy across all matching files in two directories."""
    
    gt_path = Path(ground_truth_dir)
    ocr_path = Path(ocr_output_dir)
    
    if not gt_path.exists():
        print(f"Error: Ground truth directory '{ground_truth_dir}' not found")
        return
    
    if not ocr_path.exists():
        print(f"Error: OCR output directory '{ocr_output_dir}' not found")
        return
    
    # Find all .txt files in ground truth directory
    gt_files = list(gt_path.glob("*.txt"))
    
    if not gt_files:
        print(f"No .txt files found in {ground_truth_dir}")
        return
    
    print(f"Found {len(gt_files)} ground truth files")
    
    # Process each file pair
    all_results = {}
    overall_char_substitutions = Counter()
    overall_word_substitutions = Counter()
    total_wer_sum = 0
    total_cer_sum = 0
    total_lines = 0
    total_lines_with_errors = 0
    successful_comparisons = 0
    
    for gt_file in sorted(gt_files):
        ocr_file = ocr_path / gt_file.name
        
        if not ocr_file.exists():
            print(f"Warning: No matching OCR file for {gt_file.name}")
            continue
        
        print(f"Processing: {gt_file.name}")
        
        # Analyze this file pair
        result = detailed_error_analysis_single(gt_file, ocr_file, max_lines_to_show)
        
        if result:
            all_results[gt_file.name] = result
            
            # Accumulate overall statistics
            total_wer_sum += result['wer']
            total_cer_sum += result['cer']
            total_lines += result['total_lines']
            total_lines_with_errors += result['lines_with_errors']
            successful_comparisons += 1
            
            # Merge error patterns
            overall_char_substitutions.update(result['char_substitutions'])
            overall_word_substitutions.update(result['word_substitutions'])
    
    if successful_comparisons == 0:
        print("No successful file comparisons!")
        return
    
    # Calculate overall metrics
    avg_wer = total_wer_sum / successful_comparisons
    avg_cer = total_cer_sum / successful_comparisons
    overall_line_accuracy = ((total_lines - total_lines_with_errors) / total_lines * 100) if total_lines > 0 else 0
    
    # Display overall results
    print("\n" + "="*70)
    print("OVERALL OCR EVALUATION RESULTS")
    print("="*70)
    print(f"Files processed:            {successful_comparisons}")
    print(f"Average Word Error Rate:    {avg_wer:.3f} ({avg_wer*100:.1f}%)")
    print(f"Average Character Error Rate: {avg_cer:.3f} ({avg_cer*100:.1f}%)")
    print(f"Average Word Accuracy:      {(1-avg_wer)*100:.1f}%")
    print(f"Average Character Accuracy: {(1-avg_cer)*100:.1f}%")
    print(f"Total lines compared:       {total_lines}")
    print(f"Lines with errors:          {total_lines_with_errors}")
    print(f"Overall line accuracy:      {overall_line_accuracy:.1f}%")
    
    # Show most common errors across all files
    print(f"\nMOST COMMON CHARACTER/SEQUENCE ERRORS (across all files):")
    print("-" * 60)
    for error, count in overall_char_substitutions.most_common(20):
        print(f"  {error:<35} ({count:>3} times)")
    
    print(f"\nMOST COMMON WORD SUBSTITUTIONS (across all files):")
    print("-" * 50)
    for error, count in overall_word_substitutions.most_common(20):
        print(f"  {error:<35} ({count:>3} times)")
    
    # Generate suggested fixes
    print(f"\nSUGGESTED FIXES FOR YOUR CLEANING FUNCTION:")
    print("-" * 50)
    print("Add these to your common_fixes dictionary:")
    
    fix_count = 0
    for error, count in overall_char_substitutions.most_common(20):
        if '→' in error and count >= 2:  # Only suggest fixes that occur multiple times
            wrong, right = error.split('→')
            if wrong not in ['(none)', '(deleted)'] and right not in ['(none)', '(deleted)']:
                print(f"    '{wrong}': '{right}',  # occurs {count} times")
                fix_count += 1
                if fix_count >= 15:  # Limit to top 15 suggestions
                    break
    
    print(f"\nTo improve further:")
    print("1. Add the suggested fixes above to your text cleaning function")
    print("2. Focus on the most frequent errors first")
    print("3. Consider preprocessing improvements if CER > 5%")
    print("4. Test different Tesseract configurations if WER > 10%")
    
    return all_results

def main():
    if len(sys.argv) not in [3, 4, 5]:
        print("Usage:")
        print("  Single file: python detailed_ocr_eval.py ground_truth.txt ocr_output.txt [max_lines_to_show]")
        print("  Directory:   python detailed_ocr_eval.py ground_truth_dir/ ocr_output_dir/ [max_lines_to_show]")
        print("")
        print("  max_lines_to_show: Number of error examples to display per file (default: 10)")
        sys.exit(1)
    
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    max_lines = int(sys.argv[3]) if len(sys.argv) >= 4 else 10
    
    # Check if arguments are directories or files
    path1 = Path(arg1)
    path2 = Path(arg2)
    
    if path1.is_dir() and path2.is_dir():
        # Directory comparison
        print("Directory comparison mode")
        print(f"Ground truth: {arg1}")
        print(f"OCR output:   {arg2}")
        print("-" * 50)
        detailed_error_analysis_directory(arg1, arg2, max_lines)
    elif path1.is_file() and path2.is_file():
        # Single file comparison
        print("Single file comparison mode")
        result = detailed_error_analysis_single(arg1, arg2, max_lines)
        if result:
            print("OCR Evaluation Results")
            print("=" * 50)
            print(f"Word Error Rate (WER):      {result['wer']:.3f} ({result['wer']*100:.1f}%)")
            print(f"Character Error Rate (CER): {result['cer']:.3f} ({result['cer']*100:.1f}%)")
            print(f"Word Accuracy:              {(1-result['wer'])*100:.1f}%")
            print(f"Character Accuracy:         {(1-result['cer'])*100:.1f}%")
            print(f"Line accuracy:              {((result['total_lines'] - result['lines_with_errors']) / result['total_lines'] * 100):.1f}%")
    else:
        print("Error: Both arguments must be either files or directories")
        print("Mixed file/directory arguments are not supported")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Ground Truth Annotation Helper

Helps you annotate ground truth data more efficiently.
"""

import json
from pathlib import Path

def load_template(template_file):
    """Load the annotation template."""
    with open(template_file, 'r') as f:
        return json.load(f)

def annotate_entry_interactive(entry):
    """Interactively annotate a single entry."""
    print(f"\n--- LINE {entry['line_number']} ---")
    print(f"Raw text: {entry['raw_text']}")
    print(f"Context: {entry['context_info']['status']}")
    
    if entry['annotation_guidance']['surname_guidance']:
        print(f"Surname hint: {entry['annotation_guidance']['surname_guidance']}")
    
    if entry['annotation_guidance']['address_guidance']:
        print(f"Address hint: {entry['annotation_guidance']['address_guidance']}")
    
    for note in entry['annotation_guidance']['special_notes']:
        print(f"Note: {note}")
    
    print("\nFill in ground truth (press Enter to skip):")
    
    # Get user input for each field
    entry['ground_truth']['surname'] = input("Surname: ").strip()
    entry['ground_truth']['first_name'] = input("First name: ").strip()
    entry['ground_truth']['home_address'] = input("Home address: ").strip()
    entry['ground_truth']['occupation'] = input("Occupation: ").strip()
    entry['ground_truth']['spouse_name'] = input("Spouse name (if any): ").strip()
    
    return entry

def main():
    """Run interactive annotation."""
    template_file = input("Enter template file path: ").strip()
    
    if not Path(template_file).exists():
        print("Template file not found!")
        return
    
    template = load_template(template_file)
    
    print(f"\nAnnotating {len(template['entries'])} entries...")
    print("Instructions:")
    for instruction in template['metadata']['annotation_instructions']:
        print(f"  {instruction}")
    
    # Annotate each entry
    for entry in template['entries']:
        annotate_entry_interactive(entry)
    
    # Save result
    output_file = template_file.replace('_template.json', '_annotated.json')
    with open(output_file, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"\nAnnotation complete! Saved to: {output_file}")

if __name__ == "__main__":
    main()

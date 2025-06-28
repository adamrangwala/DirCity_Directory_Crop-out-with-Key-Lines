#!/usr/bin/env python3
"""
Minimalist City Directory JSON Parser

Converts OCR text output from city directories into structured JSON format,
handling cross-references between entries that share surnames and addresses.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

class CityDirectoryParser:
    def __init__(self, year: str = "1900"):
        self.year = year
        self.current_surname = ""
        self.current_address = ""
        self.entries = []
        
        # Common abbreviations from 1900 Minneapolis directory page 0111
        self.abbreviations = {
            "acct": "accountant",
            "adv": "advertisement",
            "agt": "agent",
            "appr": "apprentice",
            "assn": "association",
            "asst": "assistant",
            "av": "avenue",
            "b": "boards",
            "bartndr": "bartender",
            "bet": "between",
            "bkbndr": "bookbinder",
            "bkpr": "bookkeeper",
            "blksmith": "blacksmith",
            "bldg": "building",
            "blk": "block",
            "boul": "boulevard",
            "cabmkr": "cabinet maker",
            "carp": "carpenter",
            "civ eng": "civil engineer",
            "clk": "clerk",
            "clnr": "cleaner",
            "collr": "collector",
            "commr": "commissioner",
            "comn": "commission",
            "comp": "compositor",
            "cond": "conductor",
            "conf": "confectioner",
            "contr": "contractor",
            "cor": "corner",
            "ct": "court",
            "dep": "deputy",
            "dept": "department",
            "dom": "domestic",
            "e": "east",
            "elev": "elevator",
            "eng": "engineer",
            "engr": "engraver",
            "exp": "express",
            "e s": "east side",
            "frt": "freight",
            "gen": "general",
            "ins": "insurance",
            "insptr": "inspector",
            "lab": "laborer",
            "mach": "machinist",
            "mech": "mechanic",
            "messr": "messenger",
            "mkr": "maker",
            "mnfr": "manufacturer",
            "mngr": "manager",
            "n": "north",
            "nr": "near",
            "n e": "northeast",
            "n s": "north side",
            "nw": "northwest",
            "opp": "opposite",
            "opr": "operator",
            "photogr": "photographer",
            "phys": "physician",
            "pk": "park",
            "pkr": "packer",
            "pl": "place",
            "P O": "Postoffice",
            "pres": "president",
            "prin": "principal",
            "prof": "professor",
            "propr": "proprietor",
            "pub": "publisher",
            "r": "residence",
            "rd": "road",
            "real est": "real estate",
            "repr": "repairer",
            "ret": "retail",
            "R M S": "railway mail service",
            "s": "south",
            "se": "southeast",
            "s s": "south side",
            "s w": "southwest",
            "slsmn": "salesman",
            "smstrs": "seamstress",
            "solr": "solicitor",
            "stenogr": "stenographer",
            "supt": "superintendent",
            "tchr": "teacher",
            "tel": "telephone",
            "tmstr": "teamster",
            "tndr": "tender",
            "trav": "traveling",
            "upholstr": "upholsterer",
            "vet surg": "veterinary surgeon",
            "w": "west",
            "Washn": "Washington",
            "whol": "wholesale",
            "wid": "widow",
            "w s": "west side"
            # End of provided directory abbreviations
        }
    
    def parse_text_file(self, file_path: str) -> List[Dict]:
        """Parse a single OCR text file into structured JSON entries."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            self.entries = []
            self.current_surname = ""
            self.current_address = ""
            
            for line_num, line in enumerate(lines, 1):
                entry = self.parse_line(line, line_num)
                if entry:
                    self.entries.append(entry)
            
            return self.entries
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return []
    
    def parse_line(self, line: str, line_num: int) -> Optional[Dict]:
        """Parse a single line into a structured entry."""
        if not line or len(line) < 3:
            return None
        
        # Initialize entry structure
        entry = {
            "line_number": line_num,
            "raw_text": line,
            "year": self.year,
            "surname": "",
            "first_name": "",
            "spouse_name": "",
            "home_address": "",
            "residence_indicator": "",
            "occupation": "",
            "employer": "",
            "employer_address": "",
            "parsing_notes": []
        }
        
        # Check if this line starts with a new surname (capitalized word)
        if self.starts_with_surname(line):
            # Extract surname and update current context
            parts = line.split(' ', 2)
            if len(parts) >= 2:
                self.current_surname = parts[0].rstrip(',')
                entry["surname"] = self.current_surname
                
                # Parse the rest after surname
                remainder = ' '.join(parts[1:]) if len(parts) > 1 else ""
                self.parse_name_and_details(entry, remainder)
            else:
                entry["parsing_notes"].append("Could not parse surname properly")
                return None
        else:
            # This line references the previous surname
            entry["surname"] = self.current_surname
            entry["parsing_notes"].append("Inherited surname from previous entry")
            
            # Parse the entire line as name and details
            self.parse_name_and_details(entry, line)
        
        # Try to extract address information
        self.extract_address_info(entry)
        
        return entry
    
    def starts_with_surname(self, line: str) -> bool:
        """Check if line starts with a new surname (capitalized, often followed by comma)."""
        # Look for pattern: "SURNAME" or "Surname," at start of line
        first_word = line.split()[0] if line.split() else ""
        
        # Check if it's all caps or title case and likely a surname
        if first_word and (first_word.isupper() or 
                          (first_word[0].isupper() and first_word.endswith(','))):
            return True
        
        return False
    
    def parse_name_and_details(self, entry: Dict, text: str) -> None:
        """Parse first name and other details from the text."""
        if not text:
            return
        
        # Remove leading comma if present
        text = text.lstrip(', ')
        
        # Try to extract first name (usually the first word)
        words = text.split()
        if words:
            potential_first_name = words[0].rstrip(',')
            
            # Simple heuristic: if it's not obviously an occupation/description
            if not self.looks_like_occupation(potential_first_name):
                entry["first_name"] = potential_first_name
                remaining_text = ' '.join(words[1:])
            else:
                remaining_text = text
            
            # Parse remaining details
            self.parse_occupation_and_employer(entry, remaining_text)
    
    def looks_like_occupation(self, word: str) -> bool:
        """Check if a word looks like an occupation rather than a first name."""
        occupation_indicators = ['clk', 'lab', 'tmstr', 'mach', 'eng', 'student', 'moved', 'died']
        return word.lower() in occupation_indicators
    
    def parse_occupation_and_employer(self, entry: Dict, text: str) -> None:
        """Extract occupation and employer information."""
        if not text:
            return
        
        # Look for patterns like "occupation employer, address"
        # This is a simplified approach - can be enhanced
        
        # Check for widow notation
        if 'wid ' in text.lower():
            match = re.search(r'wid ([^,)]+)', text, re.IGNORECASE)
            if match:
                entry["spouse_name"] = match.group(1).strip()
                entry["parsing_notes"].append("Identified as widow")
        
        # Look for common occupation patterns
        for abbrev, full_form in self.abbreviations.items():
            if abbrev in text:
                if not entry["occupation"] and abbrev in ['clk', 'lab', 'tmstr', 'mach', 'eng']:
                    entry["occupation"] = full_form
                    break
        
        # Store remaining text for further analysis
        entry["parsing_notes"].append(f"Remaining text: {text}")
    
    def extract_address_info(self, entry: Dict) -> None:
        """Extract address and residence indicators."""
        text = entry["raw_text"].lower()
        
        # Look for residence indicators
        residence_patterns = [
            (r'\br\s+([^,]+)', 'resides'),
            (r'\bb\s+([^,]+)', 'boards'),
            (r'\brms\s+([^,]+)', 'rooms'),
            (r'\bh\s+([^,]+)', 'house')
        ]
        
        for pattern, indicator in residence_patterns:
            match = re.search(pattern, text)
            if match:
                entry["residence_indicator"] = indicator
                entry["home_address"] = match.group(1).strip()
                self.current_address = entry["home_address"]  # Update context
                break
        
        # If no residence indicator found, look for address patterns
        if not entry["home_address"]:
            # Look for number + street patterns
            addr_match = re.search(r'\b(\d+[^,]*(?:av|st|blvd|rd)[^,]*)', text)
            if addr_match:
                entry["home_address"] = addr_match.group(1).strip()
            elif self.current_address:
                entry["home_address"] = self.current_address
                entry["parsing_notes"].append("Inherited address from previous entry")
    
    def save_to_json(self, output_file: str) -> None:
        """Save parsed entries to JSON file in structures_JSON directory."""
        try:
            # Create structures_JSON directory if it doesn't exist
            output_dir = Path("structured_JSON")
            output_dir.mkdir(exist_ok=True)
            
            # Ensure output file is in the structures_JSON directory
            output_path = output_dir / Path(output_file).name
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.entries, f, indent=2, ensure_ascii=False)
            print(f"Successfully saved {len(self.entries)} entries to {output_path}")
        except Exception as e:
            print(f"Error saving JSON: {e}")
    
    def print_summary(self) -> None:
        """Print a summary of parsed entries."""
        print(f"\nParsing Summary:")
        print(f"Total entries: {len(self.entries)}")
        
        surnames = set(entry["surname"] for entry in self.entries if entry["surname"])
        print(f"Unique surnames: {len(surnames)}")
        
        with_addresses = sum(1 for entry in self.entries if entry["home_address"])
        print(f"Entries with addresses: {with_addresses}")
        
        with_occupations = sum(1 for entry in self.entries if entry["occupation"])
        print(f"Entries with occupations: {with_occupations}")


def main():
    """Example usage of the parser."""
    # Initialize parser
    parser = CityDirectoryParser(year="1900")
    
    # Parse the example file (adjust path as needed)
    input_file = "ocr_text_files/1900_0362.txt"  # Your OCR output file
    
    if not Path(input_file).exists():
        print(f"Input file {input_file} not found!")
        print("Please provide the path to your OCR text file.")
        return
    
    # Parse the file
    print(f"Parsing {input_file}...")
    entries = parser.parse_text_file(input_file)
    
    if not entries:
        print("No entries parsed!")
        return
    
    # Save to JSON
    output_file = input_file.replace('.txt', '_structured.json')
    parser.save_to_json(output_file)
    
    # Print summary
    parser.print_summary()
    
    # Show first few entries as examples
    print(f"\nFirst 3 entries:")
    for i, entry in enumerate(entries[:3]):
        print(f"\nEntry {i+1}:")
        print(f"  Surname: {entry['surname']}")
        print(f"  First Name: {entry['first_name']}")
        print(f"  Address: {entry['home_address']}")
        print(f"  Occupation: {entry['occupation']}")
        print(f"  Notes: {entry['parsing_notes']}")


if __name__ == "__main__":
    main()
import re
import csv
import os
import logging
import pandas as pd
from datetime import datetime
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("extraction_log.txt"),
        logging.StreamHandler()
    ]
)

def extract_entries(file_path, batch_size=1000):
    """Extract individual article entries from the input file using multiple approaches."""
    try:
        logging.info(f"Reading file {file_path}...")
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            content = file.read()
        
        logging.info(f"File size: {len(content)} characters. Extracting entries...")
        
        # First approach: Pattern to match entire entries (from @article to next @article)
        entry_pattern = r'@[a-zA-Z]+\{[^@]*'
        entries = re.findall(entry_pattern, content, re.DOTALL)
        
        # Check if the last entry is captured (if it's not followed by another @article)
        if not content.strip().endswith('}'):
            last_entry_start = content.rstrip().rfind('@')
            if last_entry_start != -1:
                last_entry = content[last_entry_start:]
                if last_entry not in entries:
                    entries.append(last_entry)
        
        # Alternative approach: Split by @article and reconstruct
        alt_entries = []
        entry_types = ['article', 'book', 'incollection', 'inproceedings', 'misc', 'techreport', 'unpublished']
        for entry_type in entry_types:
            pattern = '@' + entry_type + '{'
            type_entries = content.split(pattern)
            # Skip first empty element and reconstruct
            if len(type_entries) > 1:
                type_entries = [pattern + entry for entry in type_entries[1:]]
                alt_entries.extend(type_entries)
        
        # More aggressive approach to catch non-standard format entries
        lines = content.splitlines()
        current_entry = []
        extra_entries = []
        
        for line in lines:
            if re.match(r'@[a-zA-Z]+\{', line.strip()):
                if current_entry:
                    extra_entries.append('\n'.join(current_entry))
                current_entry = [line]
            elif current_entry:
                current_entry.append(line)
        
        if current_entry:
            extra_entries.append('\n'.join(current_entry))
        
        # Merge all approaches to ensure maximum coverage
        merged_entries = set()
        for entry in entries + alt_entries + extra_entries:
            merged_entries.add(entry.strip())
        
        # Convert back to list for processing
        entries = list(merged_entries)
        
        logging.info(f"Extracted {len(entries)} entries from {file_path}")
        return entries
    except Exception as e:
        logging.error(f"Error extracting entries from {file_path}: {str(e)}")
        return []

def parse_entry(entry, entry_index=None):
    """Parse an entry and extract key fields using multiple pattern matching approaches."""
    try:
        result = {
            'author': "",
            'year': "",
            'title': "",
            'abstract': "",
            'url_or_doi': ""
        }
        
        # Extract citation key for reference
        citation_key_match = re.search(r'@[a-zA-Z]+\{([^,]*),', entry, re.IGNORECASE)
        if citation_key_match:
            result['citation_key'] = citation_key_match.group(1).strip()
        else:
            result['citation_key'] = f"unknown_{entry_index}" if entry_index is not None else "unknown"
        
        # Extract entry type
        entry_type_match = re.search(r'@([a-zA-Z]+)\{', entry, re.IGNORECASE)
        if entry_type_match:
            result['entry_type'] = entry_type_match.group(1).lower()
        else:
            result['entry_type'] = "unknown"
        
        # Improved field extraction - capture entire multi-line fields
        fields_to_extract = {
            'author': r'author\s*=\s*(\{[^}]*(?:\{[^}]*\}[^}]*)*\}|"[^"]*")',
            'year': r'year\s*=\s*(\{[^}]*\}|"[^"]*"|[0-9]{4})',
            'title': r'title\s*=\s*(\{[^}]*(?:\{[^}]*\}[^}]*)*\}|"[^"]*")',
            'abstract': r'abstract\s*=\s*(\{[^}]*(?:\{[^}]*\}[^}]*)*\}|"[^"]*")',
            'doi': r'doi\s*=\s*(\{[^}]*\}|"[^"]*"|[^,\n]*)',
            'url': r'url\s*=\s*(\{[^}]*\}|"[^"]*"|[^,\n]*)'
        }
        
        for field, pattern in fields_to_extract.items():
            match = re.search(pattern, entry, re.IGNORECASE | re.DOTALL)
            if match:
                value = match.group(1).strip()
                # Remove outer braces or quotes
                value = re.sub(r'^[\{\"]|[\}\"]$', '', value)
                
                if field == 'doi':
                    if not result['url_or_doi']:
                        # Add https://doi.org/ prefix if not already present
                        if not value.startswith("http"):
                            result['url_or_doi'] = "https://doi.org/" + value
                        else:
                            result['url_or_doi'] = value
                elif field == 'url':
                    if not result['url_or_doi']:
                        result['url_or_doi'] = value
                else:
                    result[field] = value
        
        # If we still don't have results, try the old patterns as fallback
        if not any(result.values()):
            # Author - Try multiple patterns
            author_patterns = [
                r'author\s*=\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',  # Standard with nested braces
                r'author\s*=\s*"([^"]*)"',                         # Double quoted
                r'author\s*=\s*([^,\n]*),',                        # No quotes/braces until comma
                r'author\s+=\s+(\S[^,\n]*\S),',                    # Flexible spacing
                r'authors?\s*=\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',  # 'authors' variation with braces
                r'authors?\s*=\s*"([^"]*)"',                         # 'authors' variation with quotes
            ]
            
            for pattern in author_patterns:
                author_match = re.search(pattern, entry, re.IGNORECASE)
                if author_match and not result['author']:
                    result['author'] = author_match.group(1).strip()
            
            # Year - Try multiple patterns
            year_patterns = [
                r'year\s*=\s*\{([^{}]*)\}',           # Standard with braces
                r'year\s*=\s*"([^"]*)"',              # Double quoted
                r'year\s*=\s*(\d{4})',                # Just the year number
                r'date\s*=\s*\{([^{}]*)\}',           # Date field with braces
                r'date\s*=\s*"([^"]*)"',              # Date with quotes
                r'year[\s=]+(\d{4})',                 # Flexible spacing with just digits
            ]
            
            for pattern in year_patterns:
                year_match = re.search(pattern, entry, re.IGNORECASE)
                if year_match and not result['year']:
                    year_text = year_match.group(1).strip()
                    # Extract just the year if it's a full date
                    year_only_match = re.search(r'(\d{4})', year_text)
                    if year_only_match:
                        result['year'] = year_only_match.group(1)
                    else:
                        result['year'] = year_text
            
            # Title - Try multiple patterns
            title_patterns = [
                r'title\s*=\s*\{\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\}',  # Double braced
                r'title\s*=\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',      # Single braced
                r'title\s*=\s*"([^"]*)"',                              # Double quoted
                r'[tT]itle[\s=]+"([^"]+)"',                             # Flexible spacing with quotes
                r'[tT]itle[\s=]+\{([^{}]+)\}',                          # Flexible spacing with braces
                r'[tT]itle[\s=]+([^,\n]{10,}),',                        # No quotes/braces but minimum length
            ]
            
            for pattern in title_patterns:
                title_match = re.search(pattern, entry, re.IGNORECASE)
                if title_match and not result['title']:
                    result['title'] = title_match.group(1).strip()
            
            # Abstract - Try multiple patterns
            abstract_patterns = [
                r'abstract\s*=\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',  # Standard with nested braces
                r'abstract\s*=\s*"([^"]*)"',                          # Double quoted
                r'abstract[\s=]+"([^"]+)"',                           # Flexible spacing with quotes
                r'abstract[\s=]+\{([^{}]+)\}',                        # Flexible spacing with braces
                r'abstract[\s=]+([^,\n]{30,})',                       # No quotes but minimum length
            ]
            
            for pattern in abstract_patterns:
                abstract_match = re.search(pattern, entry, re.IGNORECASE)
                if abstract_match and not result['abstract']:
                    result['abstract'] = abstract_match.group(1).strip()
            
            # DOI or URL - Try multiple patterns
            # First try DOI
            doi_patterns = [
                r'doi\s*=\s*\{([^{}]*)\}',           # Standard with braces
                r'doi\s*=\s*"([^"]*)"',              # Double quoted
                r'doi\s*=\s*([^,\n]*),',             # No quotes/braces until comma
                r'doi[\s=]+([^,\n]+),',              # Flexible spacing
                r'(10\.\d{4,}[\d\.]+/\S+)',          # Raw DOI pattern
            ]
            
            for pattern in doi_patterns:
                doi_match = re.search(pattern, entry, re.IGNORECASE)
                if doi_match and not result['url_or_doi']:
                    doi = doi_match.group(1).strip()
                    if doi.endswith('.'):
                        doi = doi[:-1]  # Remove trailing period
                    
                    # Add https://doi.org/ prefix if not already present
                    if not doi.startswith("http"):
                        result['url_or_doi'] = "https://doi.org/" + doi
                    else:
                        result['url_or_doi'] = doi
            
            # If DOI not found, try URL
            if not result['url_or_doi']:
                url_patterns = [
                    r'url\s*=\s*\{([^{}]*)\}',           # Standard with braces
                    r'url\s*=\s*"([^"]*)"',              # Double quoted
                    r'url\s*=\s*([^,\n]*),',             # No quotes/braces until comma
                    r'url[\s=]+([^,\n]+),',              # Flexible spacing
                    r'(https?://\S+)',                    # Any URL in the text
                ]
                
                for pattern in url_patterns:
                    url_match = re.search(pattern, entry, re.IGNORECASE)
                    if url_match and not result['url_or_doi']:
                        url = url_match.group(1).strip()
                        if url.endswith('.'):
                            url = url[:-1]  # Remove trailing period
                        result['url_or_doi'] = url
        
        return result
    except Exception as e:
        logging.error(f"Error parsing entry {entry_index}: {str(e)}")
        return {
            'citation_key': f"error_{entry_index}" if entry_index is not None else "error",
            'author': "",
            'year': "",
            'title': "",
            'abstract': "",
            'url_or_doi': ""
        }

def check_completeness(entry):
    """Check if an entry has all required fields."""
    missing_fields = []
    
    for field in ['author', 'year', 'title', 'abstract', 'url_or_doi']:
        if not entry.get(field, "").strip():
            missing_fields.append(field)
    
    return missing_fields

def attempt_recovery(entry, source_text, original_entry_text):
    """Attempt to recover missing fields using alternative parsing approaches."""
    missing_fields = check_completeness(entry)
    recovered_fields = []
    
    if not missing_fields:
        return entry, recovered_fields
    
    # Look for complete field blocks in the text
    for field in missing_fields:
        try:
            # Match the complete field pattern including braces content
            field_pattern = rf'{field}\s*=\s*(\{{[^}}]*(?:\{{[^}}]*\}}[^}}]*)*\}}|"[^"]*")'
            field_match = re.search(field_pattern, original_entry_text, re.IGNORECASE | re.DOTALL)
            
            if field_match and not entry[field]:
                value = field_match.group(1).strip()
                # Clean up outer braces or quotes
                value = re.sub(r'^[\{\"]|[\}\"]$', '', value)
                entry[field] = value
                recovered_fields.append(field)
                continue
                
            # Alternative approach: capture from field= to next field or closing brace
            lines = original_entry_text.splitlines()
            field_content = []
            capturing = False
            
            for i, line in enumerate(lines):
                line = line.strip()
                if re.match(rf'{field}\s*=\s*', line, re.IGNORECASE):
                    capturing = True
                    # Extract content after the equals sign
                    content_part = re.sub(rf'{field}\s*=\s*', '', line, flags=re.IGNORECASE)
                    field_content.append(content_part)
                elif capturing:
                    # If we reach another field or closing brace, stop capturing
                    if re.match(r'[a-zA-Z_]+\s*=', line) or line == '}':
                        break
                    field_content.append(line)
            
            if field_content and not entry[field]:
                joined_content = ' '.join(field_content)
                # Clean up outer braces or quotes and trailing commas
                value = re.sub(r'^[\{\"]|[\}\",]$', '', joined_content)
                entry[field] = value
                recovered_fields.append(field)
        except Exception as e:
            logging.debug(f"Failed to recover {field} for {entry['citation_key']}: {str(e)}")
    
    # Try more aggressive methods for DOI/URL
    if 'url_or_doi' in missing_fields:
        try:
            # Look for DOI pattern
            doi_match = re.search(r'10\.\d{4,}[\d\.]+/\S+', original_entry_text)
            if doi_match:
                entry['url_or_doi'] = "https://doi.org/" + doi_match.group(0)
                recovered_fields.append('url_or_doi')
        except Exception as e:
            logging.debug(f"Failed to recover DOI for {entry['citation_key']}: {str(e)}")
    
    # Try to extract abstract by looking for largest text block
    if 'abstract' in missing_fields:
        try:
            # Find the largest block of text that's not already assigned to another field
            lines = original_entry_text.split('\n')
            abstract_block = []
            capturing = False
            
            for line in lines:
                if re.match(r'abstract\s*=\s*', line, re.IGNORECASE):
                    capturing = True
                    # Get content after the equals sign
                    content = re.sub(r'abstract\s*=\s*', '', line, re.IGNORECASE).strip()
                    if content:
                        abstract_block.append(content)
                elif capturing:
                    if re.match(r'[a-zA-Z_]+\s*=', line) or line.strip() == '}':
                        break
                    abstract_block.append(line.strip())
            
            if abstract_block:
                abstract_text = ' '.join(abstract_block)
                # Clean up braces and quotes
                abstract_text = re.sub(r'^[\{\"]|[\}\",]$', '', abstract_text)
                entry['abstract'] = abstract_text
                recovered_fields.append('abstract')
        except Exception as e:
            logging.debug(f"Failed to recover abstract from text blocks: {str(e)}")
    
    # Try to infer missing fields from available information
    if 'author' in missing_fields and entry.get('title'):
        # Look for authors in other entries with the same title
        for text in source_text:
            if entry['title'] in text and text != original_entry_text:
                author_match = re.search(r'author\s*=\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text, re.IGNORECASE | re.DOTALL)
                if author_match:
                    entry['author'] = author_match.group(1).strip()
                    recovered_fields.append('author')
                    break
    
    # Try to extract year from URLs or citation keys if available
    if 'year' in missing_fields:
        # Look for year pattern in the URL
        if entry.get('url_or_doi'):
            year_match = re.search(r'/(20\d{2})/', entry['url_or_doi'])
            if year_match:
                entry['year'] = year_match.group(1)
                recovered_fields.append('year')
        
        # Look for year in citation key
        if 'citation_key' in entry:
            year_match = re.search(r'(19|20)\d{2}', entry['citation_key'])
            if year_match:
                entry['year'] = year_match.group(0)
                recovered_fields.append('year')
                
        # Look for year in the entire entry text
        if not entry.get('year'):
            year_matches = re.findall(r'(20\d{2})', original_entry_text)
            if year_matches:
                entry['year'] = year_matches[0]
                recovered_fields.append('year')
    
    return entry, recovered_fields

def generate_report(incomplete_entries, output_file):
    """Generate a README.md file with information about incomplete entries."""
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write("# Incomplete Publication Records Report\n\n")
        file.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        file.write(f"Total incomplete records: {len(incomplete_entries)}\n\n")
        
        file.write("## Summary of Missing Fields\n\n")
        missing_counts = {'author': 0, 'year': 0, 'title': 0, 'abstract': 0, 'url_or_doi': 0}
        
        for entry in incomplete_entries:
            for field in missing_counts.keys():
                if not entry.get(field, "").strip():
                    missing_counts[field] += 1
        
        file.write("| Field | Missing Count |\n")
        file.write("|-------|---------------|\n")
        for field, count in missing_counts.items():
            file.write(f"| {field} | {count} |\n")
        
        file.write("\n## Detailed List of Incomplete Records\n\n")
        
        for i, entry in enumerate(incomplete_entries):
            file.write(f"### Record {i+1}: {entry.get('citation_key', 'Unknown')}\n\n")
            
            # List available fields
            file.write("**Available Fields:**\n\n")
            for field in ['author', 'year', 'title', 'abstract', 'url_or_doi']:
                if entry.get(field, "").strip():
                    if field == 'abstract' and len(entry[field]) > 300:
                        file.write(f"- **{field}**: {entry[field][:300]}...\n")
                    else:
                        file.write(f"- **{field}**: {entry[field]}\n")
            
            # List missing fields
            missing = [field for field in ['author', 'year', 'title', 'abstract', 'url_or_doi'] 
                      if not entry.get(field, "").strip()]
            file.write("\n**Missing Fields:**\n\n")
            for field in missing:
                file.write(f"- {field}\n")
            
            file.write("\n---\n\n")

def process_in_batches(entries, batch_size=1000):
    """Process entries in batches to reduce memory usage."""
    results = []
    total_batches = (len(entries) + batch_size - 1) // batch_size  # Ceiling division
    
    for i in range(0, len(entries), batch_size):
        batch_entries = entries[i:i+batch_size]
        batch_num = i // batch_size + 1
        logging.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_entries)} entries)")
        
        # First pass: parse all entries in batch
        batch_results = []
        for j, entry in enumerate(batch_entries):
            parsed_entry = parse_entry(entry, i + j)
            batch_results.append(parsed_entry)
        
        # Identify incomplete entries in batch
        incomplete_entries = []
        for j, entry in enumerate(batch_results):
            missing_fields = check_completeness(entry)
            if missing_fields:
                incomplete_entries.append((j, entry, missing_fields))
        
        logging.info(f"Batch {batch_num}: Found {len(incomplete_entries)} incomplete entries")
        
        # Recovery pass for batch
        recovered_count = 0
        for j, entry, missing_fields in incomplete_entries:
            original_entry_text = batch_entries[j] if j < len(batch_entries) else ""
            recovered_entry, recovered_fields = attempt_recovery(entry, batch_entries, original_entry_text)
            
            # Update the results
            batch_results[j] = recovered_entry
            
            if recovered_fields:
                recovered_count += 1
                logging.debug(f"Recovered fields for entry {i+j} ({entry.get('citation_key', 'unknown')}): {', '.join(recovered_fields)}")
        
        logging.info(f"Batch {batch_num}: Recovered data for {recovered_count} entries")
        
        # Add batch results to main results
        results.extend(batch_results)
    
    return results

def main():
    start_time = time.time()
    
    # Use file in the current workspace instead of hardcoded path
    input_file = "d:/Research/Smart City publication/notes1.txt"
    output_file = "Lit_summar.csv"
    incomplete_report_file = "README_incomplete_notes1.md"
    
    # Process parameters
    batch_size = 1000  # Process entries in batches of this size
    
    # Check if file exists
    if not os.path.exists(input_file):
        # Try with absolute path as fallback
        alt_input_file = os.path.abspath(input_file)
        if not os.path.exists(alt_input_file):
            logging.error(f"Error: Could not find file at {input_file} or {alt_input_file}")
            return
        input_file = alt_input_file
    
    logging.info(f"Extracting entries from {input_file}...")
    entries = extract_entries(input_file)
    logging.info(f"Found {len(entries)} entries. Processing...")
    
    # Process entries in batches to reduce memory usage
    results = process_in_batches(entries, batch_size)
    
    # Identify entries that are still incomplete after recovery
    still_incomplete = []
    for entry in results:
        missing_fields = check_completeness(entry)
        if missing_fields:
            still_incomplete.append(entry)
    
    logging.info(f"Still have {len(still_incomplete)} incomplete entries after recovery")
    
    # Write results to CSV
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['author', 'year', 'title', 'abstract', 'url_or_doi', 'entry_type']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                # Write only the required fields to the CSV
                filtered_result = {k: result.get(k, "") for k in fieldnames}
                writer.writerow(filtered_result)
        
        logging.info(f"Results written to {output_file}")
    except Exception as e:
        logging.error(f"Error writing to CSV file: {str(e)}")
    
    # Generate report for incomplete entries
    if still_incomplete:
        try:
            generate_report(still_incomplete, incomplete_report_file)
            logging.info(f"Incomplete records report generated: {incomplete_report_file}")
        except Exception as e:
            logging.error(f"Error generating incomplete records report: {str(e)}")
    
    # Read the CSV to check for empty cells
    try:
        df = pd.read_csv(output_file)
        total_rows = len(df)
        empty_cells = df.isna().sum()
        
        logging.info(f"CSV Analysis:")
        logging.info(f"Total rows: {total_rows}")
        logging.info(f"Empty cells per column:")
        for column, count in empty_cells.items():
            logging.info(f"  - {column}: {count} ({count/total_rows*100:.2f}%)")
    except Exception as e:
        logging.error(f"Error analyzing CSV file: {str(e)}")
    
    end_time = time.time()
    execution_time = end_time - start_time
    logging.info(f"Execution completed in {execution_time:.2f} seconds")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Unhandled exception in main: {str(e)}") 
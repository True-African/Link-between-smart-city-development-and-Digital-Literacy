import pandas as pd
import re
import os
import logging
from datetime import datetime
import difflib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("similarity_check_log.txt"),
        logging.StreamHandler()
    ]
)

def normalize_text(text):
    """Normalize text for better comparison."""
    if not isinstance(text, str):
        return ""
    # Convert to lowercase, remove extra whitespace
    text = re.sub(r'\s+', ' ', str(text).lower().strip())
    # Remove common LaTeX formatting
    text = re.sub(r'[\{\}\\]', '', text)
    return text

def compute_similarity(str1, str2):
    """Compute similarity between two strings using difflib."""
    if not isinstance(str1, str) or not isinstance(str2, str):
        return 0.0
    return difflib.SequenceMatcher(None, str1, str2).ratio()

def find_matches_and_mismatches(df1, df2, source_name1, source_name2, similarity_threshold=0.85):
    """Find matches and mismatches between two dataframes."""
    # Create dictionaries to store matches and mismatches
    matches = []
    only_in_df1 = []
    only_in_df2 = []
    
    # Normalize titles for comparison
    df1['normalized_title'] = df1['title'].apply(normalize_text)
    df2['normalized_title'] = df2['title'].apply(normalize_text)
    
    # Create a set of normalized titles from df2 for faster lookup
    df2_titles = set(df2['normalized_title'].dropna())
    
    # Check each row in df1
    for idx1, row1 in df1.iterrows():
        if not row1['normalized_title']:
            continue
            
        # Look for exact title match
        if row1['normalized_title'] in df2_titles:
            # Find matching row(s) in df2
            matching_rows = df2[df2['normalized_title'] == row1['normalized_title']]
            
            for idx2, row2 in matching_rows.iterrows():
                # Check if other fields also match
                abstract_similarity = compute_similarity(normalize_text(row1['abstract']), normalize_text(row2['abstract']))
                author_similarity = compute_similarity(normalize_text(row1['author']), normalize_text(row2['author']))
                
                # If most fields match, consider it a full match
                if abstract_similarity > similarity_threshold and author_similarity > similarity_threshold:
                    matches.append({
                        'title': row1['title'],
                        'author': row1['author'],
                        'year': row1['year'],
                        'source1': source_name1,
                        'source2': source_name2,
                        'abstract_similarity': abstract_similarity,
                        'author_similarity': author_similarity
                    })
                    break
            else:
                # Partial match (title only)
                only_in_df1.append({
                    'title': row1['title'],
                    'author': row1['author'],
                    'year': row1['year'],
                    'abstract': row1['abstract'][:100] + "..." if isinstance(row1['abstract'], str) and len(row1['abstract']) > 100 else row1['abstract'],
                    'source': source_name1,
                    'reason': 'Title matched but other fields differ'
                })
        else:
            # Check for fuzzy title matches
            best_match = None
            best_similarity = 0
            
            for idx2, row2 in df2.iterrows():
                if not row2['normalized_title']:
                    continue
                    
                similarity = compute_similarity(row1['normalized_title'], row2['normalized_title'])
                
                if similarity > similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = row2
            
            if best_match is not None:
                # Fuzzy match
                matches.append({
                    'title': row1['title'],
                    'author': row1['author'],
                    'year': row1['year'],
                    'source1': source_name1,
                    'source2': source_name2,
                    'title_similarity': best_similarity,
                    'matched_title': best_match['title']
                })
            else:
                # No match found
                only_in_df1.append({
                    'title': row1['title'], 
                    'author': row1['author'],
                    'year': row1['year'],
                    'abstract': row1['abstract'][:100] + "..." if isinstance(row1['abstract'], str) and len(row1['abstract']) > 100 else row1['abstract'],
                    'source': source_name1,
                    'reason': 'No matching title found'
                })
    
    # Find entries only in df2
    df1_titles = set(df1['normalized_title'].dropna())
    
    for idx2, row2 in df2.iterrows():
        if not row2['normalized_title'] or row2['normalized_title'] in df1_titles:
            continue
            
        # Check for fuzzy matches that might have been missed
        best_match = None
        best_similarity = 0
        
        for idx1, row1 in df1.iterrows():
            if not row1['normalized_title']:
                continue
                
            similarity = compute_similarity(row2['normalized_title'], row1['normalized_title'])
            
            if similarity > similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = row1
        
        if best_match is None:
            # No match found, only in df2
            only_in_df2.append({
                'title': row2['title'],
                'author': row2['author'],
                'year': row2['year'],
                'abstract': row2['abstract'][:100] + "..." if isinstance(row2['abstract'], str) and len(row2['abstract']) > 100 else row2['abstract'],
                'source': source_name2,
                'reason': 'No matching title found'
            })
    
    return matches, only_in_df1, only_in_df2

def generate_report(matches, only_in_file1, only_in_file2, file1_name, file2_name, output_file):
    """Generate a markdown report of the comparison."""
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write("# CSV Files Comparison Report\n\n")
        file.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        file.write("## Summary\n\n")
        file.write(f"- **Total records in {file1_name}**: {len(only_in_file1) + len(matches)}\n")
        file.write(f"- **Total records in {file2_name}**: {len(only_in_file2) + len(matches)}\n")
        file.write(f"- **Matching records**: {len(matches)}\n")
        file.write(f"- **Records only in {file1_name}**: {len(only_in_file1)}\n")
        file.write(f"- **Records only in {file2_name}**: {len(only_in_file2)}\n\n")
        
        # Calculate overlap percentage
        if len(only_in_file1) + len(matches) > 0:
            overlap1 = (len(matches) / (len(only_in_file1) + len(matches))) * 100
            file.write(f"- **Overlap percentage in {file1_name}**: {overlap1:.2f}%\n")
        
        if len(only_in_file2) + len(matches) > 0:
            overlap2 = (len(matches) / (len(only_in_file2) + len(matches))) * 100
            file.write(f"- **Overlap percentage in {file2_name}**: {overlap2:.2f}%\n\n")
        
        # Sample of matching records
        file.write("## Sample of Matching Records\n\n")
        file.write("| Title | Author | Year |\n")
        file.write("|-------|--------|------|\n")
        
        for i, match in enumerate(matches[:10]):  # Show first 10 matches
            title = match['title'][:50] + "..." if len(match['title']) > 50 else match['title']
            author = match['author'][:30] + "..." if isinstance(match['author'], str) and len(match['author']) > 30 else match['author']
            file.write(f"| {title} | {author} | {match['year']} |\n")
        
        if len(matches) > 10:
            file.write(f"*... and {len(matches) - 10} more matches*\n\n")
        else:
            file.write("\n")
        
        # Records only in file1
        file.write(f"## Records Only in {file1_name}\n\n")
        file.write("| Title | Author | Year | Reason |\n")
        file.write("|-------|--------|------|--------|\n")
        
        for i, record in enumerate(only_in_file1[:20]):  # Show first 20 records
            title = record['title'][:50] + "..." if isinstance(record['title'], str) and len(record['title']) > 50 else record['title']
            author = record['author'][:30] + "..." if isinstance(record['author'], str) and len(record['author']) > 30 else record['author']
            file.write(f"| {title} | {author} | {record['year']} | {record['reason']} |\n")
        
        if len(only_in_file1) > 20:
            file.write(f"*... and {len(only_in_file1) - 20} more records*\n\n")
        else:
            file.write("\n")
        
        # Records only in file2
        file.write(f"## Records Only in {file2_name}\n\n")
        file.write("| Title | Author | Year | Reason |\n")
        file.write("|-------|--------|------|--------|\n")
        
        for i, record in enumerate(only_in_file2[:20]):  # Show first 20 records
            title = record['title'][:50] + "..." if isinstance(record['title'], str) and len(record['title']) > 50 else record['title']
            author = record['author'][:30] + "..." if isinstance(record['author'], str) and len(record['author']) > 30 else record['author']
            file.write(f"| {title} | {author} | {record['year']} | {record['reason']} |\n")
        
        if len(only_in_file2) > 20:
            file.write(f"*... and {len(only_in_file2) - 20} more records*\n\n")
        else:
            file.write("\n")
        
        # Recommendations and next steps
        file.write("## Recommendations and Next Steps\n\n")
        file.write("1. **Merge the datasets**: Consider creating a unified dataset combining unique records from both sources.\n")
        file.write("2. **Investigate mismatches**: Some records appear in one file but not the other. This could be due to:\n")
        file.write("   - Different extraction patterns capturing different entries\n")
        file.write("   - Variations in how record fields were parsed\n")
        file.write("   - Differences in handling special characters or formatting\n\n")
        file.write("3. **Improve extraction logic**: Based on the mismatches, enhance the extraction algorithms to better handle:\n")
        file.write("   - Multiple entry types (article, book, inproceedings, etc.)\n")
        file.write("   - Complex formatting in titles and abstracts\n")
        file.write("   - Special characters and LaTeX formatting\n\n")
        file.write("4. **Data quality improvement**: Create a merged dataset with the highest quality data from both sources.\n")

def merge_datasets(df1, df2, output_file):
    """Merge unique records from both datasets into a new CSV file."""
    # Normalize titles for comparison
    df1['normalized_title'] = df1['title'].apply(normalize_text)
    df2['normalized_title'] = df2['title'].apply(normalize_text)
    
    # Find all unique normalized titles
    all_titles = set(df1['normalized_title'].dropna()).union(set(df2['normalized_title'].dropna()))
    
    # Create dictionary to store the merged records
    merged_records = []
    
    # Process each unique title
    for title in all_titles:
        if not title:
            continue
            
        df1_matches = df1[df1['normalized_title'] == title]
        df2_matches = df2[df2['normalized_title'] == title]
        
        if len(df1_matches) > 0 and len(df2_matches) > 0:
            # Record exists in both datasets, select the more complete one
            row1 = df1_matches.iloc[0]
            row2 = df2_matches.iloc[0]
            
            merged_record = {}
            
            # For each field, take the non-empty one or the longer one
            for field in ['author', 'year', 'title', 'abstract', 'url_or_doi']:
                val1 = str(row1[field]) if pd.notna(row1[field]) else ""
                val2 = str(row2[field]) if pd.notna(row2[field]) else ""
                
                if not val1 and not val2:
                    merged_record[field] = ""
                elif not val1:
                    merged_record[field] = val2
                elif not val2:
                    merged_record[field] = val1
                else:
                    # Take the longer value
                    merged_record[field] = val1 if len(val1) >= len(val2) else val2
            
            # Add entry type if available
            if 'entry_type' in row2.index:
                merged_record['entry_type'] = row2['entry_type']
            
            merged_records.append(merged_record)
        elif len(df1_matches) > 0:
            # Record only in df1
            row = df1_matches.iloc[0]
            merged_record = {
                'author': row['author'] if pd.notna(row['author']) else "",
                'year': row['year'] if pd.notna(row['year']) else "",
                'title': row['title'] if pd.notna(row['title']) else "",
                'abstract': row['abstract'] if pd.notna(row['abstract']) else "",
                'url_or_doi': row['url_or_doi'] if pd.notna(row['url_or_doi']) else ""
            }
            
            merged_records.append(merged_record)
        elif len(df2_matches) > 0:
            # Record only in df2
            row = df2_matches.iloc[0]
            merged_record = {
                'author': row['author'] if pd.notna(row['author']) else "",
                'year': row['year'] if pd.notna(row['year']) else "",
                'title': row['title'] if pd.notna(row['title']) else "",
                'abstract': row['abstract'] if pd.notna(row['abstract']) else "",
                'url_or_doi': row['url_or_doi'] if pd.notna(row['url_or_doi']) else ""
            }
            
            # Add entry type if available
            if 'entry_type' in row.index:
                merged_record['entry_type'] = row['entry_type']
            
            merged_records.append(merged_record)
    
    # Create a new dataframe from the merged records
    merged_df = pd.DataFrame(merged_records)
    
    # Sort by title
    if 'title' in merged_df.columns:
        merged_df = merged_df.sort_values('title')
    
    # Write to CSV
    merged_df.to_csv(output_file, index=False)
    
    return merged_df

def main():
    # File paths
    file1 = "literature_results_Abstra3.csv"
    file2 = "Lit_summar.csv"
    report_file = "similarity_report.md"
    merged_file = "merged_dataset.csv"
    
    # Load CSV files
    logging.info(f"Loading {file1}...")
    df1 = pd.read_csv(file1)
    logging.info(f"Loaded {len(df1)} records from {file1}")
    
    logging.info(f"Loading {file2}...")
    df2 = pd.read_csv(file2)
    logging.info(f"Loaded {len(df2)} records from {file2}")
    
    # Find matches and mismatches
    logging.info("Comparing datasets...")
    matches, only_in_df1, only_in_df2 = find_matches_and_mismatches(df1, df2, file1, file2)
    
    logging.info(f"Found {len(matches)} matching records")
    logging.info(f"Found {len(only_in_df1)} records only in {file1}")
    logging.info(f"Found {len(only_in_df2)} records only in {file2}")
    
    # Generate report
    logging.info(f"Generating report to {report_file}...")
    generate_report(matches, only_in_df1, only_in_df2, file1, file2, report_file)
    
    # Merge datasets
    logging.info(f"Merging datasets to {merged_file}...")
    merged_df = merge_datasets(df1, df2, merged_file)
    logging.info(f"Created merged dataset with {len(merged_df)} records")
    
    logging.info("Done!")

if __name__ == "__main__":
    main() 
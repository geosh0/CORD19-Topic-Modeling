import os
import json
import pandas as pd
from tqdm import tqdm

def load_metadata(metadata_path):
    """
    Loads and filters metadata for papers with valid dates.
    """
    print("Loading and Filtering Metadata...")
    cols_to_keep = ['cord_uid', 'sha', 'pmcid', 'title', 'abstract', 'publish_time']
    
    # Load
    meta_df = pd.read_csv(metadata_path, dtype={'sha': str, 'pmcid': str}, usecols=cols_to_keep)
    
    # Filter Dates
    meta_df['publish_time'] = pd.to_datetime(meta_df['publish_time'], errors='coerce')
    meta_df = meta_df.dropna(subset=['publish_time'])
    
    print(f"Metadata loaded. Rows: {len(meta_df)}")
    return meta_df

def _load_json_batch(directory, valid_ids, suffix_to_remove, source_label):
    """
    Internal helper to load JSON files.
    """
    extracted_data = []
    
    if not os.path.exists(directory):
        print(f"ERROR: Directory not found: {directory}")
        return pd.DataFrame()

    files = [f for f in os.listdir(directory) if f.endswith(".json")]
    print(f"Scanning {len(files)} files in {directory} (Source: {source_label})...")
    
    for filename in tqdm(files):
        file_id = filename.replace(suffix_to_remove, "")
        
        if file_id not in valid_ids:
            continue
            
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                json_content = json.load(f)
                
                # Extract Body Text
                body_text = ""
                if 'body_text' in json_content:
                    body_text = "\n".join([item['text'] for item in json_content['body_text']])
                
                extracted_data.append({
                    'paper_id': file_id,
                    'body_text': body_text,
                    'text_source': source_label
                })
        except Exception:
            continue
            
    return pd.DataFrame(extracted_data)

def combine_text_cols(row):
    """Helper to merge title, abstract, and body."""
    title = str(row['title']) if pd.notna(row['title']) else ""
    abstract = str(row['abstract']) if pd.notna(row['abstract']) else ""
    body = str(row['body_text']) if pd.notna(row['body_text']) else ""
    return f"{title} [SEP] {abstract} [SEP] {body}"

def run_loading_pipeline(base_path):
    """
    Main function to execute the full loading process.
    """
    metadata_path = os.path.join(base_path, "metadata.csv")
    pdf_path = os.path.join(base_path, "document_parses", "pdf_json")
    pmc_path = os.path.join(base_path, "document_parses", "pmc_json")

    # 1. Load Metadata
    meta_df = load_metadata(metadata_path)
    
    # 2. Prepare Valid IDs
    valid_shas = set(meta_df['sha'].dropna())
    valid_pmcids = set(meta_df['pmcid'].dropna())

    # 3. Load JSONs
    df_pmc = _load_json_batch(pmc_path, valid_pmcids, ".xml.json", "pmc")
    df_pdf = _load_json_batch(pdf_path, valid_shas, ".json", "pdf")

    # 4. Merge
    print("\nMerging Data...")
    merged_pmc = pd.DataFrame()
    merged_pdf = pd.DataFrame()

    if not df_pmc.empty:
        merged_pmc = pd.merge(df_pmc, meta_df, left_on='paper_id', right_on='pmcid', how='inner')
    
    if not df_pdf.empty:
        merged_pdf = pd.merge(df_pdf, meta_df, left_on='paper_id', right_on='sha', how='inner')

    # Concatenate (PMC first)
    full_data = pd.concat([merged_pmc, merged_pdf])

    # 5. Deduplicate
    print(f"Rows before deduplication: {len(full_data)}")
    full_data = full_data.drop_duplicates(subset=['cord_uid'], keep='first')
    print(f"Rows after deduplication: {len(full_data)}")

    # 6. Create Full Text
    tqdm.pandas(desc="Combining Text")
    full_data['full_text'] = full_data.progress_apply(combine_text_cols, axis=1)

    # Return final clean dataframe
    return full_data[['cord_uid', 'publish_time', 'title', 'abstract', 'full_text']]
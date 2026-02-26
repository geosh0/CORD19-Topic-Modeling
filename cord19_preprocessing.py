import spacy
import re
from gensim.models.phrases import Phrases, Phraser
from tqdm import tqdm

# Initialize Spacy once when the module is imported
# This prevents reloading it every time you call a function
try:
    nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])
except OSError:
    print("Warning: Spacy model 'en_core_web_sm' not found. Please download it.")

# Define Stopwords
custom_stops = {
    "doi", "preprint", "copyright", "peer", "reviewed", "org", "https", "et", "al",
    "author", "figure", "table", "rights", "reserved", "permission", "use", "used",
    "using", "biorxiv", "medrxiv", "license", "fig", "data", "study", "results",
    "conclusion", "method", "significant", "showed", "shown", "virus", "covid-19",
    "background", "objective", "methods", "discussion", "introduction",
    "abstract", "paper", "report", "case"
}
stop_words = nlp.Defaults.stop_words.union(custom_stops)

def clean_text_robust(text):
    """Regex cleaning for LaTeX, URLs, and formatting."""
    text = str(text).lower()
    text = re.sub(r'\\[a-z]+', ' ', text)       # LaTeX
    text = re.sub(r'\{.*?\}', ' ', text)        # Brackets
    text = re.sub(r'http\S+', '', text)         # URLs
    text = re.sub(r'\S*@\S*\s?', '', text)      # Emails
    text = re.sub(r'\[.*?\]', '', text)         # [SEP]
    text = re.sub(r'[^a-z\s]', '', text)        # Numbers/Special chars
    text = re.sub(r'\s+', ' ', text).strip()    # Whitespace
    return text

def spacy_process(text):
    """Tokenization and Lemmatization."""
    doc = nlp(text)
    # Keep tokens: Not stopwords, length > 3
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and len(token.text) > 3]
    return tokens

def run_nlp_pipeline(df):
    """
    Main function to run the full NLP pipeline on a dataframe.
    Expects dataframe to have 'full_text' and 'publish_time'.
    """
    # 1. Filter for Pandemic Era (Dec 2019+)
    print("Filtering for Pandemic Era (Dec 2019+)...")
    df = df[df['publish_time'] >= '2019-12-01'].copy()
    
    # 2. Filter Short Docs
    print("Removing short documents...")
    df = df[df['full_text'].str.split().str.len() > 50].copy()
    print(f"Documents remaining: {len(df)}")

    # 3. Regex Cleaning
    print("Applying Regex Cleaning...")
    tqdm.pandas()
    df['clean_text'] = df['full_text'].progress_apply(clean_text_robust)

    # 4. Spacy Processing
    print("Tokenizing & Lemmatizing (This takes time)...")
    df['tokens'] = df['clean_text'].progress_apply(spacy_process)

    # 5. Bigrams
    print("Building Bigrams...")
    phrases = Phrases(df['tokens'], min_count=10, threshold=10)
    bigram_mod = Phraser(phrases)
    
    df['tokens_bigram'] = df['tokens'].progress_apply(lambda x: bigram_mod[x])

    print("Preprocessing Complete.")
    return df
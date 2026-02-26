from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
import pandas as pd

def train_scientific_bertopic(docs, n_neighbors=15, min_cluster_size=15):
    """
    Trains a BERTopic model optimized for Scientific Text (CORD-19) using
    the SPECTER embedding model and KeyBERT for topic representation.
    """
    print("--- 1. Initializing Scientific Embeddings (SPECTER) ---")
    # We use AllenAI's SPECTER, specifically designed for scientific papers
    embedding_model = SentenceTransformer("allenai/specter")
    
    # Pre-calculate embeddings to speed up the process
    print("Generating embeddings (this may take a while on CPU)...")
    embeddings = embedding_model.encode(docs, show_progress_bar=True)

    print("\n--- 2. Configuring Sub-Models ---")
    # UMAP for dimensionality reduction
    umap_model = UMAP(n_neighbors=n_neighbors, 
                      n_components=5, 
                      min_dist=0.0, 
                      metric='cosine', 
                      random_state=42)
    
    # HDBSCAN for clustering
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, 
                            metric='euclidean', 
                            cluster_selection_method='eom', 
                            prediction_data=True)
    
    # KeyBERTInspired for more intuitive topic representations
    representation_model = KeyBERTInspired()

    print("\n--- 3. Training BERTopic Model ---")
    topic_model = BERTopic(
        embedding_model=embedding_model, 
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        representation_model=representation_model,
        language="english",
        calculate_probabilities=True,
        verbose=True
    )

    # Train the model using the pre-calculated embeddings
    topics, probs = topic_model.fit_transform(docs, embeddings)
    
    print(f"\nTraining Complete. Found {len(topic_model.get_topic_info()) - 1} topics (excluding outlier topic -1).")
    return topic_model, embeddings # Return embeddings to avoid recalculating

def run_temporal_analysis(topic_model, docs, timestamps):
    """
    Runs the temporal analysis step (Topics over Time).
    """
    print("\n--- Running Temporal Analysis ---")
    
    # FIX: Force conversion to Series so .dt accessor works
    dates = pd.Series(pd.to_datetime(timestamps))
    
    # Calculate unique months for binning
    num_bins = dates.dt.to_period('M').nunique()
    
    print(f"Detected {num_bins} monthly time bins.")

    topics_over_time = topic_model.topics_over_time(
        docs=docs,
        timestamps=timestamps,
        nr_bins=num_bins,
        global_tuning=True,
        evolution_tuning=True
    )
    
    return topics_over_time
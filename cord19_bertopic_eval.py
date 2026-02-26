import gensim.corpora as corpora
from gensim.models import CoherenceModel
import pandas as pd

def get_bertopic_topics_as_list(topic_model, top_n_words=10):
    """
    Extracts the top words for each topic from BERTopic and converts them
    into the list-of-lists format required by Gensim.
    Excludes the outlier topic (-1).
    """
    topic_data = topic_model.get_topics()
    topics_list = []
    
    # BERTopic stores topics in a dict: {topic_id: [(word, score), ...]}
    for topic_id, word_scores in topic_data.items():
        if topic_id == -1:
            continue # Skip outliers
        
        # Extract just the words, discarding the c-TF-IDF scores
        words = [word for word, score in word_scores[:top_n_words]]
        topics_list.append(words)
        
    return topics_list

def calculate_bertopic_coherence(topic_model, doc_tokens, top_n_words=10):
    """
    Calculates the C_v coherence score for a trained BERTopic model.
    
    Args:
        topic_model: The trained BERTopic model.
        doc_tokens: List of tokenized documents (e.g. df_clean['tokens_bigram']).
                    CRITICAL: Must be tokens, not raw strings.
        top_n_words: How many top words per topic to consider (default 10).
        
    Returns:
        score: The C_v coherence score (0.0 to 1.0).
    """
    print("--- Preparing BERTopic data for Coherence Calculation ---")
    
    # 1. Extract Topics
    topics = get_bertopic_topics_as_list(topic_model, top_n_words)
    print(f"Extracted {len(topics)} topics for evaluation.")
    
    # 2. Build Dictionary (Gensim requires a dictionary mapping)
    # We build a fresh dictionary from the tokens to ensure all words are covered
    print("Building temporary dictionary...")
    dictionary = corpora.Dictionary(doc_tokens)
    
    # 3. Calculate Coherence
    print("Calculating C_v Score (this may take a moment)...")
    cm = CoherenceModel(topics=topics, 
                        texts=doc_tokens, 
                        dictionary=dictionary, 
                        coherence='c_v')
    
    score = cm.get_coherence()
    print(f"BERTopic Coherence Score: {score:.4f}")
    
    return score
import gensim.corpora as corpora
from pprint import pprint

def analyze_vocabulary(tokens):
    """
    Prints statistics about rare and frequent words to help choose filter parameters.
    """
    print("--- 1. Initial Dictionary Build ---")
    id2word = corpora.Dictionary(tokens)
    total_docs = len(tokens)
    print(f"Total Documents: {total_docs}")
    print(f"Total Unique Words: {len(id2word)}")

    # Analyze Rare Words
    print("\n--- Rare Word Analysis (Bottom End) ---")
    for count in [5, 10, 20, 50]:
        n_words = sum(1 for doc_freq in id2word.dfs.values() if doc_freq < count)
        print(f"Words appearing in < {count} docs: {n_words} ({n_words/len(id2word):.1%})")

    return id2word

def create_dictionary_corpus(tokens, no_below=20, no_above=0.35):
    """
    Creates the final Dictionary and Corpus after filtering and cleaning.
    """
    print(f"\n--- 2. Building Final Dictionary/Corpus ---")
    id2word = corpora.Dictionary(tokens)
    print(f"Original Vocabulary Size: {len(id2word)}")

    # 1. Remove German Stopwords (Hardcoded list based on EDA)
    german_stops = ['eine', 'einer', 'oder', 'nicht', 'sind', 'auch', 'para', 
                    'werden', 'sich', 'bei', 'dem', 'dass', 'durch', 'nach', 
                    'wird', 'einem', 'bat', 'knnen', 'einen', 'kann', 'sowie', 
                    'diese', 'uber']
    
    bad_ids = [id2word.token2id[word] for word in german_stops if word in id2word.token2id]
    id2word.filter_tokens(bad_ids=bad_ids)
    print(f"Removed {len(bad_ids)} German stopwords.")

    # 2. Filter Extremes
    id2word.filter_extremes(no_below=no_below, no_above=no_above)
    print(f"Vocabulary Size after Filtering: {len(id2word)}")

    # 3. Create Corpus
    corpus = [id2word.doc2bow(text) for text in tokens]

    # 4. Save
    id2word.save("cord19_dictionary.gensim")
    corpora.MmCorpus.serialize("cord19_corpus.mm", corpus)
    print("Dictionary and Corpus saved to disk.")
    
    # 5. Preview
    print("\nTop 20 Most Frequent Words in Corpus:")
    top_words = sorted(id2word.cfs.items(), key=lambda x: x[1], reverse=True)[:20]
    for word_id, count in top_words:
        print(f"{id2word[word_id]}: {count}")

    return id2word, corpus
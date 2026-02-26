import gensim
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
from pprint import pprint

def run_optimization_loop(dictionary, corpus, texts, start=5, limit=31, step=5):
    """
    Trains multiple LDA models to find the optimal Coherence Score.
    """
    topic_range = range(start, limit, step)
    model_results = {'num_topics': [], 'coherence': [], 'model': []}
    
    print(f"Starting Hyperparameter Tuning on {len(corpus)} documents...")

    for k in topic_range:
        print(f"Training model with {k} topics...", end=" ")
        
        # Train LdaModel (Single Core for 'auto' alpha)
        lda_model = gensim.models.LdaModel(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
        
        # Calculate Coherence
        coherence_model = CoherenceModel(model=lda_model, 
                                         texts=texts, 
                                         dictionary=dictionary, 
                                         coherence='c_v')
        cv_score = coherence_model.get_coherence()
        
        # Store
        model_results['num_topics'].append(k)
        model_results['coherence'].append(cv_score)
        model_results['model'].append(lda_model)
        
        print(f"Coherence: {cv_score:.4f}")
    
    return model_results

def plot_coherence(model_results):
    """Visualizes the tuning results."""
    plt.figure(figsize=(10, 5))
    plt.plot(model_results['num_topics'], model_results['coherence'], marker='o', color='teal')
    plt.title("Coherence Score vs Number of Topics")
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score (C_v)")
    plt.grid(True)
    plt.show()

def save_best_model(model_results):
    """Identifies and saves the model with the highest score."""
    # Find max score index
    best_index = model_results['coherence'].index(max(model_results['coherence']))
    best_k = model_results['num_topics'][best_index]
    best_score = model_results['coherence'][best_index]
    best_model = model_results['model'][best_index]

    print(f"\nWinner: {best_k} Topics with Score: {best_score:.4f}")
    
    # Save
    filename = f"lda_optimal_{best_k}.model"
    best_model.save(filename)
    print(f"Optimal model saved as '{filename}'")
    
    print("\n--- FULL TOPIC LIST ---")
    pprint(best_model.print_topics(num_topics=best_k, num_words=10))
    
    return best_model
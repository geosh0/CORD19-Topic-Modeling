# CORD-19: Evolutionary Topic Modeling of the Pandemic

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NLP](https://img.shields.io/badge/NLP-Spacy%20%7C%20Gensim-green)
![Models](https://img.shields.io/badge/Models-LDA%20%7C%20BERTopic-orange)

## 📌 Project Overview
The COVID-19 Open Research Dataset (CORD-19) represents one of the largest open-science initiatives in history. This project applies unsupervised machine learning techniques to analyze **over 11,000 scientific papers** published during the critical first year of the pandemic (Dec 2019 – Oct 2020).

**The Objective:** To reconstruct the timeline of scientific discovery using NLP. Instead of treating the corpus as a static snapshot, this project performs **Temporal Topic Analysis** to visualize how the global research focus shifted from *crisis management* (Ventilators, AI Diagnostics) to *solutions* (Vaccines, Education).

## 🚀 Key Findings
1.  **The "AI Spike" (Q1 2020):** In the early months, research into Deep Learning and Computer Vision surged immediately. Lacking biological samples, researchers relied on X-Ray/CT datasets to build diagnostic models.
2.  **The Clinical Crisis (Q2 2020):** Papers regarding "Respiratory Failure" and "Ventilators" peaked in April/May 2020, aligning with the first wave of hospitalizations.
3.  **The Vaccine Pivot (Q3 2020):** General Virology topics dominated early on, but were steadily overtaken by Immunology and Vaccine Development topics starting in Summer 2020.

## 🛠️ Methodology & Pipeline
This project implements a modular data science pipeline, moving beyond "spaghetti code" notebooks into structured Python modules.

### 1. Data Ingestion (`cord19_loader.py`)
*   **Source:** AllenAI CORD-19 Dataset (October 2020 snapshot).
*   **Challenge:** Handling mixed data formats (PDF JSON vs. PMC XML) and duplicates.
*   **Solution:** Built a prioritization pipeline favoring PMC XML parses for higher text quality and performed deduplication based on `cord_uid`.

### 2. NLP Preprocessing (`cord19_preprocessing.py`)
*   **Noise Removal:** Custom Regex to strip LaTeX tags, URLs, and academic boilerplate.
*   **Normalization:** Lemmatization using `SciSpaCy` (`en_core_web_sm`).
*   **Phrase Detection:** Applied Gensim's Phraser to treat medical compound terms (e.g., `herd_immunity`, `cytokine_storm`) as single tokens.

### 3. Probabilistic Modeling: LDA (`cord19_lda.py`)
*   **Approach:** Latent Dirichlet Allocation (Bag-of-Words).
*   **Optimization:** Tuned the number of topics ($K$) using $C_v$ Coherence Scores.
*   **Result:** The model stabilized at **$K=25$** (Coherence: **0.6451**), effectively identifying broad research disciplines.

### 4. Semantic Modeling: BERTopic (`cord19_bertopic.py`)
*   **Approach:** Transformer-based clustering.
*   **Engine:** Utilized **AllenAI SPECTER** embeddings, fine-tuned on scientific citations, to cluster papers based on semantic meaning rather than just keyword overlap.
*   **Result:** Generated **100** highly granular topics, distinguishing specific pathogens (e.g., *Acinetobacter* co-infections) that LDA grouped together.

## 📊 Model Comparison

| Feature | LDA (Probabilistic) | BERTopic (Semantic) |
| :--- | :--- | :--- |
| **Input** | Bag of Words (Frequency) | Embeddings (Context) |
| **Coherence ($C_v$)** | **0.6451** (High) | 0.4009 (Moderate) |
| **Granularity** | 25 Broad Topics | **100 Specific Topics** |
| **Use Case** | Document Routing / High-level Overview | Literature Review / Specific Information Retrieval |

**Conclusion:** LDA proved superior for high-level categorization and statistical coherence, while BERTopic excelled at identifying nuanced sub-topics and filtering non-English noise automatically.

## 📂 Repository Structure

```bash
├── cord19_loader.py          # Data ingestion and deduplication logic
├── cord19_preprocessing.py   # Spacy pipeline, Regex cleaning, Bigrams
├── cord19_dictionary.py      # Dictionary creation and extreme value filtering
├── cord19_lda.py             # LDA training, tuning loop, and model saving
├── cord19_analysis.py        # Dominant topic assignment and label mapping
├── cord19_bertopic.py        # Semantic model training (SPECTER)
├── cord19_bertopic_eval.py   # Coherence evaluation for BERTopic
├── notebook_main.ipynb       # The main execution notebook (Report)
└── README.md                 # Project documentation
```

## 📚 Data Source & Citation

This project analyzes the **CORD-19 Open Research Dataset**, curated by the Allen Institute for AI and partners.

If you utilize this repository or the dataset, please cite the original paper:

> Wang, L. L., Lo, K., et al. (2020). **CORD-19: The COVID-19 Open Research Dataset.** *Proceedings of the 1st Workshop on NLP for COVID-19 at ACL 2020.*

**BibTeX:**
```bibtex
@inproceedings{wang-etal-2020-cord,
    title = "{CORD-19}: The {COVID-19} Open Research Dataset",
    author = "Wang, Lucy Lu  and Lo, Kyle  and Chandrasekhar, Yoganand  and Reas, Russell  and Yang, Jiangjiang  and Burdick, Doug  and Eide, Darrin  and Funk, Kathryn  and Katsis, Yannis  and Kinney, Rodney Michael  and Li, Yunyao  and Liu, Ziyang  and Merrill, William  and Mooney, Paul  and Murdick, Dewey A.  and Rishi, Devvret  and Sheehan, Jerry  and Shen, Zhihong  and Stilson, Brandon  and Wade, Alex D.  and Wang, Kuansan  and Wang, Nancy Xin Ru  and Wilhelm, Christopher  and Xie, Boya  and Raymond, Douglas M.  and Weld, Daniel S.  and Etzioni, Oren  and Kohlmeier, Sebastian",
    booktitle = "Proceedings of the 1st Workshop on {NLP} for {COVID-19} at {ACL} 2020",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.nlpcovid19-acl.1"
}

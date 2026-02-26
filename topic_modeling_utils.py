import pandas as pd
from gensim.models.ldamodel import LdaModel

# It's good practice to include the topic label map here as well,
# so all your topic modeling utilities are in one place.
TOPIC_LABEL_MAP = {
    0: "Biochem/Protein Interaction",
    1: "Bacterial/Antibiotics",
    2: "Clinical Symptoms",
    3: "Surgery & Radiology",
    4: "AI/Deep Learning",
    5: "Sensors/Physics",
    6: "Diagnostic Testing",
    7: "Vaccine/Immunology",
    8: "Ventilators/Respiratory Failure",
    9: "Cellular Biology",
    10: "Genomics/Viral Evolution",
    11: "Clinical Mortality/Risk Factors",
    12: "Neuro/Pathway Mechanisms",
    13: "Govt Policy/Economy",
    14: "Social Context/Community",
    15: "Pediatrics/Family Health",
    16: "Epidemiology/Spread",
    17: "Math Modeling/Simulation",
    18: "Education/Surveys",
    19: "Environmental/Urban",
    20: "Lab Assays/Protocols",
    21: "Literature Review",
    22: "Statistical Forecasting",
    23: "Healthcare Services",
    24: "General Virology"
}

def assign_dominant_topics(ldamodel: LdaModel, corpus: list, original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns the dominant topic to each document in a corpus and merges it
    with the original DataFrame.

    Args:
        ldamodel: The trained LDA model.
        corpus: The document-term matrix (corpus).
        original_df: The original DataFrame to merge the topic information with.

    Returns:
        A new DataFrame with dominant topic information and labels.
    """
    # 1. Get dominant topic for each document
    sent_topics_df_list = []

    # Iterate through every document
    for i, row_list in enumerate(ldamodel[corpus]):
        # The model might return a tuple; we only need the topic probabilities
        if isinstance(row_list, tuple):
            row_list = row_list[0]

        # Sort topics by probability
        row_list = sorted(row_list, key=lambda x: x[1], reverse=True)

        # Get the dominant topic, its percentage contribution, and keywords
        if len(row_list) > 0:
            topic_num, prop_topic = row_list[0]
            wp = ldamodel.show_topic(topic_num)
            topic_keywords = ", ".join([word for word, prop in wp[:5]])
            sent_topics_df_list.append([int(topic_num), round(prop_topic, 4), topic_keywords])
        else:
            # Append a default row if no topics are found
            sent_topics_df_list.append([-1, 0.0, "No Topics Found"])

    # Convert the list of lists to a DataFrame
    df_topic_info = pd.DataFrame(sent_topics_df_list, columns=['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords'])

    # 2. Merge with the original DataFrame
    # Ensure indices are aligned for a clean merge
    original_df = original_df.reset_index(drop=True)
    df_dominant = pd.concat([original_df, df_topic_info], axis=1)

    # 3. Apply the human-readable labels
    df_dominant['Topic_Label'] = df_dominant['Dominant_Topic'].map(TOPIC_LABEL_MAP)
    # Optional: Fill unmapped topics with their topic number
    df_dominant['Topic_Label'] = df_dominant['Topic_Label'].fillna(df_dominant['Dominant_Topic'].astype(str))

    print(f"Topic assignment complete. New DataFrame shape: {df_dominant.shape}")
    return df_dominant
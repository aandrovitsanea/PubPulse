#!/bin/env python3

import pandas as pd

# Define the function to parse the text
def parse_txt(text, data_source):
    """
    Parses the provided text into structured data and captures different sections of a research paper.

    This function separates the text content into distinct sections such as abstracts, title paragraphs,
    and description tables. It identifies each section by the presence of specific markers ('ABSTRACT',
    'TITLE PARAGRAPH:', and 'DESCRIPTION TABLE:'), then extracts the title (if any) and the associated
    content. The parsed data is returned as a pandas DataFrame with the columns 'type', 'title', 'input',
    and 'data_source'.

    Parameters:
    - text (str): The text content of a document to parse.
    - data_source (str): The name or identifier for the text data source (e.g., the file name).

    Returns:
    - DataFrame: A pandas DataFrame with the structured data extracted from the text. The DataFrame
      contains columns for the section type ('type'), the title of the section ('title'), the content
      of the section ('input'), and the data source ('data_source').

    The 'type' column categorizes the section into 'TITLE_PARAGRAPH', 'DESCRIPTION_TABLE', 'ABSTRACT',
    or 'Other'. The 'title' column includes the title for 'TITLE_PARAGRAPH' and 'DESCRIPTION_TABLE'
    sections, and default titles for 'ABSTRACT' or 'Other' sections. The 'input' column contains the
    actual text content for each section, and the 'data_source' column records the provided data source
    name for all entries.

    Example:
    ```
    text_content = "ABSTRACT\n ... \n----\nTITLE PARAGRAPH: Introduction\n ... \n----\n"
    data_source_name = "example.txt"
    parsed_df = parse_text(text_content, data_source_name)
    ```
    """
    import pandas as pd

    # Split text into sections
    sections = text.split('\n----\n')

    # Prepare a DataFrame to store the parsed data
    parsed_data = {
        'type': [],
        'title': [],
        'input': [],
        'data_source': [],
    }

    # Iterate through the sections to parse the content
    for section in sections:
        section = section.strip()
        if 'TITLE PARAGRAPH:' in section:
            # Remove the title identifier and extract the title
            title = section.split('\n', 1)[0].replace('TITLE PARAGRAPH:', '').strip()
            content = section.split('\n', 1)[1].strip() if '\n' in section else ''
            parsed_data['input'].append(content)
            parsed_data['type'].append('TITLE_PARAGRAPH')
            parsed_data['title'].append(title)
        elif 'DESCRIPTION TABLE:' in section:
            # Remove the table identifier
            title = 'DESCRIPTION_TABLE'
            content = section.replace('DESCRIPTION TABLE:', '').strip()
            parsed_data['input'].append(content)
            parsed_data['type'].append('DESCRIPTION_TABLE')
            parsed_data['title'].append(title)
        else:
            # For abstract or any other type that is not defined above
            title = 'ABSTRACT' if 'ABSTRACT' in section else 'Other'
            parsed_data['input'].append(section)
            parsed_data['type'].append('ABSTRACT' if 'ABSTRACT' in section else 'Other')
            parsed_data['title'].append(title)

        parsed_data['data_source'].append(data_source)

    return pd.DataFrame(parsed_data) # return parsed data as DataFrame


def plot_token_distribution(token_counts, title):
  import matplotlib.pyplot as plt
  import seaborn as sns
  sns.set_style("whitegrid")
  plt.figure(figsize=(15, 6))
  plt.hist(token_counts, bins=50, color='#3498db', edgecolor='black')
  plt.title(title, fontsize=16)
  plt.xlabel("Number of tokens", fontsize=14)
  plt.ylabel("Number of examples", fontsize=14)
  plt.xticks(fontsize=12)
  plt.yticks(fontsize=12)
  plt.tight_layout()
  plt.show()



def deduplicate_dataframe(df: pd.DataFrame, 
                          model: str, 
                          threshold: float):


    from sentence_transformers import SentenceTransformer
    import faiss
    import pandas as pd
    from tqdm.autonotebook import tqdm
    import numpy as np
    sentence_model = SentenceTransformer(model)
    inputs = df["input"].tolist()

    print("Converting text to embeddings...")
    embeddings = sentence_model.encode(inputs, show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index.add(normalized_embeddings)

    print("Filtering out near-duplicates...")
    D, I = index.search(normalized_embeddings, k=2)
    to_keep = []

    for i in tqdm(range(len(embeddings)), desc="Filtering"):
        if D[i, 1] < threshold:
            to_keep.append(i)
        else:
            nearest_neighbor = I[i, 1]
            if i not in to_keep and nearest_neighbor not in to_keep:
                to_keep.append(i)

    deduped_df = df.iloc[to_keep].reset_index(drop=True)
    return deduped_df

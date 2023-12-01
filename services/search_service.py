#!/bin/env python3

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from tqdm.autonotebook import tqdm
import sys

# Appending paths to sys.path for module imports

sys.path.append("..")
sys.path.append("../lib")
sys.path.append("../data")
sys.path.append("../services")
sys.path.append("../embedding_storage")

# Define constants for storage directory and file names
EMBEDDINGS_FILE = os.path.join(STORAGE_DIR, 'embeddings.npy')
FAISS_INDEX_FILE = os.path.join(STORAGE_DIR, 'faiss_index.faiss')
STORAGE_DIR = '../embedding_storage'

# Create storage directory if it doesn't exist
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)


def load_or_create_embeddings(dataset,
                              model_name):
    """
    Loads existing embeddings and FAISS index or creates them if they don't exist.

    Args:
        dataset (DataFrame): The dataset to use for creating embeddings.
        model_name (str): Name of the SentenceTransformer model.

    Returns:
        Tuple of embeddings array, FAISS index, and sentence model.
    """
    sentence_model = SentenceTransformer(model_name)

    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(FAISS_INDEX_FILE):
        embeddings = np.load(EMBEDDINGS_FILE)
        faiss_index = faiss.read_index(FAISS_INDEX_FILE)
    else:
        embeddings, faiss_index = create_embeddings_and_faiss_index(dataset,
                                                                    sentence_model)
        np.save(EMBEDDINGS_FILE, embeddings)
        faiss.write_index(faiss_index,
                          FAISS_INDEX_FILE)

    return embeddings, faiss_index, sentence_model

def create_embeddings_and_faiss_index(dataset,
                                      sentence_model):
    """
    Creates embeddings and FAISS index for a given dataset.

    Args:
        dataset (DataFrame): The dataset to use for creating embeddings.
        sentence_model (SentenceTransformer): Pre-initialized SentenceTransformer model.

    Returns:
        Tuple of embeddings array and FAISS index.
    """
    # Extract title paragraphs for embedding
    title_paragraphs = dataset[dataset['type'] == 'TITLE_PARAGRAPH']['input'].tolist()
    embeddings = sentence_model.encode(title_paragraphs,
                                       show_progress_bar=True)
    normalized_embeddings = embeddings / np.linalg.norm(embeddings,
                                                        axis=1,
                                                        keepdims=True)
    # Create FAISS index
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(normalized_embeddings.astype(np.float32))

    return embeddings, faiss_index

def update_embeddings_and_index(new_dataset,
                                sentence_model,
                                embeddings,
                                faiss_index):
    """
    Updates embeddings and FAISS index with new dataset entries.

    Args:
        new_dataset (DataFrame): New dataset entries to be added.
        sentence_model (SentenceTransformer): Pre-initialized SentenceTransformer model.
        embeddings (ndarray): Existing embeddings.
        faiss_index (faiss.IndexFlatIP): Existing FAISS index.

    Returns:
        Updated embeddings and FAISS index.
    """
    new_embeddings = sentence_model.encode(new_dataset['input'].tolist(), show_progress_bar=True)
    new_normalized_embeddings = new_embeddings / np.linalg.norm(new_embeddings,
                                                                axis=1,
                                                                keepdims=True)
    # Update embeddings and FAISS index
    updated_embeddings = np.concatenate((embeddings,
                                         new_normalized_embeddings),
                                        axis=0)
    faiss_index.add(new_normalized_embeddings.astype(np.float32))

    # Save updated embeddings and index
    np.save(EMBEDDINGS_FILE, updated_embeddings)
    faiss.write_index(faiss_index, FAISS_INDEX_FILE)

    return updated_embeddings, faiss_index

def search_similar_papers(query_paragraph,
                          sentence_model,
                          faiss_index,
                          dataset,
                          top_k=3):
    """
    Searches for similar papers based on a query paragraph.

    Args:
        query_paragraph (str): Paragraph to search for similar papers.
        sentence_model (SentenceTransformer): Pre-initialized SentenceTransformer model.
        faiss_index (faiss.IndexFlatIP): FAISS index for similarity search.
        dataset (DataFrame): Dataset containing paragraphs and their sources.
        top_k (int): Number of top similar results to return.

    Returns:
        DataFrame: Similar papers with their data sources and similarity scores.
    """
    query_embedding = sentence_model.encode([query_paragraph], show_progress_bar=False)
    query_embedding = query_embedding / np.linalg.norm(query_embedding,
                                                       axis=1,
                                                       keepdims=True)

    distances, indices = faiss_index.search(query_embedding.astype(np.float32), top_k + 1)

    similar_indices = indices.flatten()[1:1+top_k]
    similar_distances = distances.flatten()[1:1+top_k]

    similar_papers = dataset.iloc[similar_indices]
    similar_papers.loc[similar_indices, 'score'] = similar_distances

    # Exclude the input data source from the results
    input_data_source = dataset[dataset['input'] == query_paragraph]['data_source'].iloc[0]
    similar_papers = similar_papers[similar_papers['data_source'] != input_data_source]

    return similar_papers[['data_source', 'score']].reset_index(drop=True)

def engine(dataset, model):
     """
    Analyzes a dataset to find similar papers for each 'TITLE_PARAGRAPH'.

    This function first loads or creates embeddings and a FAISS index for the dataset. It then iterates over
    each 'TITLE_PARAGRAPH' in the dataset, computes its similarity with other paragraphs using the FAISS index,
    and finds the top 3 similar papers. The function returns these results in a pandas DataFrame.

    Parameters:
    - dataset (pd.DataFrame): The dataset containing paragraphs and their types.
    - model (str): The name of the Sentence Transformer model to use for embeddings.

    Returns:
    - pd.DataFrame: A DataFrame with each input paragraph, its source, and the top 3 similar papers along with their similarity scores.
    """

    # Load or create embeddings and FAISS index for the given dataset and model
    embeddings, faiss_index, sentence_model = load_or_create_embeddings(dataset, model)

    results = []

    # Iterate over each row in the dataset where the type is 'TITLE_PARAGRAPH'
    for index, row in tqdm(dataset[dataset['type'] == 'TITLE_PARAGRAPH'].iterrows(),
                           total=dataset.shape[0]):
        # Only process if index is within the length of embeddings
        if index < len(embeddings):
            # Find top 3 similar papers for the current paragraph
            similar_papers_df = search_similar_papers(row['input'],
                                                      sentence_model,
                                                      faiss_index,
                                                      dataset,
                                                      top_k=3)
            # Append results including input paragraph, its source, similar papers, and scores
            results.append({
                'input_paragraph': row['input'],
                'input_paper': row['data_source'],
                'similar_papers': similar_papers_df['data_source'].tolist(),
                'scores': similar_papers_df['score'].tolist()
            })
    # Return the results as a DataFrame
    return pd.DataFrame(results)

def add_new_data_to_index(new_dataset, model_name):
    """
    Adds new data to the existing embeddings and FAISS index.

    This function first loads the existing embeddings and FAISS index. It then updates them with the new data
    from the provided dataset using the specified Sentence Transformer model.

    Parameters:
    - new_dataset (pd.DataFrame): New data to be added to the index.
    - model_name (str): The Sentence Transformer model name for embedding generation.

    Note:
    - The embeddings and index are updated in-place and saved to their respective files.
    """

    # Load existing embeddings and FAISS index
    embeddings, faiss_index, sentence_model = load_or_create_embeddings(dataset,
                                                                        model_name)
    # Update embeddings and index with the new data
    update_embeddings_and_index(new_dataset,
                                sentence_model,
                                embeddings,
                                faiss_index)

def find_similar_papers_for_input(input_text,
                                  model_name,
                                  dataset,
                                  top_k=3):
    """
    Finds similar papers for a given input text using a specified Sentence Transformer model.

    This function creates embeddings for the input text and uses a FAISS index to find the top_k similar
    papers in the dataset. It ensures that the results are distinct and not duplicates.

    Parameters:
    - input_text (str): The text for which similar papers are to be found.
    - model_name (str): The Sentence Transformer model name for embedding generation.
    - dataset (pd.DataFrame): The dataset to search in.
    - top_k (int): Number of top similar results to return.

    Returns:
    - list: A list of dictionaries containing the similar papers' data sources and their similarity scores.
    """

    # Load or create embeddings and FAISS index
    embeddings, faiss_index, sentence_model = load_or_create_embeddings(dataset,
                                                                        model_name)

    # Ensure input_text is a string
    if isinstance(input_text, pd.DataFrame) and 'input' in input_text.columns:
        # Concatenate all input text into a single string
        input_text = input_text['input'].str.cat(sep=' ')

    # Create embedding for the input text
    input_embedding = sentence_model.encode([input_text], show_progress_bar=False)
    # Normalize the embedding
    input_embedding = input_embedding / np.linalg.norm(input_embedding,
                                                       axis=1,
                                                       keepdims=True)

    # Search for similar papers, considering more than top_k to ensure distinctness
    distances, indices = faiss_index.search(input_embedding.astype(np.float32), top_k + 10)  # Search more to ensure distinctness

    similar_papers = []
    unique_data_sources = set()
    found = 0
    # Iterate over the indices to find distinct similar papers
    for i in range(indices.shape[1]):
        if found >= top_k:
            break

        similar_index = indices[0][i]
        # Check if the result is not the query itself and not a duplicate
        data_source = dataset.iloc[similar_index]['data_source']
        if similar_index < len(dataset) and data_source != input_text and data_source not in unique_data_sources:
            # Add unique similar paper information
            similar_paper_info = {
                "data_source": data_source,
                "score": distances[0][i]
            }
            similar_papers.append(similar_paper_info)
            unique_data_sources.add(data_source)
            found += 1

    for i, paper in enumerate(similar_papers, 1):
        print(f"Similar Paper {i}: {paper['data_source']}, Score: {paper['score']}")

    return similar_papers


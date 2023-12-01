#!/bin/env python3

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from tqdm.autonotebook import tqdm
import sys
sys.path.append("..")
sys.path.append("../lib")
sys.path.append("../data")
sys.path.append("../services")
sys.path.append("../embedding_storage")

# Directory for saving/loading embeddings and FAISS index
STORAGE_DIR = '../embedding_storage'

# Ensure the storage directory exists
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

# File paths within the storage directory
EMBEDDINGS_FILE = os.path.join(STORAGE_DIR, 'embeddings.npy')
FAISS_INDEX_FILE = os.path.join(STORAGE_DIR, 'faiss_index.faiss')



def load_or_create_embeddings(dataset, model_name):
    sentence_model = SentenceTransformer(model_name)

    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(FAISS_INDEX_FILE):
        embeddings = np.load(EMBEDDINGS_FILE)
        faiss_index = faiss.read_index(FAISS_INDEX_FILE)
    else:
        embeddings, faiss_index = create_embeddings_and_faiss_index(dataset, sentence_model)
        np.save(EMBEDDINGS_FILE, embeddings)
        faiss.write_index(faiss_index, FAISS_INDEX_FILE)

    return embeddings, faiss_index, sentence_model

def create_embeddings_and_faiss_index(dataset, sentence_model):
    title_paragraphs = dataset[dataset['type'] == 'TITLE_PARAGRAPH']['input'].tolist()
    embeddings = sentence_model.encode(title_paragraphs, show_progress_bar=True)
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(normalized_embeddings.astype(np.float32))

    return embeddings, faiss_index

def update_embeddings_and_index(new_dataset, sentence_model, embeddings, faiss_index):
    new_embeddings = sentence_model.encode(new_dataset['input'].tolist(), show_progress_bar=True)
    new_normalized_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)

    updated_embeddings = np.concatenate((embeddings, new_normalized_embeddings), axis=0)
    faiss_index.add(new_normalized_embeddings.astype(np.float32))

    np.save(EMBEDDINGS_FILE, updated_embeddings)
    faiss.write_index(faiss_index, FAISS_INDEX_FILE)

    return updated_embeddings, faiss_index

def search_similar_papers(query_paragraph, sentence_model, faiss_index, dataset, top_k=3):
    query_embedding = sentence_model.encode([query_paragraph], show_progress_bar=False)
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

    distances, indices = faiss_index.search(query_embedding.astype(np.float32), top_k + 1)

    similar_indices = indices.flatten()[1:1+top_k]
    similar_distances = distances.flatten()[1:1+top_k]

    similar_papers = dataset.iloc[similar_indices]
    similar_papers.loc[similar_indices, 'score'] = similar_distances

    input_data_source = dataset[dataset['input'] == query_paragraph]['data_source'].iloc[0]
    similar_papers = similar_papers[similar_papers['data_source'] != input_data_source]

    return similar_papers[['data_source', 'score']].reset_index(drop=True)

def engine(dataset, model):
    embeddings, faiss_index, sentence_model = load_or_create_embeddings(dataset, model)
    results = []

    for index, row in tqdm(dataset[dataset['type'] == 'TITLE_PARAGRAPH'].iterrows(), total=dataset.shape[0]):
        if index < len(embeddings):
            similar_papers_df = search_similar_papers(row['input'], sentence_model, faiss_index, dataset, top_k=3)
            results.append({
                'input_paragraph': row['input'],
                'input_paper': row['data_source'],
                'similar_papers': similar_papers_df['data_source'].tolist(),
                'scores': similar_papers_df['score'].tolist()
            })

    return pd.DataFrame(results)

def add_new_data_to_index(new_dataset, model_name):
    embeddings, faiss_index, sentence_model = load_or_create_embeddings(dataset, model_name)
    update_embeddings_and_index(new_dataset, sentence_model, embeddings, faiss_index)

def find_similar_papers_for_input(input_text, model_name, dataset, top_k=3):
    embeddings, faiss_index, sentence_model = load_or_create_embeddings(dataset, model_name)

    # Ensure input_text is a string
    if isinstance(input_text, pd.DataFrame) and 'input' in input_text.columns:
        input_text = input_text['input'].str.cat(sep=' ')  # Concatenate all input text

    # Create embedding for the input text
    input_embedding = sentence_model.encode([input_text], show_progress_bar=False)
    input_embedding = input_embedding / np.linalg.norm(input_embedding, axis=1, keepdims=True)

    # Search for similar papers
    distances, indices = faiss_index.search(input_embedding.astype(np.float32), top_k + 10)  # Search more to ensure distinctness

    similar_papers = []
    unique_data_sources = set()
    found = 0
    for i in range(indices.shape[1]):
        if found >= top_k:
            break

        similar_index = indices[0][i]
        # Check if the result is not the query itself and not a duplicate
        data_source = dataset.iloc[similar_index]['data_source']
        if similar_index < len(dataset) and data_source != input_text and data_source not in unique_data_sources:
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


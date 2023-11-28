#!/bin/env python3

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm.autonotebook import tqdm
import pandas as pd



# Helper function to search for similar papers
def search_similar_papers(query_paragraph: str, 
                          model, 
                          faiss_index, 
                          dataset, 
                          top_k: int = 3):
    """
    Search for the top_k most similar papers based on the input paragraph.

    Parameters:
    - query_paragraph: str - The input paragraph from the paper.
    - sentence_model: SentenceTransformer - The model used to convert text to embeddings. 
                                            ie. "sentence-transformers/all-MiniLM-L6-v2"
    - faiss_index: faiss.IndexFlatIP - The FAISS index for similarity search.
    - dataset: pd.DataFrame - DataFrame containing metadata and embeddings of the papers.
    - top_k: int - Number of most similar papers to return.

    Returns:
    - pd.DataFrame - A DataFrame containing the metadata of the top_k most similar papers.
    """
    
    # Initialize the sentence model
    sentence_model = SentenceTransformer(model)

    # Embed the query paragraph
    query_embedding = sentence_model.encode([query_paragraph], 
                                            show_progress_bar=False)
    query_embedding = query_embedding / np.linalg.norm(query_embedding, 
                                                       axis=1, 
                                                       keepdims=True)
    
    # Search the FAISS index for the top_k similar vectors, excluding the query itself
    D, I = faiss_index.search(query_embedding.astype(np.float32), 
                              top_k + 1)
    
    similar_papers_indices = I.flatten()[1:]  # Skip the first result as it is the query itself
    similar_papers_metadata = dataset.iloc[similar_papers_indices]
    
    # Filter out any potential duplicates including the input paper itself
    similar_papers_metadata = similar_papers_metadata[similar_papers_metadata['data_source'] != dataset[dataset['input'] == query_paragraph]['data_source'].iloc[0]]
    similar_papers_metadata = similar_papers_metadata.drop_duplicates(subset=['data_source']).head(top_k)
    
    return similar_papers_metadata.reset_index(drop=True)

# Embed all title paragraphs and create the FAISS index
def create_embeddings_and_faiss_index(dataset,
                                      model):
    # Initialize the sentence model
    sentence_model = SentenceTransformer(model)  
    title_paragraphs = dataset[dataset['type'] == 'TITLE_PARAGRAPH']['input'].tolist()
    embeddings = sentence_model.encode(title_paragraphs, show_progress_bar=True)
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Create the FAISS index
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(normalized_embeddings.astype(np.float32))

    return embeddings, normalized_embeddings, faiss_index

# Function to process the dataset and find similar papers for each title paragraph
def process_dataset(dataset, model):
    embeddings, normalized_embeddings, faiss_index = create_embeddings_and_faiss_index(dataset,
                                                                                      model)
    results = []

    for index, row in dataset[dataset['type'] == 'TITLE_PARAGRAPH'].iterrows():
        if index < len(normalized_embeddings):
            similar_papers_df = search_similar_papers(row['input'], 
                                                      model, 
                                                      faiss_index, 
                                                      dataset, 
                                                      top_k=3)
            results.append({
                'input_paragraph': row['input'],
                'input_paper': row['data_source'],
                'similar_papers': similar_papers_df['data_source'].tolist()
            })

    return pd.DataFrame(results)


from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm.autonotebook import tqdm
import pandas as pd

def engine(dataset, model):
    def search_similar_papers(query_paragraph: str, 
                              sentence_model, 
                              faiss_index, 
                              dataset, 
                              top_k: int = 3):
        
        # Embed the query paragraph
        query_embedding = sentence_model.encode([query_paragraph], show_progress_bar=False)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search the FAISS index for the top_k similar vectors, excluding the query itself
        distances, indices = faiss_index.search(query_embedding.astype(np.float32), top_k + 1)
        
        # Skip the first result as it is the query itself and get the top_k results
        similar_indices = indices.flatten()[1:1+top_k]
        similar_distances = distances.flatten()[1:1+top_k]
        
        # Create a DataFrame for the similar papers with distances as scores
        similar_papers = dataset.iloc[similar_indices]
        similar_papers.loc[similar_indices, 'score'] = similar_distances
        
        # Filter out any potential duplicates including the input paper itself
        input_data_source = dataset[dataset['input'] == query_paragraph]['data_source'].iloc[0]
        similar_papers = similar_papers[similar_papers['data_source'] != input_data_source]
        
        return similar_papers[['data_source', 'score']].reset_index(drop=True)

    # Embed all title paragraphs and create the FAISS index
    def create_embeddings_and_faiss_index(dataset,
                                          sentence_model):

        title_paragraphs = dataset[dataset['type'] == 'TITLE_PARAGRAPH']['input'].tolist()
        embeddings = sentence_model.encode(title_paragraphs, show_progress_bar=True)
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Create the FAISS index
        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dimension)
        faiss_index.add(normalized_embeddings.astype(np.float32))

        return embeddings, normalized_embeddings, faiss_index

    # Process the dataset and find similar papers with scores for each title paragraph
    def process_dataset_with_scores(dataset, model):
        # Initialize the sentence model
        sentence_model = SentenceTransformer(model)  

        embeddings, normalized_embeddings, faiss_index = create_embeddings_and_faiss_index(dataset, 
                                                                                           sentence_model)
        
        results = []

        for index, row in dataset[dataset['type'] == 'TITLE_PARAGRAPH'].iterrows():
            if index < len(normalized_embeddings):
                similar_papers_df = search_similar_papers(row['input'], 
                                                        sentence_model, 
                                                        faiss_index, 
                                                        dataset, 
                                                        top_k=3)
                results.append({
                    'input_paragraph': row['input'],
                    'input_paper': row['data_source'],
                    'similar_papers': similar_papers_df['data_source'].tolist(),
                    'scores': similar_papers_df['score'].tolist()
                })

        return pd.DataFrame(results)
    return process_dataset_with_scores(dataset, model)
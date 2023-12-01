#!/bin/env python3

import pandas as pd
from pdfminer.high_level import extract_text
import sys
import re
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm.autonotebook import tqdm

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
    """
    Plots the distribution of token counts in a histogram.

    Args:
    token_counts (list): List of token counts.
    title (str): Title of the plot.
    """
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
    """
    Deduplicates a DataFrame based on text embeddings similarity.

    Args:
    df (pd.DataFrame): DataFrame to deduplicate.
    model (str): Sentence Transformer model for embedding generation.
    threshold (float): Threshold for cosine similarity to consider duplicates.

    Returns:
    pd.DataFrame: Deduplicated DataFrame.
    """

    # Load Sentence Transformer model
    sentence_model = SentenceTransformer(model)
    inputs = df["input"].tolist()

    print("Converting text to embeddings...")

    # Convert texts to embeddings
    embeddings = sentence_model.encode(inputs,
                                       show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    normalized_embeddings = embeddings / np.linalg.norm(embeddings,
                                                        axis=1,
                                                        keepdims=True)
    index.add(normalized_embeddings)

    # Filtering out near-duplicates
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

def make_dataset_from_txt(data_dir):
    """
    Creates a dataset from text files in a specified directory.

    Args:
    data_dir (str): Directory containing text files.

    Returns:
    pd.DataFrame: A DataFrame containing parsed data from all text files.
    """

    # List to store DataFrames from each file
    dfs = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.txt'):
            file_path = os.path.join(data_dir,
                                     file_name)
            with open(file_path, 'r') as file:
                content = file.read()
                # Parse and append DataFrame
                dfs.append(parse_txt(content,
                                file_name))
    # Concatenate all DataFrames
    return pd.concat(dfs, ignore_index=True)

def process_single_file(file_path):
    """
    Processes a single text file and returns structured data.

    Args:
    file_path (str): Path to the text file.

    Returns:
    pd.DataFrame: A DataFrame containing parsed data from the file.
    """


    # Check if the file is a text file
    if file_path.endswith('.txt'):
        with open(file_path, 'r') as file:
            content = file.read()
            return parse_txt(content,
                             os.path.basename(file_path))
    else:
        raise ValueError("File is not a .txt file")

def convert_pdf_to_txt(pdf_path):
    """
    Converts a PDF file to text and splits it into paragraphs.

    Args:
    pdf_path (str): Path to the PDF file.

    Returns:
    tuple: A tuple containing the extracted text and a DataFrame of paragraphs.
    """
    # Extract text from PDF
    text = extract_text(pdf_path)

    # Split the text into paragraphs
    paragraphs = text.split('\n\n')
    # Create a DataFrame
    df = pd.DataFrame(paragraphs, columns=['paragraph'])

    return text, df



def redact_below_author_info(text):
    """
    Removes any text that appears after certain keywords like 'AUTHOR INFORMATION',
    'ACKNOWLEDGMENTS', or 'REFERENCES'.

    Args:
    text (str): The text to be redacted.

    Returns:
    str: The redacted text.
    """

    # Define pattern to search for the specified keywords
    pattern = r'(\n\n■ AUTHOR INFORMATION.*|\n\n■ ACKNOWLEDGMENTS.*|\n\n■ REFERENCES.*)'

    # Find the pattern in the text
    match = re.search(pattern, text, flags=re.DOTALL)
    if match:
        # If found, truncate the text from the start of the pattern
        text = text[:match.start()]

    return text

def redact_specified_parts(text):
    """
    Redacts specific elements from academic papers, like DOI references, author affiliations,
    acknowledgments, specific dates, email addresses, names, links, numbers, figures, and addresses.

    Args:
    text (str): The text to be redacted.

    Returns:
    str: The redacted text.
    """

    # Define patterns for redaction
    patterns_to_redact = [
         # Author affiliations and acknowledgments
        r'Perspective\n\npubs\.acs\.org/JACS\n\n†,‡\n\n§,∥\n\n.*?\n, \n\n',
        # Received section with journal and society names
        r'Received:\n\n.*?© XXXX  Society\n\nA\n\n',
        # Other specific sections
        r'For\n\nB\n\n.*?\n\n\x0cJournal of the  Society\n\n',
        r'DOI: .*/jacs\.5b09974\nJ\. Am\. Chem\. Soc\. XXXX, XXX, XXX−XXX\n\n\x0cJournal of the  Society\n\n',
        r'AUTHOR INFORMATION\n\n.*?Nat\. Biotechnol\. ,  \(\), \.\n',
        # Pattern to remove references to papers
        r'DOI: \d+\.\d+/[a-zA-Z0-9.]+\nJ\. Am\. Chem\. Soc\. XXXX, XXX, XXX−XXX\n\n\\x0cJournal of the American Chemical Society\n\nPerspective\n\n',

    ]

    # Apply redaction for each pattern
    for pattern in patterns_to_redact:
        text = re.sub(pattern, '', text, flags=re.DOTALL)

    # Remove names
    text = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '', text)

    # Remove links
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove numbers
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'\[\d+\]', '', text)

    # Remove figures and addresses
    text = re.sub(r'Figure \d+', '', text)
    text = re.sub(r'\d+ [A-Za-z]+ [A-Za-z]+', '', text)

    text = redact_below_author_info(text)

    return text

# Function to split and filter paragraphs
def split_and_filter_paragraphs(paragraphs,
                                max_length=500,
                                min_length=100):
    """
    Splits and filters paragraphs based on length constraints.

    Args:
    paragraphs (list): List of paragraphs to be processed.
    max_length (int, optional): Maximum length for a chunk of text. Defaults to 500.
    min_length (int, optional): Minimum length to keep a chunk of text. Defaults to 100.

    Returns:
    list: List of processed paragraphs.
    """

    split_paragraphs = []
    for paragraph in paragraphs:
        # Split paragraph into chunks of approximately max_length characters
        for i in range(0, len(paragraph), max_length):
            chunk = paragraph[i:i+max_length]
            # Only add chunks that are at least min_length characters long
            if len(chunk) >= min_length:
                split_paragraphs.append(chunk)
    return split_paragraphs

def remove_sentences_not_starting_with_capital(text):
    """
    Filters out sentences from a text that do not start with a capital letter.

    Args:
    text (str): Text containing multiple sentences.

    Returns:
    str: Filtered text with only properly capitalized sentences.
    """

    # Split text into sentences using regular expression
    sentences = re.split(r'(?<=[.!?]) +', text)

    # Keep sentences that start with a capital letter
    filtered_sentences = [sentence for sentence in sentences if sentence[0].isupper()]

    # Join the filtered sentences back into a single string
    return ' '.join(filtered_sentences)

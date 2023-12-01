#!/bin/env python3

import sys
import os
sys.path.append("..")
sys.path.append("../lib")
sys.path.append("../services")
sys.path.append("../data")
import preproc as pre
import search_service as search
import summarize as summa


def generate_similar_papers(pdf_path, model_name, top_k=3):
    """
    Generates a list of papers similar to the given PDF.

    This function reads text from a PDF file, converts it to text, and then
    uses a pre-defined model to find the top 'k' similar papers from a dataset.

    Args:
        pdf_path (str): The file path of the PDF to analyze.
        model_name (str): The name of the model to use for generating embeddings.
        top_k (int): Number of top similar papers to return.

    Returns:
        list: A list of similar papers with their scores.
    """

    # Convert PDF to text
    text, _ = pre.convert_pdf_to_txt(pdf_path)

    # Load the dataset of extracted text
    dataset = pre.make_dataset_from_txt('../data/extracted-text/')

    # Find and return the top 'k' similar papers
    similar_papers = search.find_similar_papers_for_input(text, model_name, dataset, top_k)
    return similar_papers


def generate_summary(pdf_path):
    """
    Generates a summary from the content of a given PDF.

    This function reads a PDF, converts it into text, applies redaction to sensitive
    parts, and then processes the paragraphs to generate a summary.

    Args:
        pdf_path (str): The file path of the PDF to summarize.

    Returns:
        str: The generated summary of the PDF content.
    """

    # Convert PDF to text and store it in a DataFrame
    _, df_pdf = pre.convert_pdf_to_txt(pdf_path)

    # Apply redaction to the paragraphs in the DataFrame
    df_pdf['input_redact'] = df_pdf['paragraph'].apply(pre.redact_specified_parts)

    # Filter paragraphs longer than 100 characters
    paragraphs = df_pdf.input_redact.tolist()
    paragraphs = [paragraph for paragraph in paragraphs if len(paragraph) > 100]

    # Split and filter paragraphs for further processing
    processed_paragraphs = pre.split_and_filter_paragraphs(paragraphs)

    # Generate and return the summary
    combined_summary = summa.llama_2_7b_q2_default(processed_paragraphs)
    return combined_summary

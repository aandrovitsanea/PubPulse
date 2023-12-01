#!/bin/env python3

import sys
sys.path.append("..")
sys.path.append("../lib")
sys.path.append("../data")
sys.path.append("../models")
import importlib
import preproc as pre
import search_service as search
import os
import pandas as pd
import re

def llama_2_7b_q2(processed_paragraphs):
    """
    Generates a summary from a list of processed paragraphs using Llama 2-7B model with custom parameters.

    This function tokenizes each paragraph, checks for length constraints, and generates a summary
    for each paragraph using the Llama 2-7B model. It concatenates these summaries into a single string.

    Args:
        processed_paragraphs (list): List of pre-processed paragraphs for summarization.

    Returns:
        str: A combined summary of all paragraphs.
    """

    # Import necessary modules
    from llama_cpp import Llama
    from transformers import AutoTokenizer
    from tqdm import tqdm


    # Initialize tokenizer and Llama model
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
    llm_l = Llama(model_path="../models/llama-2-7b.Q2_K.gguf",
                  n_ctx=500,
                  n_gpu_layers=-1)

    summary = []
    for paragraph in tqdm(processed_paragraphs, desc="Summarizing paragraphs"):
        # Tokenize and check paragraph length
        tokens = tokenizer.tokenize(paragraph)
        if len(tokens) > 512:
            continue

        # Formulate prompt
        formatted_prompt = "Summarize this paragraph: {}".format(paragraph)

        # Generate summary
        output = llm_l(formatted_prompt,
                       max_tokens=60,
                       stop=["\n"],
                       echo=False)
        generated_text = output['choices'][0]['text'].strip()

        # Append non-empty summaries
        if generated_text:
            summary.append(generated_text)

    # Combine and groom summaries
    combined_summary = ' '.join(summary)
    combined_summary_groomed = pre.remove_sentences_not_starting_with_capital(combined_summary)

    return combined_summary_groomed



def llama_2_7b_q2_default(processed_paragraphs):
    """
    Generates a summary using the Llama 2-7B model with default parameters.

    This function processes each paragraph, creating a summary with the Llama 2-7B model.
    The summaries are then concatenated into a single string.

    Args:
        processed_paragraphs (list): List of pre-processed paragraphs for summarization.

    Returns:
        str: A combined summary of all paragraphs.
    """

    # Import necessary modules
    from llama_cpp import Llama
    from transformers import AutoTokenizer
    from tqdm import tqdm
    # Initialize tokenizer and Llama model
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
    llm = Llama(model_path="../models/llama-2-7b.Q2_K.gguf")


    summary = []
    for paragraph in tqdm(processed_paragraphs, desc="Summarizing paragraphs"):
        # Tokenize and check paragraph length
        tokens = tokenizer.tokenize(paragraph)
        if len(tokens) > 512:
            continue

        # Formulate prompt
        formatted_prompt = "Q: Create a summary of this {}. Summary: ".format(paragraph)

        # Generate summary
        output = llm(formatted_prompt,
                     max_tokens=150,
                     stop=["Q:", "\n"],
                     echo=False)
        generated_text = output['choices'][0]['text']
        summary_index = generated_text.find("Summary:")
        summary_text = generated_text[summary_index + len("Summary:"):].strip()

        # Append non-empty summaries
        if generated_text:
            summary.append(generated_text)

    # Combine and groom summaries
    combined_summary = ' '.join(summary)
    combined_summary_groomed = pre.remove_sentences_not_starting_with_capital(combined_summary)

    return combined_summary_groomed

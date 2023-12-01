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
    from llama_cpp import Llama
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")

    llm_l = Llama(model_path="../models/llama-2-7b.Q2_K.gguf",
                n_ctx=500,
                n_gpu_layers=-1)

    from tqdm import tqdm  # Import tqdm

    summary = []
    for paragraph in tqdm(processed_paragraphs, desc="Summarizing paragraphs"):
        # Tokenize the individual paragraph to check its length
        tokens = tokenizer.tokenize(paragraph)

        # Skip paragraphs that are too long
        if len(tokens) > 512:
            continue

        # Simplified prompt
        formatted_prompt = "Summarize this paragraph: {}".format(paragraph)

        # Generate summary with a possibly increased max_tokens
        output = llm_l(
            formatted_prompt,
            max_tokens=60,  # Increased max_tokens
            stop=["\n"],
            echo=False
        )

        # Extracting summary text
        generated_text = output['choices'][0]['text'].strip()

        # Append only if generated_text is not empty
        if generated_text:
            summary.append(generated_text)

    # Concatenating with space and ensuring each summary starts on a new line
    combined_summary = ' '.join(summary)

    combined_summary_groomed = pre.remove_sentences_not_starting_with_capital

    return combined_summary_groomed


def llama_2_7b_q2_default(processed_paragraphs):
    from llama_cpp import Llama
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")

    llm = Llama(model_path="../models/llama-2-7b.Q2_K.gguf")

    from tqdm import tqdm  # Import tqdm

    summary = []
    for paragraph in tqdm(processed_paragraphs, desc="Summarizing paragraphs"):
        # Tokenize the individual paragraph to check its length
        tokens = tokenizer.tokenize(paragraph)

        # Skip paragraphs that are too long
        if len(tokens) > 512:
            continue

        # Simplified prompt
        formatted_prompt = "Q: Create a summary of this {}. Summary: ".format(paragraph)

        # Generate summary
        output = llm(
            formatted_prompt,
            max_tokens=150,
            stop=["Q:", "\n"],
            echo=False
        )

        generated_text = output['choices'][0]['text']
        summary_index = generated_text.find("Summary:")
        summary_text = generated_text[summary_index + len("Summary:"):].strip()

        # Append only if generated_text is not empty
        if generated_text:
            summary.append(generated_text)

    # Concatenating with space and ensuring each summary starts on a new line
    combined_summary = ' '.join(summary)

    combined_summary_groomed = pre.remove_sentences_not_starting_with_capital

    return combined_summary_groomed

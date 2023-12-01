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


def generate_similar_papers(pdf_path,
                            model_name,
                            top_k=3):
    text, _ = pre.convert_pdf_to_txt(pdf_path)
    dataset = pre.make_dataset_from_txt('../data/extracted-text/')
    similar_papers = search.find_similar_papers_for_input(text,
                                                          model_name,
                                                          dataset, top_k)
    return similar_papers

def generate_summary(pdf_path):
    _, df_pdf = pre.convert_pdf_to_txt(pdf_path)

    # Apply redaction and filter paragraphs
    df_pdf['input_redact'] = df_pdf['paragraph'].apply(pre.redact_specified_parts)
    paragraphs = df_pdf.input_redact.tolist()
    paragraphs = [paragraph for paragraph in paragraphs if len(paragraph) > 100]
    processed_paragraphs = pre.split_and_filter_paragraphs(paragraphs)

    # Generate summary
    combined_summary = summa.llama_2_7b_q2_default(processed_paragraphs)
    return combined_summary

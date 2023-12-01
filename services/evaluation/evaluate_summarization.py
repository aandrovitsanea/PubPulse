#!/bin/env python3

from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import pandas as pd
import sys
import os
sys.path.append("..")
sys.path.append("../../lib")
sys.path.append("../../models")
sys.path.append("../../data")
sys.path.append("../../services")
import preproc as pre
import pipelines as pipe
import summarize as summa

def generate_summaries_abstracts_for_all(path_data): # '../data/extracted-text/'

    dataset = pre.make_dataset_from_txt(path_data)
    dataset['input_redact'] = dataset['input'].apply(pre.redact_specified_parts)

    # Extract abstract per paper
    combined_abstracts = []
    for article in list(set(dataset.data_source)):
        combined_abstracts.append(' '.join(dataset[(dataset.data_source == article) &\
                                    (dataset.type == 'ABSTRACT')].input_redact.tolist()))

    # Extract paragraphs per paper
    paragraphs_all_papers = {}

    for article in dataset['data_source'].unique():
        paragraphs = dataset[(dataset['data_source'] == article) & (dataset['type'] != 'ABSTRACT')]['input_redact'].tolist()
        paragraphs = [paragraph for paragraph in paragraphs if len(paragraph) > 100]
        paragraphs = pre.split_and_filter_paragraphs(paragraphs)

        article_key = article if isinstance(article, str) else str(article)
        paragraphs_all_papers[article_key] = paragraphs

    # Summary per paper

    summaries = []
    for key in paragraphs_all_papers.keys():
        summaries.append(summa.llama_2_7b_q2_default(paragraphs_all_papers[key]))

    return summaries, combined_abstracts


def evaluate_summarization(summary,
                           abstract):

    bleu_score = sentence_bleu([abstract.split()],
                               summary.split())
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'],
                                      use_stemmer=True)
    rouge_score = scorer.score(abstract,
                               summary)

    print(f"BLEU score: {bleu_score}")
    print(f"ROUGE-1 and ROUGE-L: {rouge_score}")

def apply_eval():
    summaries, combined_abstracts = generate_summaries_abstracts_for_all('../../data/extracted-text/')
    for summary, abstract in zip(summaries, combined_abstracts):
        evaluate_summarization(summary, abstract)


if __name__ == "__main__":
    apply_eval()

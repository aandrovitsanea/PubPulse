#!/bin/env python3

from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import pandas as pd
import sys
import os
sys.path.append("..")
sys.path.append("../../lib")
sys.path.append("../../data/")
import preproc as pre
import pipeline as pipe


def generate_summaries_for_all(df):
     # take abstract from the df
    df = pre.make_dataset_from_txt("../../data/")
    # Add a new column for summaries
    df['summary'] = ''

    for index, row in df.iterrows():
        pdf_path = row['pdf_path']  # Ensure this column exists in your DataFrame
        summary = generate_summary(pdf_path)
        df.at[index, 'summary'] = summary

    return df


def evaluate_summarization(df.summary,
                           df.abstract):

    bleu_score = sentence_bleu([df.abstract.split()],
                               df.summary.split())
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'],
                                      use_stemmer=True)
    rouge_score = scorer.score(df.abstract,
                               df.summary)

    print(f"BLEU score: {bleu_score}")
    print(f"ROUGE-1 and ROUGE-L: {rouge_score}")

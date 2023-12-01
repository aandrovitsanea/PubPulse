#!/bin/env python3

import sys
import os
sys.path.append("..")
sys.path.append("../services")

import unittest
import pipelines as pipe

class TestGenerateSimilarPapers(unittest.TestCase):

    def setUp(self):
        # Set up necessary variables for the test
        self.pdf_path = "The Safety and Effectiveness of mRNA Vaccines Against SARS-CoV-2 - PMC.pdf"
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def test_generate_similar_papers(self):
        # Test the generate_similar_papers function
        similar_papers = pipe.generate_similar_papers(self.pdf_path, self.model_name, top_k=3)

        # Assertions
        self.assertIsInstance(similar_papers, list, "Output should be a list")
        self.assertEqual(len(similar_papers), 3, "There should be 3 similar papers")
        for paper in similar_papers:
            self.assertIn('data_source', paper, "Each item should have a data_source key")
            self.assertIn('score', paper, "Each item should have a score key")
            self.assertGreaterEqual(paper['score'], 0, "Score should be non-negative")
            self.assertLessEqual(paper['score'], 1, "Score should be at most 1")

if __name__ == '__main__':
    unittest.main()

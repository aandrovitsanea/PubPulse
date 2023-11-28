#!/bin/bash

# Copying the extracted-text directory
gcloud storage cp -r gs://llm-technical-test-data/extracted-text data

# Copying the raw-pdf directory
gcloud storage cp -r gs://llm-technical-test-data/raw-pdf data


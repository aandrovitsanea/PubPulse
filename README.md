# PubPulse: Research Reinforcement Tool

PubPulse streamlines the research process by swiftly identifying and recommending the most relevant and recent academic papers to enrich and update your existing work. It harnesses the power of AI to keep your research at the forefront of academic discovery and discourse. Apart from recommending you relevant papers based on a given paper, it uses `llama-2` llm to provide a summary of the paper you



## Environment Setup

### Prerequisites
- Python 3.6+
- Git (for version control)
- Access to Google Cloud Storage

### Installation

1. Set up a Python virtual environment and activate it:

```bash
python3 -m venv pubpulse
source pubpulse/bin/activate  # On Windows use `pubpulse\Scripts\activate`
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

3. Clone the repository:

```shell
git clone git@github.com:aandrovitsanea/PubPulse.git
cd pubpulse
   ```

4. Download the training set:

```bash
./lib/get_raw_data.sh
```

5. Download the model and store it in folder `models`:

```bash
cd models
wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q2_K.gguf

```

## Structure

The project repository is structured as follows:

- `apps/:` Contains application scripts.
- `data/:` Directory containing the raw and processed data.
- `embedding_storage/:` Stores vectore database and embeddings for inference.
- `img/:` Images and graphical resources used in the application.
- `lib/:` Library code for the project, shared across different applications and scripts.
- `models/:` Contains the pretrained machine learning models.
- `notebooks/:` Jupyter notebooks with exploratory data analysis and model prototyping.
- `services/:` Services supporting the main application.
- `dashboard.py:` The main Streamlit application script.
- `README.md:` Documentation file providing an overview, setup instructions, and additional information about the project.
- `requirements.txt:` Python dependencies for the project.

## Usage

To run the application, navigate to the project directory and execute the following:

```bash
python3 dashboard.py
```
This launches the web application, and you can interact with it through your web browser by uploading a PDF file of the paper in question.

Alternatively you can use each service separately in your console:

- Top 3 papers

```bash
cd services
ipython3
```

There you are in a python environment.
Run:

```python
import pipelines as pipe
similar_papers = pipe.generate_similar_papers("../data/raw-pdf/PMC8198544.pdf",
                                              "sentence-transformers/all-MiniLM-L6-v2",
                                              top_k=3)
```

- Summary

```bash
cd services
ipython3

```

There you are in a python environment.
Run:

```python
import pipelines as pipe
combined_summary = pipe.generate_summary("../data/raw-pdf/PMC8198544.pdf")

```

## Unit tests

### TestGenerateSimilarPapers Unit Test

#### Description

This unit test, named `TestGenerateSimilarPapers`, evaluates the functionality of the `generate_similar_papers` function within the `pipelines` module. The test is designed to assess the capability of the function to identify similar papers based on the content of a given PDF file. The test setup includes defining the PDF file path and the Sentence Transformer model name.

#### Test Procedure

1. **Set Up:**
   - Define the PDF file path and the Sentence Transformer model name.
   - Initialize necessary variables for the test.

2. **Test Functionality:**
   - Call the `generate_similar_papers` function with the specified PDF file and model name, requesting the top 3 similar papers.
   - Verify that the output is a list.
   - Confirm that the list contains exactly 3 items.
   - For each item in the list:
     - Ensure it has a 'data_source' key.
     - Ensure it has a 'score' key.
     - Verify that the score is non-negative.
     - Verify that the score is at most 1.

#### Purpose

This unit test aims to validate the accuracy and reliability of the paper similarity generation process. It provides a comprehensive check on the expected structure of the output, ensuring that each result includes essential information such as data source and similarity score within the defined bounds.

#### Execution

Run the test script to assess the functionality of the `generate_similar_papers` function and confirm its adherence to the expected behavior.

## Tool Evaluation Pipeline

### Setup
- Ensure you are working on a `virtual env`.
- Ensure all dependencies are installed as per `requirements.txt`.

### Step 1: Data Preparation
- Use `lib/preproc.py` for preparing datasets.
- For PDF files, use `convert_pdf_to_txt()` to convert them to text.
   - **Purpose:** Converts a PDF file into text, which is essential for processing documents in your pipeline.
   - **Evaluation:** Ensure that the text extraction is accurate. Manually check a few converted documents to see if there are missing or misinterpreted segments.
   -**Implementation:**

   ```python
   pdf_path = "../data/raw-pdf/sample.pdf"
   text, df = convert_pdf_to_txt(pdf_path)
   # Manually inspect 'text' and 'df' for accuracy
   ```

### Step 2: Generating Embeddings
- Use `services/create_embeddings_and_faiss_index.py` for preparing datasets.
   - **Purpose:** Generates embeddings for text data and creates a FAISS index for efficient similarity searches.
   - **Evaluation:** Inspect embeddings for consistency and diversity. Plotting distributions of some embedding dimensions can give insights into the embedding space.
   - **Implementation:**

   ```python
   dataset = make_dataset_from_txt("../data/extracted-text/")
   sentence_model = SentenceTransformer("model_name")
   embeddings, faiss_index = create_embeddings_and_faiss_index(dataset, sentence_model)
   # Optionally plot embeddings for inspection
   ```

### Step 3: Summarization Evaluation with ROUGE and BLEU

- **Purpose:** Creates a summary for a given document.
- **Evaluation:** Compare the generated summaries with human-written summaries (if available) for the same documents. Use metrics like ROUGE or BLEU for a quantitative assessment.
- **Implementation:**
   - Extract existing abstracts from your documents. These will serve as reference summaries.
   - Use function `generate_summary` from `services/pipeline.py` to generate summary.
   - Compute **BLEU Score:** Compare n-gram of the generated text with the n-gram of the reference text to calculate a score.
   ```python
   from nltk.translate.bleu_score import sentence_bleu

   bleu_score = sentence_bleu([dataset.abstract.split()],
   generated_summary.split())
   print(f"BLEU score: {bleu_score}")
   ```
   - Compute **ROUGE Score:** Focus on recall (coverage of reference summary by the generated summary).
   ```python
   from rouge_score import rouge_scorer

   scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
   rouge_score = scorer.score(dataset.abstract, generated_summary)
   print(f"ROUGE-1 and ROUGE-L: {rouge_score}")
   ```
   - **Evaluation:** High **BLEU** and **ROUGE** scores indicate better quality summaries.
   Compare the scores across various documents to evaluate the consistency of the summarization model.


### Conclusion
- This pipeline aids in evaluating the effectiveness and accuracy of the research reinforcement tool.
- Regular use of this pipeline ensures consistent quality and performance of the tool.


## Contributions
To contribute to this project, please create a branch and submit a pull request.

## Contact
For any queries regarding the setup or execution of the project, please contact me at anna.androvitsanea@gmail.com.



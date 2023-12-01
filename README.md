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

### Structure

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

### Usage

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

### Unit tests

#### TestGenerateSimilarPapers Unit Test

##### Description

This unit test, named `TestGenerateSimilarPapers`, evaluates the functionality of the `generate_similar_papers` function within the `pipelines` module. The test is designed to assess the capability of the function to identify similar papers based on the content of a given PDF file. The test setup includes defining the PDF file path and the Sentence Transformer model name.

##### Test Procedure

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

##### Purpose

This unit test aims to validate the accuracy and reliability of the paper similarity generation process. It provides a comprehensive check on the expected structure of the output, ensuring that each result includes essential information such as data source and similarity score within the defined bounds.

##### Execution

Run the test script to assess the functionality of the `generate_similar_papers` function and confirm its adherence to the expected behavior.


### Contributions
To contribute to this project, please create a branch and submit a pull request.

### Contact
For any queries regarding the setup or execution of the project, please contact me at anna.androvitsanea@gmail.com.



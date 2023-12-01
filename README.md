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

4. Download the training set

```bash
./lib/get_raw_data.sh
```
### Structure

The project repository is structured as follows:

- `data/:` Directory containing the raw and processed data.
- `models/:` Contains the trained machine learning models.
- `notebooks/:` Jupyter notebooks with exploratory data analysis and model prototyping.
- `lib/:` Library code for the project.
- `app.py:` The main Streamlit application script.
- `requirements.txt:` Python dependencies for the project.

### Usage

To run the application, navigate to the project directory and execute the following:

```bash
python3 dashboardv2.py
```
This will launch the web application, and you can interact with it through your web browser.

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



### Contributions
To contribute to this project, please create a branch and submit a pull request.

### Contact
For any queries regarding the setup or execution of the project, please contact us at anna.androvitsanea@gmail.com.



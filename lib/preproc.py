#!/bin/env python
import pandas as pd

# Define the function to parse the text
def parse_txt(text, data_source):
    """
    Parses the provided text into structured data and captures different sections of a research paper.

    This function separates the text content into distinct sections such as abstracts, title paragraphs,
    and description tables. It identifies each section by the presence of specific markers ('ABSTRACT',
    'TITLE PARAGRAPH:', and 'DESCRIPTION TABLE:'), then extracts the title (if any) and the associated
    content. The parsed data is returned as a pandas DataFrame with the columns 'type', 'title', 'input',
    and 'data_source'.

    Parameters:
    - text (str): The text content of a document to parse.
    - data_source (str): The name or identifier for the text data source (e.g., the file name).

    Returns:
    - DataFrame: A pandas DataFrame with the structured data extracted from the text. The DataFrame
      contains columns for the section type ('type'), the title of the section ('title'), the content
      of the section ('input'), and the data source ('data_source').

    The 'type' column categorizes the section into 'TITLE_PARAGRAPH', 'DESCRIPTION_TABLE', 'ABSTRACT',
    or 'Other'. The 'title' column includes the title for 'TITLE_PARAGRAPH' and 'DESCRIPTION_TABLE'
    sections, and default titles for 'ABSTRACT' or 'Other' sections. The 'input' column contains the
    actual text content for each section, and the 'data_source' column records the provided data source
    name for all entries.

    Example:
    ```
    text_content = "ABSTRACT\n ... \n----\nTITLE PARAGRAPH: Introduction\n ... \n----\n"
    data_source_name = "example.txt"
    parsed_df = parse_text(text_content, data_source_name)
    ```
    """

  # Split text into sections
    sections = text.split('\n----\n')

    # Prepare a DataFrame to store the parsed data
    parsed_data = {
        'type': [],
        'title': [],
        'input': [],
        'data_source': [],
    }

    # Iterate through the sections to parse the content
    for section in sections:
        section = section.strip()
        if 'TITLE PARAGRAPH:' in section:
            # Remove the title identifier and extract the title
            title = section.split('\n', 1)[0].replace('TITLE PARAGRAPH:', '').strip()
            content = section.split('\n', 1)[1].strip() if '\n' in section else ''
            parsed_data['input'].append(content)
            parsed_data['type'].append('TITLE_PARAGRAPH')
            parsed_data['title'].append(title)
        elif 'DESCRIPTION TABLE:' in section:
            # Remove the table identifier
            title = 'DESCRIPTION_TABLE'
            content = section.replace('DESCRIPTION TABLE:', '').strip()
            parsed_data['input'].append(content)
            parsed_data['type'].append('DESCRIPTION_TABLE')
            parsed_data['title'].append(title)
        else:
            # For abstract or any other type that is not defined above
            title = 'ABSTRACT' if 'ABSTRACT' in section else 'Other'
            parsed_data['input'].append(section)
            parsed_data['type'].append('ABSTRACT' if 'ABSTRACT' in section else 'Other')
            parsed_data['title'].append(title)

        parsed_data['data_source'].append(data_source)

    return pd.DataFrame(parsed_data) # return parsed data as DataFrame

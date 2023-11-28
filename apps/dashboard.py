# Import required libraries
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc  # Import Bootstrap components
import sys
sys.path.append("..")
sys.path.append("../img")

# Initialize the Dash app with Bootstrap
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                assets_folder='../img')

# Define the layout of the app
app.layout = dbc.Container([
    dbc.Row(
        dbc.Col(
            html.Img(src=app.get_asset_url('logo.png'),
                     style={'width': '25%',
                            'height': 'auto',
                            'display': 'block',
                            'margin-left': 'auto',
                            'margin-right': 'auto'}),
            width=12
        ),
        justify="center"
    ),
    dbc.Row(
        dbc.Col(
            html.H1('Research Reinforcement Tool',
                    className='text-center mb-4'),
            width=12
        ),
        justify="center"
    ),
    dcc.Upload(
        id='upload-data',
        children=html.Button('Upload PDF Paper',
                             className='btn btn-primary btn-lg',
                             style={'display': 'block',
                                    'margin-left': 'auto',
                                    'margin-right': 'auto'}),
        multiple=False,  # Allow one file to be uploaded
        className='d-block my-4'
    ),
    dbc.Row([  # Use Rows and Columns for better layout structure
        dbc.Col([
            dcc.Checklist(
                options=[{'label': '  Give me top 3 most similar papers',
                          'value': 'TOP3'}],
                value=[],
                id='checkbox-top-3',
                inline=True  # Aligns the checkbox and label on the same line
            ),
            dbc.Alert(id='output-top-3',
                      color="light",
                      style={'height': '100px'}),
        ], width=6),
        dbc.Col([
            dcc.Checklist(
                options=[{'label': '  Give me a summary',
                          'value': 'SUM'}],
                value=[],
                id='checkbox-summary',
                inline=True
            ),
            dbc.Alert(id='output-summary',
                      color="light",
                      style={'height': '100px'}),
        ], width=6),
    ])
], fluid=True)


# Define callback for uploading and processing the PDF
@app.callback(
    Output('output-summary', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    Input('checkbox-summary', 'value')
)
def process_pdf(contents, filename, summary_requested):
    if contents is not None:
        # Process the PDF and generate a summary
        summary = "This is where the summary will appear."
        if 'SUM' in summary_requested:
            return html.Div(summary,
                            style={'whiteSpace': 'pre-line'})
        else:
            return html.Div('Summary not requested.',
                            style={'whiteSpace': 'pre-line'})

# Define callback for finding similar articles
@app.callback(
    Output('output-top-3', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    Input('checkbox-top-3', 'value')
)

def find_similar_articles(contents, filename, top_3_requested):
    if contents is not None:
        # Implement the logic to find similar articles
        # This could involve machine learning models, database queries, etc.
        similar_articles = ["Article 1", "Article 2", "Article 3"]
        if 'TOP3' in top_3_requested:  # Check if 'TOP3' is in the list of values
            return html.Ul([html.Li(article) for article in similar_articles])
        else:
            return html.Div('Top-3 recommendations not requested.')

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

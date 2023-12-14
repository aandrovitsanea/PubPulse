#!/bin/env python3

# Import required libraries
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import sys
import os
sys.path.append("..")
sys.path.append("../img")
sys.path.append("../lib")
sys.path.append("../services")
sys.path.append("../data")
import preproc as pre
import search_service as search
import summarize as summa
import pipelines as pipe

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
        multiple=False,
        className='d-block my-4'
    ),
    dbc.Row([
        dbc.Col([
            html.H3('Top 3 Relevant Papers',
                    className='text-center'),
            dcc.Loading(
                id="loading-top-3",
                children=[dbc.Alert(id='output-top-3',
                                    color="light")],
                type="default",
            )
        ], width=6),
        dbc.Col([
            html.H3('Summary',
                    className='text-center'),
            dcc.Loading(
                id="loading-summary",
                children=[dbc.Alert(id='output-summary',
                                    color="light")],
                type="default",
            )
        ], width=6)
    ])
], fluid=True)

# Callback for Top 3 Relevant Papers
@app.callback(
    Output('output-top-3', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)

def update_top_3(contents, filename):
    papers_list = html.Div()  # Default empty div
    if contents is not None:
        # Assuming the file is saved in '../data/raw-pdf/' with the same filename
        saved_file_path = f'../data/raw-pdf/{filename}'
        similar_papers = pipe.generate_similar_papers(saved_file_path,
                                                      "sentence-transformers/all-MiniLM-L6-v2",
                                                      top_k=3)

        # Generate the list of similar papers for display
        papers_list = html.Ul([html.Li(f"{paper['data_source']} (Score: {paper['score']:.2f})") for paper in similar_papers])
    return html.Div(papers_list, style={'whiteSpace': 'pre-line'})

# Callback for Summary Generation
@app.callback(
    Output('output-summary', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)

def update_summary(contents, filename):
    combined_summary = html.Div()  # Default empty div
    if contents is not None:
        # Assuming the file is saved in '../data/raw-pdf/' with the same filename
        saved_file_path = f'../data/raw-pdf/{filename}'
        summary = pipe.generate_summary(saved_file_path)
        combined_summary = html.Div(summary,
                                    style={'whiteSpace': 'pre-line'})
    return combined_summary

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

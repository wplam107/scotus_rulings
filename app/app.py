import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import pickle

from app_functions import *

# Dash tutorial CSS, major thanks to Chris Parmer of Plotly/Dash for his answers on the community forums
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Load all data
# List of all courts
with open('./data/courts.p', 'rb') as f:
    courts = pickle.load(f)

# DataFrame of each court projected onto 2 principal components
with open('./data/pca_df.p', 'rb') as f:
    pca_df = pickle.load(f)

# List of similarity matrices of all courts
with open('./data/sim_mats.p', 'rb') as f:
    sim_mats = pickle.load(f)

# Similarity DataFrame of all justices
with open('./data/sim_df.p', 'rb') as f:
    sim_df = pickle.load(f)

# Cases DataFrame
with open('./data/multi_df.p', 'rb') as f:
    big_df = pickle.load(f)

justices = list(sim_df.index)

# Get justice options
justice_options = ['All'] + get_justices(courts)

markdown_1 = '''Choose a Supreme Court Justice who served between 1999-2019.
The courts the justice has served on will appear in the dropdown menu on the right.
Below is a 2-component scatter plot for the courts the selected justice has served on.
More on principal component analysis (PCA) at [Wikipedia](https://en.wikipedia.org/wiki/Principal_component_analysis).
'''

markdown_2 = '''Choose a Supreme Court by justice composition of the court.
The selectable courts are the court compositions that the selected justice has served on.
The heatmaps below are cosine similarity heatmaps (0 = no similarity, 1 = exactly the same) for the selected court.
More on cosine similarity at [Wikipedia](https://en.wikipedia.org/wiki/Cosine_similarity).
'''

app.layout = html.Div(children=[
    html.Div(children=[
        html.H1(
            html.A('SCOTUS Dashboard', href='https://github.com/wplam107/scotus_rulings')
        )
    ]),

    html.Div(children=[
        html.Div(children=[
            dcc.Graph(
                id='network-graph',
                figure=plot_network(sim_df, big_df)
            )
        ], className='six columns'
        ),
        html.Div(children=[
            dcc.Graph(
                id='all-sim',
                figure=sim_heatmap(sim_df, justices)
            )
        ], className='six columns'
        ),
    ], className='row'
    ),

    html.Div(children=[
        html.Div(children=[dcc.Markdown(children=markdown_1)], className='six columns'),
        html.Div(children=[dcc.Markdown(children=markdown_2)], className='six columns'),
    ], className='row'
    ),

    html.Hr(),

    html.Div(children=[
        html.Div(children=[
            html.Div(children='Select Justice'),
            dcc.Dropdown(
                id='justice-select',
                options=[{'label': j, 'value': j} for j in justice_options],
                value = 'All',
                clearable=False,
            ),
        ], className='six columns'),
        html.Div(children=[
            html.Div(children='Select Court'),
            dcc.Dropdown(
                id='court-select',
                value = 0,
                clearable=False,
            ),
        ], className='six columns'),
    ], className='row'
    ),

    html.Div(children=[
        html.Div(dcc.Graph(id='pca-graph'), className='six columns'),
        html.Div(dcc.Graph(id='heatmap-graph'), className='six columns'),
    ], className='row'
    ),
])

@app.callback(
    [Output('court-select', 'options'), Output('pca-graph', 'figure')],
    [Input('justice-select', 'value')]
)
def set_court_options(selected_justice):
    if selected_justice == 'All':
        all_courts = list(range(len(courts)))
        fig = animated_2comp(pca_df, all_courts)
        return [ {'label': ' '.join(courts[c]), 'value': c} for c in all_courts ], fig

    else:   
        j_courts = []
        for i, c in enumerate(courts):
            if selected_justice in c:
                j_courts.append(i)
        fig = animated_2comp(pca_df, j_courts)
        return [ {'label': ' '.join(courts[c]), 'value': c} for c in j_courts ], fig

@app.callback(
    Output('heatmap-graph', 'figure'),
    [Input('court-select', 'value')]
)
def update_heatmap(selected_court):
    justices = courts[selected_court]
    sim_mat = sim_mats[selected_court]
    fig = sim_heatmap(sim_mat, justices)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
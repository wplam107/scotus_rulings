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
server = app.server

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

# Cases worked together DataFrame
with open('./data/total_df.p', 'rb') as f:
    total_df = pickle.load(f)

justices = list(sim_df.index)

# Get justice options
justice_options = ['All'] + get_justices(courts)

markdown_1 = '''Below is a network graph where each edge (the line from point to point) illustrates
the number of cases worked between 2 justices, stronger lines mean more cases worked together.
The size of each node indicates the number of cases a justice has participated in.
'''

markdown_2 = '''The heatmap below is the similarity of justices over the entire duration 1999-2019.
Empty (white) blocks indicated that the similarity between justices is non-existent since they have not
participated in cases together.  Higher values (from 0 to 1) indicate great similarity between 2 justices.
'''

markdown_3 = '''Choose a Supreme Court Justice who served between 1999-2019.
The courts the justice has served on will appear in the dropdown menu on the right.
The selectable courts are the court compositions that the selected justice has served on.
'''

markdown_4 = '''Below are a most similar justices bar graph,
a 2-component scatter plot for the courts the selected justice has served on,
and a heatmap of selected court similarities.
More on principal component analysis and cosine similarity at
[Wikipedia PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) and
[Wikipedia Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity).
'''

app.layout = html.Div(children=[
    html.Div(children=[
        html.H1(
            html.A('SCOTUS Dashboard', href='https://github.com/wplam107/scotus_rulings')
        )
    ]),

    html.Div(children=[
        html.Div(children=[dcc.Markdown(children=markdown_1)], className='six columns'),
        html.Div(children=[dcc.Markdown(children=markdown_2)], className='six columns'),
    ], className='row'
    ),

    html.Div(children=[
        html.Div(children=[
            dcc.Graph(
                id='network-graph',
                figure=plot_network(total_df, big_df)
            )
        ], className='six columns'
        ),
        html.Div(children=[
            dcc.Graph(
                id='all-sim',
                figure=sim_heatmap(sim_df, justices, all_justices=True)
            )
        ], className='six columns'
        ),
    ], className='row'
    ),

    html.Hr(),

    html.Div(children=[
        html.Div(children=[dcc.Markdown(children=markdown_3)], className='six columns'),
        html.Div(children=[dcc.Markdown(children=markdown_4)], className='six columns'),
    ], className='row'
    ),

    html.Div(children=[
        html.Div(children=[
            html.Div(children='Select Justice'),
            dcc.Dropdown(
                id='justice-select',
                options=[{'label': j, 'value': j} for j in justice_options],
                value='All',
                clearable=False,
            ),
        ], className='six columns'),
        html.Div(children=[
            html.Div(children='Select Court'),
            dcc.Dropdown(
                id='court-select',
                value=0,
                clearable=False,
            ),
        ], className='six columns'),
    ], className='row'
    ),

    html.Div(children=[
        html.Div(dcc.Graph(id='most-similar'), className='four columns'),
        html.Div(dcc.Graph(id='pca-graph'), className='four columns'),
        html.Div(dcc.Graph(id='heatmap-graph'), className='four columns'),
    ], className='row'
    ),
])

@app.callback(
    [Output('court-select', 'options'), Output('pca-graph', 'figure'), Output('most-similar', 'figure')],
    [Input('justice-select', 'value')]
)
def set_court_options(selected_justice):
    if selected_justice == 'All':
        all_courts = list(range(len(courts)))
        fig1 = animated_2comp(pca_df, all_courts)
        fig2 = default_pic()
        return [ {'label': ' '.join(courts[c]), 'value': c} for c in all_courts ], fig1, fig2

    else:   
        j_courts = []
        for i, c in enumerate(courts):
            if selected_justice in c:
                j_courts.append(i)
        fig1 = animated_2comp(pca_df, j_courts)
        fig2 = plot_most_similar(big_df, selected_justice)
        return [ {'label': ' '.join(courts[c]), 'value': c} for c in j_courts ], fig1, fig2

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
    app.run_server()
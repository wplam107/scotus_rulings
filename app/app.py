import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

import pandas as pd
import pickle

from app_functions import *

# Dash tutorial CSS
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

with open('./data/mul_df.p', 'rb') as f:
    df = pickle.load(f)


app.layout = html.Div([
    dcc.Graph(id='graph-with-slider'),
])



if __name__ == '__main__':
    app.run_server(debug=True)
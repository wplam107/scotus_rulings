import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

# Plotly graph objects
def animated_2comp(df):
    '''
    Returns Plotly 2 component scatter plot figure
    '''
    fig = px.scatter(df, x='pc1', y='pc2',
                     animation_frame='court',
                     animation_group='justice',
                     text='justice',
                     title='Justices Along 2 Components (PCA)',
                     labels={'pc1': 'PC1', 'pc2': 'PC2'},
                     width=500,
                     height=500,
                     range_x=(-0.2, 1.2),
                     range_y=(-0.2, 1.2),
                    )
    fig.update_traces(textposition='top center')
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 2000
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 3000
    return fig

def sim_heatmap(sim_mat, justices):
    '''
    Returns Plotly heatmap figure
    '''
    fig = go.Figure(data=(go.Heatmap(z=sim_mat, x=justices, y=justices, colorscale='RdBu')))
    fig.update_layout(title='Heatmap of Cosine Similarity Between Justices',
                      height=500,
                      width=500,
                     )
    return fig
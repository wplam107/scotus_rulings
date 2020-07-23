import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

def get_justices(courts):
    justices = []
    for court in courts:
        justices += court
    justices = sorted(list(set(justices)))
    return justices

# Plotly graph objects
def animated_2comp(df, select_courts):
    '''
    Returns Plotly 2 component scatter plot figure
    '''
    if select_courts == [0]:
        temp_df = df.loc[df['court'] == 0]
        fig = px.scatter(temp_df, x='pc1', y='pc2',
                        text='justice',
                        title='Justices Along 2 Components (PCA)',
                        labels={'pc1': 'Principal Component 1', 'pc2': 'Principal Component 2'},
                        width=500,
                        height=500,
                        range_x=(-0.2, 1.2),
                        range_y=(-0.2, 1.2),
                        )
        fig.update_traces(textposition='top center')

    else:
        temp_df = df.loc[df['court'].isin(select_courts)]
        fig = px.scatter(temp_df, x='pc1', y='pc2',
                        animation_frame='court',
                        animation_group='justice',
                        text='justice',
                        title='Justices Along 2 Components (PCA)',
                        labels={'pc1': 'Principal Component 1', 'pc2': 'Principal Component 2'},
                        width=600,
                        height=600,
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
                      height=600,
                      width=600,
                     )
    return fig
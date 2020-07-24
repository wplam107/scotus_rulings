import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

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
    fig = go.Figure(data=(go.Heatmap(z=sim_mat, x=justices, y=justices, colorscale='Inferno')))
    fig.update_layout(title='Heatmap of Cosine Similarity Between Justices',
                      height=600,
                      width=600,
                      template='simple_white',
                     )
    return fig

# Scale data helper function
def scale_data(df):
    jus = list(df.index)
    scaler = MinMaxScaler()
    new_df = pd.DataFrame(scaler.fit_transform(df), index=jus, columns=jus)
    return new_df

# Edge builder helper function
def build_edges(df):
    l = len(df.index) # number of nodes
    edges = []
    for i in range(l):
        for j in range(i+1, l):
            if str(df.iloc[i][j]) != 'nan':
                tup = (df.iloc[i].name, df.iloc[i].index[j], df.iloc[i][j]) # (justice A, justice B, sim)
                edges.append(tup)
    return edges

def build_network(df):
    '''
    Build network function, returns nx.Graph object
    '''
    G = nx.Graph()
    new_df = scale_data(df)
    edges = build_edges(new_df)
    
    # Add edges to graph
    for edge in edges:
        G.add_edge(edge[0], edge[1], weight=edge[2])
    
    # Positions of nodes with Fruchterman-Reingold force-directed algorithm
    pos = nx.spring_layout(G)
    
    return G, pos

# Helper function to treat each edge as a separate trace
def get_edge_traces(G, pos, df):
    edge_traces = []
    for edge in G.edges():
        edge_x = []
        edge_y = []
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        
        # Exponent and multiplier applied to weight of edges
        width = 10 * df.loc[edge[0]][edge[1]]**3
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=width, color='red'),
            hoverinfo='none',
            mode='lines')
        edge_traces.append(edge_trace)
        
    return edge_traces

# Helper function to get node trace
def get_node_trace(G, pos, sim_df, cases_df):
    node_x = []
    node_y = []
    node_name = []
    node_size = []
    cases = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_size.append(len(cases_df.loc[node].dropna()) / 50)
        node_name.append(node)
        cases.append(len(cases_df.loc[node].dropna()))

    node_trace = go.Scatter(
        x=node_x, y=node_y, text=node_name, customdata=cases,
        mode='markers+text', marker=dict(size=node_size),
        hovertemplate='Justice %{text}<br>Cases: %{customdata}')
    return node_trace

def plot_network(sim_df, cases_df):
    '''
    Returns Plotly figure for networkX graph
    '''
    G, pos = build_network(sim_df)
    edge_traces = get_edge_traces(G, pos, sim_df)
    node_trace = get_node_trace(G, pos, sim_df, cases_df)
    
    fig = go.Figure(
        data=edge_traces+[node_trace],
        layout=go.Layout(
            title='SCOTUS Similarity as Network Graph',
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template='plotly_white'
        )
    )
    return fig
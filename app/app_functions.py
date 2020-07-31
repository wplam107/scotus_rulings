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
    if len(select_courts) == 1:
        temp_df = df.loc[df['court'] == select_courts[0]]
        fig = px.scatter(temp_df, x='pc1', y='pc2',
                        text='justice',
                        title='Justices Along 2 Components (PCA)',
                        labels={'pc1': 'Principal Component 1', 'pc2': 'Principal Component 2'},
                        width=600,
                        height=600,
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
                        title='Visualizing Distances Between Justices (PCA)',
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

def sim_heatmap(sim_mat, justices, all_justices=False):
    '''
    Returns Plotly heatmap figure
    '''
    fig = go.Figure(data=(go.Heatmap(
        z=sim_mat, x=justices, y=justices, colorscale='Inferno',
        hovertemplate='Similarity between %{x} and %{y}: %{z}')))

    if all_justices:
        fig.update_layout(title='Similarity Between All Justices',
                        height=600,
                        width=600,
                        template='simple_white',
                        )
    else:
        fig.update_layout(title='Similarity Between Justices in Selected Court',
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
    mid_x = []
    mid_y = []
    mid_text = []

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
        
        cases = df.loc[edge[0]][edge[1]]
        width = cases / 800
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=width, color='red'),
            mode='lines')
        edge_traces.append(edge_trace)
        mid_x.append((x0+x1)/2)
        mid_y.append((y0+y1)/2)
        mid_text.append(f'Cases worked between {edge[0]} and {edge[1]}: {int(cases)}')

    middle_node_trace = go.Scatter(
        x=mid_x, y=mid_y, text=mid_text,
        mode='markers', hoverinfo='text', marker=go.scatter.Marker(opacity=0)
    )
        
    return edge_traces, middle_node_trace

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
    edge_traces, middle_node_trace = get_edge_traces(G, pos, sim_df)
    node_trace = get_node_trace(G, pos, sim_df, cases_df)
    
    fig = go.Figure(
        data=edge_traces+[node_trace]+[middle_node_trace],
        layout=go.Layout(
            title='SCOTUS 1999-2019 Cases Between Justices',
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template='plotly_white',
            height=600,
            width=600,
        )
    )
    return fig

# Helper function for most similar
def most_similar(df, j1, js=None):
    similarity = {}
    other_justices = list(df.index)
    other_justices.remove(j1)
    for j2 in other_justices:
        temp_df = df.loc[[j1, j2]].dropna(axis=1)
        if len(temp_df.columns) != 0:
            X1 = np.array(temp_df.loc[j1])
            X2 = np.array(temp_df.loc[j2])
            similarity[j2] = round(float(cosine_similarity(X1.reshape(1, len(X1)), X2.reshape(1, len(X2)))), 2)
    return similarity

def plot_most_similar(df, j1):
    '''
    Returns Plotly figure to plot most similar
    '''
    dicty = most_similar(df, j1)
    x = list(dicty.keys())
    y = list(dicty.values())
    
    fig = go.Figure(go.Bar(x=y, y=x, orientation='h'))
    fig.update_layout(title=f'Justice {j1} Similarity', yaxis={'categoryorder':'total ascending'})
    return fig

def check_sim(df, j1, js):
    similarity = {}
    other_js = js
    temp_df = df.loc[js].dropna(axis=1)
    for j2 in other_js:
        X1 = np.array(temp_df.loc[j1])
        X2 = np.array(temp_df.loc[j2])
        similarity[j2] = round(float(cosine_similarity(X1.reshape(1, len(X1)), X2.reshape(1, len(X2)))), 2)
    return similarity

def plot_most_similar_2(df, j1, js):
    '''
    Returns Plotly figure to plot most similar
    '''
    assert len(js) != 0
    into_cs = js
    dicty = check_sim(df, j1, into_cs)
    x = list(dicty.keys())
    y = list(dicty.values())
    xn = []
    yn = []
    for i, n in enumerate(x):
        if n != j1:
            xn.append(n)
            yn.append(y[i])
        else:
            continue
    
    fig = go.Figure(go.Bar(x=yn, y=xn, orientation='h'))
    fig.update_layout(title=f'Justice {j1} Similarity for Selected Court', yaxis={'categoryorder':'total ascending'})
    return fig

# Blank figure
def default_pic():
    fig = go.Figure()
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(title='No Justice Chosen', template='simple_white')
    return fig

def default_pic2():
    fig = go.Figure()
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(title='No Court Chosen', template='simple_white')
    return fig
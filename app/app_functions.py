import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import plotly.express as px

# Helper function to get sorted list of justices
def justice_terms(df):
    justices = list(df.index)
    justices_by_cases = []
    for justice in justices:
        first_case = min(df.loc[justice].dropna().index)
        last_case = max(df.loc[justice].dropna().index)
        case_range = (first_case, last_case)
        justices_by_cases.append((justice, case_range))
    justices_by_cases.sort(key=lambda x: (x[1][0], x[1][1]))
    return [ justice[0] for justice in justices_by_cases ]

def get_courts(df):
    '''
    Returns list of different court compositions
    '''
    courts = []
    justices = justice_terms(df)
    i = 0
    j = 9
    while j <= len(df.index):
        court = justices[i:j]
        courts.append(court)
        i += 1
        j += 1
    return courts

def select_justice(justice):
    '''
    Returns list of courts a justice has participated in, assumes global variable all_courts
    '''
    return [ court for court in all_courts if justice in court ]

def get_opinions(df, court):
    '''
    Returns numpy array of a particular court's opinions dropping columns with NaN values
    '''
    return np.array(df.loc[court].dropna(axis=1))

def get_justice_all(df, courts):
    '''
    Returns list of numpy arrays of court opinions from a list of courts
    '''
    return [ get_opinions(df, court) for court in courts ]

def get_sim(court_op):
    '''
    Returns numpy cosine similarity matrix from an array of opinions
    '''
    sim_mat = np.zeros((9,9))
    for i in range(9):
        for j in range(9):
            j_a = court_op[i]
            j_a = j_a.reshape(1, len(j_a))
            j_b = court_op[j]
            j_b = j_b.reshape(1, len(j_b))
            sim_mat[i][j] = np.round(cosine_similarity(j_a, j_b), 4)
    return sim_mat

def get_pca(court_op, court):
    '''
    2-component representation of justices within a particular court (using PCA), returns DataFrame
    '''
    pca = PCA(n_components=2)
    comp = pd.DataFrame(pca.fit_transform(court_op), index=court, columns=['pc1', 'pc2'])

    # Maintain consistent axes based on Ginsburg/Thomas
    if comp.loc['Ginsburg']['pc1'] >=0:
        comp['pc1'] = -comp['pc1']
    if comp.loc['Thomas']['pc2'] >= 0:
        comp['pc2'] = -comp['pc2']
    
    # Scale for consistency
    scaler = MinMaxScaler()
    comp = scaler.fit_transform(comp)
    comp = pd.DataFrame(comp)
    
    comp.reset_index(inplace=True)
    comp.columns = ['justice', 'pc1', 'pc2']
    comp['justice'] = court
    return comp

def all_pca_df(df, courts):
    '''
    Returns a merged 2-component DataFrame
    '''
    pca_dfs = []
    for n in range(len(courts)):
        court_op = get_opinions(df2, courts[n])
        df = get_pca(court_op, courts[n])
        df['court'] = n
        pca_dfs.append(df)

    pca_dfs = pd.concat(pca_dfs).reset_index(drop=True)
    return pca_dfs

def animated_2comp(df):
    fig = px.scatter(df, x='pc1', y='pc2',
                     animation_frame='court',
                     animation_group='justice',
                     text='justice',
                     title='Justices Along 2 Components (PCA)',
                     labels={'pc1': 'PC1', 'pc2': 'PC2'},
                     width=700,
                     height=700,
                    )
    fig.update_traces(textposition='top center')
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 2000
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 3000
    return fig
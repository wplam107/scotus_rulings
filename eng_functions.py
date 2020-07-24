import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Map code string to metric function (binary)
def string_2_ints(s):    
    # Special case 'X' is no vote
    if s is np.nan:
        return np.nan
    if s == 'X':
        return np.nan
    
    # Split and remove non-integers and simplify to either majority or dissent
    nums = []
    for x in set(list(s)):
        try:
            if int(x) > 2: # Any dissent is assigned as dissent against majority opinion
                nums.append(-1)
            elif int(x) <= 2:
                nums.append(1)
            elif str(x) == 'nan':
                continue
        except:
            continue

    return nums[0]

# Map code string to metric function (multi-class)
def string_2_multi(s):    
    # Special case 'X' is no vote
    if s is np.nan:
        return np.nan
    if s == 'X':
        return np.nan
    
    # Split and remove non-integers
    nums = []
    for x in set(list(s)):
        try:
            nums.append(int(x))
        except:
            continue
    nums = np.floor(np.mean(nums))
    if nums == 1:
        value = 2
    elif nums == 2:
        value = 1
    elif nums == 3:
        value = -1
    elif nums == 4:
        value = -2
    else:
        value == np.nan
    # Return floor of mean value of justices' opinions
    return value

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

def all_pca_df(df, court_opinions, courts):
    '''
    Returns a merged 2-component DataFrame
    '''
    pca_dfs = []
    for n in range(len(court_opinions)):
        temp_df = get_pca(court_opinions[n], courts[n])
        temp_df['court'] = n
        pca_dfs.append(temp_df)

    pca_dfs = pd.concat(pca_dfs).reset_index(drop=True)
    return pca_dfs

def get_opinions(df, court):
    '''
    Returns numpy array of a particular court's opinions dropping columns with NaN values
    '''
    return np.array(df.loc[court].dropna(axis=1))

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

def all_sim(df):
    '''
    Returns a similarity DataFrame for all justices (justices who did no serve together are NaN values)
    '''
    jus = list(df.index)
    l = len(jus)
    sim_mat = np.zeros((l,l))

    for i in range(l):
        for j in range(l):   
            anb = np.where(df.loc[jus[i]].notna() & df.loc[jus[j]].notna(), df.columns, np.nan)
            if len([ x for x in anb if str(x) != 'nan' ]) != 0:
                j_a = np.array(df[[ x for x in anb if str(x) != 'nan' ]].loc[jus[i]])
                j_b = np.array(df[[ x for x in anb if str(x) != 'nan' ]].loc[jus[j]])
                sim_mat[i][j] = np.round(cosine_similarity(j_a.reshape(1, len(j_a)), j_b.reshape(1, len(j_a))), 4)
            else:
                sim_mat[i][j] = np.nan
    
    sim_mat = pd.DataFrame(sim_mat, index=jus, columns=jus)
    return sim_mat
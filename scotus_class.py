import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, euclidean_distances
from sklearn.decomposition import PCA

# Create class object encapsulating relevent data and methods
class scotus(object):
    def __init__(self, df):
        self.df = df
        self.justices = list(df.index)
        
        # Case ranges
        self.j_cases = {}
        for j in self.justices:
            cases = self.df.loc[j].dropna().index
            r = (min(cases), max(cases))
            self.j_cases[j] = r
            
        # Make initial court
        self.courts = []
        court = []
        cs = self.j_cases
        js = self.justices
        leaving = []
        starts = []
        first = min(self.df.columns)
        last = max(self.df.columns)
        for j in js:
            if cs[j][0] == first:
                court.append(j)
            else:
                starts.append((j, cs[j][0]))
            if cs[j][1] < last:
                leaving.append((j, cs[j][1]))
        
        # Find retired justices and new justices
        leaving.sort(key=lambda x: x[1])
        starts.sort(key=lambda x: x[1])
        assert len(leaving) == len(starts)
        self.courts.append(court.copy())
    
        # Build new courts
        for i, j in enumerate(leaving):
            court.remove(j[0])
            court.append(starts[i][0])
            self.courts.append(court.copy())
        
    def __len__(self):
        return len(self.df.columns)
    
    def __str__(self):
        return f'Justices: {len(self.justices)}\nCases: {len(self.df.columns)}\nCourts: {len(self.courts)}'
    
    def __repr__(self):
        return f'Justices: {len(self.justices)}\nCases: {len(self.df.columns)}\nCourts: {len(self.courts)}'
        
    def justice_term(self, justice, verbose=False):
        '''
        Returns dictionary of cosine similarities for a justice
        '''
        
        assert justice in self.justices, f'Not a justice between 1999-2019.\nChoose one of {self.justices}'
        js = list(self.justices)
        js.remove(justice)
        
        # Create dictionary {justice: similarity}
        no_sim = []
        sim = {}
        for j in js:
            anb = np.where(self.df.loc[justice].notna() & self.df.loc[j].notna(), self.df.columns, np.nan)
            if len([ x for x in anb if str(x) != 'nan' ]) != 0:
                j_a = np.array(self.df[[ x for x in anb if str(x) != 'nan' ]].loc[justice])
                j_b = np.array(self.df[[ x for x in anb if str(x) != 'nan' ]].loc[j])
                sim[j] = np.round(cosine_similarity(j_a.reshape(1, len(j_a)), j_b.reshape(1, len(j_a))), 4)
            else:
                no_sim.append(j)
        
        # Return and/or print results
        sim = { k: v for k, v in sorted(sim.items(), key=lambda item: item[1], reverse=True) }
        ordered = list(sim.keys())

        if verbose:
            print(f'Justice {justice} Cosine Similarity: (descending similarity)')
            print(20*'-')
            for j in ordered:
                print(f'{j}:', sim[j])
                if j == ordered[-1]:
                    print(20*'-')
            if len(no_sim) > 0:
                print('No rulings with:')
                print(20*'-')
                for j in no_sim:
                    print(f'{j}')
        
        return sim
        
                
    def sim_matrix(self, metric='cosine'):
        '''
        Return similarity matrix (Numpy array)
        '''

        jus = list(self.justices)
        l = len(jus)
        sim_mat = np.zeros((l,l))
        
        for i in range(l):
            for j in range(l):   
                anb = np.where(self.df.loc[jus[i]].notna() & self.df.loc[jus[j]].notna(), self.df.columns, np.nan)
                if len([ x for x in anb if str(x) != 'nan' ]) != 0:
                    j_a = np.array(self.df[[ x for x in anb if str(x) != 'nan' ]].loc[jus[i]])
                    j_b = np.array(self.df[[ x for x in anb if str(x) != 'nan' ]].loc[jus[j]])
                    sim_mat[i][j] = np.round(cosine_similarity(j_a.reshape(1, len(j_a)), j_b.reshape(1, len(j_a))), 4)
                else:
                    sim_mat[i][j] = np.nan
            
        del jus
        del l
        return sim_mat
    
    def two_dim_court(self, court_num):
        '''
        2-component representation of justices within a particular court (using PCA), returns DataFrame
        '''
        
        assert court_num in range(0, len(self.courts)), print(f'Choose int from 0-{len(self.courts)-1}')
        
        temp_df = pd.DataFrame([ self.df.loc[j] for j in self.courts[court_num] ]).dropna(axis=1)
        X = temp_df.values
        pca = PCA(n_components=2)
        comp = pd.DataFrame(pca.fit_transform(X), columns=['x', 'y'])
        comp['justice'] = temp_df.index
        
        return comp

    def justice_courts(self, justice):
        '''
        Set a justice
        '''
        courts = []
        for i, court in enumerate(self.courts):
            if justice in court:
                courts.append(i)
        self.j_courts = courts
        self.current_j = justice

    def _two_dims(self, courts):
        assert self.j_courts, 'Choose justice with justice_courts'
        comps = [ two_dim_court(court) for court in self.j_courts ]
        self.comps = comps
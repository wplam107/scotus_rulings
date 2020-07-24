# Script for converting scraped data to similarity matrices and PCA matrix
import numpy as np
import pandas as pd
import pickle
from eng_functions import *

if __name__ == '__main__':
    df = pd.read_csv('scotus_rulings.csv', index_col=0)

    # Apply data cleaning and feature engineering functions
    mul_df = pd.DataFrame(np.vectorize(string_2_multi)(df), index=df.index)

    # Get list of different courts
    all_courts = get_courts(mul_df)
    all_court_opinions = [ get_opinions(mul_df, court) for court in all_courts ]

    # Get PCA data for all courts
    pca_df = all_pca_df(mul_df, all_court_opinions, all_courts)

    # Get similarity matrices for all courts
    sim_matrices = [ get_sim(court) for court in all_court_opinions ]

    # Get single similarity matrix for all justices
    sim_df, total_df = all_sim(mul_df)

    # Pickle data
    with open('./app/data/courts.p', 'wb') as f:
        pickle.dump(all_courts, f)

    with open('./app/data/pca_df.p', 'wb') as f:
        pickle.dump(pca_df, f)

    with open('./app/data/sim_mats.p', 'wb') as f:
        pickle.dump(sim_matrices, f)

    with open('./app/data/sim_df.p', 'wb') as f:
        pickle.dump(sim_df, f)

    with open('./app/data/multi_df.p', 'wb') as f:
        pickle.dump(mul_df, f)

    with open('./app/data/total_df.p', 'wb') as f:
        pickle.dump(total_df, f)

    print('Files created')


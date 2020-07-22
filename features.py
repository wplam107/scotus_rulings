import numpy as np
import pandas as pd
import pickle
from eng_functions import string_2_ints, string_2_multi

if __name__ == '__main__':
    df = pd.read_csv('scotus_rulings.csv', index_col=0)

    # Apply data cleaning and feature engineering functions
    adj_df = pd.DataFrame(np.vectorize(string_2_ints)(df), index=df.index)
    mul_df = pd.DataFrame(np.vectorize(string_2_multi)(df), index=df.index)

    # Write DataFrames to .csv files
    with open('./app/data/adj_df.p', 'wb') as f:
        pickle.dump(adj_df, f)
    with open('./app/data/mul_df.p', 'wb') as f:
        pickle.dump(mul_df, f)
    print('Files created')
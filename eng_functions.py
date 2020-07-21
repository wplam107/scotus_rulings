import numpy as np
import pandas as pd

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
                nums.append(2)
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
            
    # Return floor of mean value of justices' opinions
    return np.floor(np.mean(nums))
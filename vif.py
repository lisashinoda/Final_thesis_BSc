#%%
import numpy as np
import pandas as pd
import time
from statsmodels.stats.outliers_influence import variance_inflation_factor    
from joblib import Parallel, delayed
import os

# Defining the function that you will run later
def calculate_vif_(X, thresh):
    variables = [X.columns[i] for i in range(X.shape[1])]
    dropped=True
    while dropped:
        dropped=False
        print(len(variables))
        vif = Parallel(n_jobs=-1,verbose=5)(delayed(variance_inflation_factor)(X[variables].values, ix) for ix in range(len(variables)))

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print(time.ctime() + ' dropping \'' + X[variables].columns[maxloc] + '\' at index: ' + str(maxloc))
            variables.pop(maxloc)
            dropped=True

    print('Remaining variables:')
    print([variables])
    return X[[i for i in variables]]

#%%
current_path=os.path.dirname(os.path.abspath("__file__"))
path=os.path.join(current_path,'excel')
number=['number1']
for i in number:
    feature=pd.read_csv(os.path.join(path,'merge_'+str(i)+'_train.csv'),usecols=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37])
    X2 = calculate_vif_(feature,3) # Actually running the function
    print(X2)

# %%

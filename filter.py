#%%
from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectKBest, f_regression
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
import os
current_path=os.path.dirname(os.path.abspath("__file__"))
path=os.path.join(current_path,'excel')
#%%
for i in ['number1','number2','number3']:
#for i in ['all']:
    x_train=pd.read_csv(os.path.join(path,'merge_'+str(i)+'_train.csv'),usecols=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37])
    y_train=pd.read_csv(os.path.join(path,'merge_'+str(i)+'_train.csv'),usecols=[1])
    # 18つの特徴量を選択
    selector = SelectKBest(score_func=f_regression, k=18) 
    selector.fit(x_train, y_train)
    vector_names = list(x_train.columns[selector.get_support(indices=True)])
    print(vector_names)
    sum_train=pd.read_csv(os.path.join(path,'merge_'+str(i)+'_train.csv'),usecols=['date','disease'])
    for j in vector_names:
        merge1=pd.read_csv(os.path.join(path,'merge_'+str(i)+'_train.csv'),usecols=['date',j])
        sum_train=pd.merge(sum_train,merge1,on='date')
    sum_train.to_csv(os.path.join(path,'select_'+str(i)+'_train.csv'), encoding='utf_8',index=False)
    sum_test=pd.read_csv(os.path.join(path,'merge_'+str(i)+'_test.csv'),usecols=['date','disease'])
    for j in vector_names:
        merge2=pd.read_csv(os.path.join(path,'merge_'+str(i)+'_test.csv'),usecols=['date',j])
        sum_test=pd.merge(sum_test,merge2,on='date')
    sum_test.to_csv(os.path.join(path,'select_'+str(i)+'_test.csv'), encoding='utf_8',index=False)
    


# %%

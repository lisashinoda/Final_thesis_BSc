#%%
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import os
current_path=os.path.dirname(os.path.abspath("__file__"))
path=os.path.join(current_path,'excel')
#%%
def corr_column(df, threshold):
    df_corr = df.corr()
    df_corr = abs(df_corr)
    columns = df_corr.columns
    # 対角線の値を0にする
    for j in range(0, len(columns)):
        df_corr.iloc[j, j] = 0
    while True:
        columns = df_corr.columns
        max_corr = 0.0
        query_column = None
        target_column = None
        df_max_column_value = df_corr.max()
        max_corr = df_max_column_value.max()
        query_column = df_max_column_value.idxmax()
        target_column = df_corr[query_column].idxmax()
        if max_corr < threshold:
            # しきい値を超えるものがなかったため終了
            break
        else:
            # しきい値を超えるものがあった場合
            delete_column = None
            saved_column = None
            # その他との相関の絶対値が大きい方を除去
            if sum(df_corr[query_column]) <= sum(df_corr[target_column]):
                delete_column = target_column
                saved_column = query_column
            else:
                delete_column = query_column
                saved_column = target_column
            # 除去すべき特徴を相関行列から消す（行、列）
            df_corr.drop([delete_column], axis=0, inplace=True)
            df_corr.drop([delete_column], axis=1, inplace=True)
    return df_corr.columns
#%%
data=['number1','number2','number3','all']
for i in data:
    x_train=pd.read_csv(os.path.join(path,'merge_'+str(i)+'_train.csv'),usecols=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40])
    x_test=pd.read_csv(os.path.join(path,'merge_'+str(i)+'_test.csv'),usecols=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40])
    train_corr=pd.read_csv(os.path.join(path,'merge_'+str(i)+'_train.csv'))
    test_corr=pd.read_csv(os.path.join(path,'merge_'+str(i)+'_test.csv'))
    train_corr=train_corr[corr_column(x_train,0.90)]
    train_corr.to_csv(os.path.join(path,''+str(i)+'_corr_train_2.csv'), encoding='utf_8',index=False)
    test_corr=test_corr[corr_column(x_train,0.90)]
    test_corr.to_csv(os.path.join(path,''+str(i)+'__corr_test_2.csv'), encoding='utf_8',index=False)
#%%

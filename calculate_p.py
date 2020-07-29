#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import statsmodels.formula.api as smf
import os
current_path=os.path.dirname(os.path.abspath("__file__"))
path=os.path.join(current_path,'excel')
#%%
def calculate(x_train,y_train,x_test,y_test):
    from sklearn.preprocessing import MinMaxScaler
    mmsc = MinMaxScaler()
    # 訓練用のデータを正規化する
    x_train_norm = mmsc.fit_transform(x_train)
    # 訓練用データを基準にテストデータも正規化
    x_test_norm = mmsc.transform(x_test)
    import statsmodels.api as sm
    model = sm.Logit(y_train.astype(float), x_train_norm.astype(float))
    result = model.fit()
    return result.pvalues
#%%--p値を計算
for i in ['number1','number2','number3','all']:
    x_train=pd.read_csv(os.path.join(path,''+str(i)+'_corr_train_2.csv'))
    y_train=pd.read_csv(os.path.join(path,'merge_'+str(i)+'_train_2.csv'),usecols=[1])
    x_test=pd.read_csv(os.path.join(path,''+str(i)+'__corr_test_2.csv'))
    y_test=pd.read_csv(os.path.join(path,'merge_'+str(i)+'_test_2.csv'),usecols=[1])
    print(calculate(x_train,y_train,x_test,y_test))
#%%
for i in ['number1','number2','number3','all']:
    path_train=os.path.join(path,''+str(i)+'_corr_train.csv')
    path_test=os.path.join(path,''+str(i)+'__corr_test.csv')
    path_train2=os.path.join(path,'merge_'+str(i)+'_train.csv')
    path_test2=os.path.join(path,'merge_'+str(i)+'_test.csv')
    train2=pd.read_csv(path_train2,usecols=[0,1])
    test2=pd.read_csv(path_test2,usecols=[0,1])
    if i=='number1':
        train=pd.read_csv(path_train,usecols=[0,2,4,5,6,7,9,10,12,13,14,15,18,19,20,21,22,23])
        test=pd.read_csv(path_test,usecols=[0,2,4,5,6,7,9,10,12,13,14,15,18,19,20,21,22,23])
    if i=='number2':
        train=pd.read_csv(path_train,usecols=[1,2,5,7,10,13,15,18,21,22,24])
        test=pd.read_csv(path_test,usecols=[1,2,5,7,10,13,15,18,21,22,24])
    if i=='number3':
        train=pd.read_csv(path_train,usecols=[1,2,5,6,7,8,9,11,12,13,14,15,16,22,23,24])
        test=pd.read_csv(path_test,usecols=[1,2,5,6,7,8,9,11,12,13,14,15,16,22,23,24])
    if i=='all':
        train=pd.read_csv(path_train,usecols=[0,1,4,6,7,8,9,14,15,18,19,22,23,24])
        test=pd.read_csv(path_test,usecols=[0,1,4,6,7,8,9,14,15,18,19,22,23,24])
    df_train=pd.concat([train2,train],axis=1)
    df_test=pd.concat([test2,test],axis=1)
    df_train.to_csv(os.path.join(path,'crr_p_'+str(i)+'_train.csv'), encoding='utf_8',index=False)
    df_test.to_csv(os.path.join(path,'crr_p_'+str(i)+'_test.csv'), encoding='utf_8',index=False)
    

# #%%
# for i in ['number1','number2','number3','all']:
#     path2=os.path.join(path,'merge_'+str(i)+'_train.csv')
#     path3=os.path.join(path,'merge_'+str(i)+'_test.csv')
#     if i=='number1':
#         train=pd.read_csv(path2,usecols=[0,1,2,5,10,11,12,13,22,23,24,25,29,33,34,35,36])
#         test=pd.read_csv(path3,usecols=[0,1,2,5,10,11,12,13,22,23,24,25,29,33,34,35,36])
#     if i=='number2':
#         train=pd.read_csv(path2,usecols=[0,1,2,6,11,13,17,23,25,35,37])
#         test=pd.read_csv(path3,usecols=[0,1,2,6,11,13,17,23,25,35,37])
#     if i=='number3':
#         train=pd.read_csv(path2,usecols=[0,1,4,6,11,12,13,22,24,33,34,35,37])
#         test=pd.read_csv(path3,usecols=[0,1,4,6,11,12,13,22,24,33,34,35,37])
#     if i=='all':
#         train=pd.read_csv(path2,usecols=[0,1,2,5,10,13,17,25,26,29,32,35,37])
#         test=pd.read_csv(path3,usecols=[0,1,2,5,10,13,17,25,26,29,32,35,37])
#     train.to_csv(os.path.join(path,'crr_p_'+str(i)+'_train.csv'), encoding='utf_8',index=False)
#     test.to_csv(os.path.join(path,'crr_p_'+str(i)+'_test.csv'), encoding='utf_8',index=False)

# %%

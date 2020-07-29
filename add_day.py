#%%
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
current_path=os.path.dirname(os.path.abspath("__file__"))
path=os.path.join(current_path)
save_path=os.path.join(path,'excel')
df_days=pd.read_csv(os.path.join(path,'days.csv'))
df_days['date']=pd.to_datetime(df_days['date'])
number=['number1','number2','number3']
for i in number:
    df_train=pd.read_csv(os.path.join(save_path,'merge_'+str(i)+'_train.csv'))
    df_test=pd.read_csv(os.path.join(save_path,'merge_'+str(i)+'_test.csv'))
    df_train['date']=pd.to_datetime(df_train['date'])
    df_test['date']=pd.to_datetime(df_test['date'])
    df_train2=pd.merge(df_train,df_days,on='date')
    df_test2=pd.merge(df_test,df_days,on='date')
    df_train2.dropna()
    df_test2.dropna()
    print(df_train2)
    df_train2.to_csv(os.path.join(save_path,'merge_'+str(i)+'_train_2.csv'), encoding='utf_8',index=False)
    df_test2.to_csv(os.path.join(save_path,'merge_'+str(i)+'_test_2.csv'), encoding='utf_8',index=False)

# %%
##データセットall作成
csv_cont_all_train= []
csv_cont_all_test= []
number=['number1','number2','number3']
for i in number:
    train=os.path.join(save_path,'merge_'+str(i)+'_train_2.csv')
    test=os.path.join(save_path,'merge_'+str(i)+'_test_2.csv')
    if os.path.exists(path):
        csv_cont_all_train.append(pd.read_csv(train))
        csv_cont_all_test.append(pd.read_csv(test))
merge_train = pd.concat(csv_cont_all_train)
merge_test = pd.concat(csv_cont_all_test)
merge_train['date'] = pd.to_datetime(merge_train['date'])
merge_test['date'] = pd.to_datetime(merge_test['date'])
merge_train.drop_duplicates(subset='date')
merge_test.drop_duplicates(subset='date')
merge_train.to_csv(os.path.join(save_path,'merge_all_train_2.csv'), encoding='utf_8',index=False)
merge_test.to_csv(os.path.join(save_path,'merge_all_test_2.csv'), encoding='utf_8',index=False)


# %%

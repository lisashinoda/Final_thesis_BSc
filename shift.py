#%%
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
current_path=os.path.dirname(os.path.abspath("__file__"))
path=os.path.join(current_path)
#%%--datasetごとにshift
save_path=os.path.join(path,'excel')
if not os.path.isdir(save_path):
    os.makedirs(save_path)
number=['number1','number2','number3']
for i in number:
    path1=os.path.join(path,'merge_'+str(i)+'.csv')
    df = pd.read_csv(path1, encoding="shift-jis",usecols=[0,2,3,4,5,6,7,8,9,10,11,12,13])
    disease = pd.read_csv(path1, encoding="shift-jis", usecols=[0,1])
    df_train, df_test = train_test_split(df, test_size=0.3, shuffle=False)
    disease['date']=pd.to_datetime(disease['date'])
    df_train['date']=pd.to_datetime(df_train['date'])
    df_test['date']=pd.to_datetime(df_test['date'])
    df_train=df_train.set_index('date')
    df_test=df_test.set_index('date')
    df_train2=pd.merge(disease,df_train.shift(1),on='date')
    df_train3=pd.merge(df_train2,df_train.shift(2),on='date')
    df_train4=pd.merge(df_train3,df_train.shift(3),on='date')
    df_train4=df_train4.dropna()
    df_train4.rename(columns={'T20_mean': 'T20_mean_z', 'ST_mean': 'ST_mean_z', 'T60_mean':'T60_mean_z','H_mean':'H_mean_z', 'T20_min':'T20_min_z', 'ST_min':'ST_min_z', 'T60_min':'T60_min_z', 'H_min':'H_min_z', 'T20_max':'T20_max_z','ST_max':'ST_max_z', 'T60_max':'T60_max_z', 'H_max':'H_max_z'},inplace=True)
    df_train4=df_train4.set_index('date')
    df_train4.to_csv(os.path.join(save_path,'merge_'+str(i)+'_train.csv'),encoding='utf_8',index=True)
    df_test2=pd.merge(disease,df_test.shift(1),on='date')
    df_test3=pd.merge(df_test2,df_test.shift(2),on='date')
    df_test4=pd.merge(df_test3,df_test.shift(3),on='date')
    df_test4=df_test4.dropna()
    df_test4.rename(columns={'T20_mean': 'T20_mean_z', 'ST_mean': 'ST_mean_z', 'T60_mean':'T60_mean_z','H_mean':'H_mean_z', 'T20_min':'T20_min_z', 'ST_min':'ST_min_z', 'T60_min':'T60_min_z', 'H_min':'H_min_z', 'T20_max':'T20_max_z','ST_max':'ST_max_z', 'T60_max':'T60_max_z', 'H_max':'H_max_z'},inplace=True)
    df_test4=df_test4.set_index('date')
    df_test4.to_csv(os.path.join(save_path,'merge_'+str(i)+'_test.csv'),encoding='utf_8',index=True)
#%% データセットall作成
csv_cont_all_train= []
csv_cont_all_test= []
number=['number1','number2','number3']
for i in number:
    train=os.path.join(save_path,'merge_'+str(i)+'_train.csv')
    test=os.path.join(save_path,'merge_'+str(i)+'_test.csv')
    if os.path.exists(path):
        csv_cont_all_train.append(pd.read_csv(train))
        csv_cont_all_test.append(pd.read_csv(test))
merge_train = pd.concat(csv_cont_all_train)
merge_test = pd.concat(csv_cont_all_test)
merge_train['date'] = pd.to_datetime(merge_train['date'])
merge_test['date'] = pd.to_datetime(merge_test['date'])
merge_train.to_csv(os.path.join(save_path,'merge_all_train.csv'), encoding='utf_8',index=False)
merge_test.to_csv(os.path.join(save_path,'merge_all_test.csv'), encoding='utf_8',index=False)
# %%--テストデータ内の病気データをカウント
dataset=['number1','number2','number3','all']
for i in dataset:
    df=pd.read_csv(os.path.join(save_path,'merge_'+str(i)+'_train.csv'))
    vc = df['disease'].value_counts()
    print(vc)

# %%

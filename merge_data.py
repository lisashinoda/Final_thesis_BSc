#%%
import os
import pandas as pd
import numpy as np
import glob
from datetime import datetime

#%%--pathを取得
current_path=os.path.dirname(os.path.abspath("__file__"))
path=os.path.join(current_path,'data')
path_before=os.path.join(current_path)
file_name=os.listdir(path)
# %%--平均値、最大値、最小値を出す
number=['number1','number2','number3']
#%%
for i in file_name:
    for j in number:
        path2=os.path.join(path,str(i),str(j),'foo.csv.csv')
        if os.path.exists(path2):
            data=pd.read_csv(path2)
            dTime=pd.read_csv(path2, usecols=[0])
            dT20=pd.read_csv(path2, usecols=[1])
            dST=pd.read_csv(path2, usecols=[2])
            dT60=pd.read_csv(path2, usecols=[3])
            dH=pd.read_csv(path2, usecols=[4])
            data['Date/Time'] = pd.to_datetime(data['Date/Time'])
            ##-- 日にちごとにgroupby
            data['date'] = data['Date/Time'].dt.date
            mean=data.groupby(['date']).mean()
            min=data.groupby(['date']).min()
            max=data.groupby(['date']).max()
            min.drop("Date/Time", axis='columns',inplace=True)
            max.drop("Date/Time", axis='columns',inplace=True)
            connect2=pd.merge(mean,min,on='date')
            connect3=pd.merge(connect2,max,on='date')
            connect3.rename(columns={'T20_x': 'T20_mean', 'ST_x': 'ST_mean', 'T60_x':'T60_mean','H_x':'H_mean', 'T20_y':'T20_min', 'ST_y':'ST_min', 'T60_y':'T60_min', 'H_y':'H_min', 'T20':'T20_max','ST':'ST_max', 'T60':'T60_max', 'H':'H_max'},inplace=True)
            connect3.to_csv(os.path.join(path,str(i),str(j),'environment_data.csv'))
            ##connect3=connect3.set_index('date')
            df_connect3=pd.merge(connect3.shift(1),connect3.shift(2),on='date')
            df_connect4=pd.merge(df_connect3,connect3.shift(3),on='date')
            df_connect4=df_connect4.dropna()
            df_connect4.rename(columns={'T20_mean': 'T20_mean_z', 'ST_mean': 'ST_mean_z', 'T60_mean':'T60_mean_z','H_mean':'H_mean_z', 'T20_min':'T20_min_z', 'ST_min':'ST_min_z', 'T60_min':'T60_min_z', 'H_min':'H_min_z', 'T20_max':'T20_max_z','ST_max':'ST_max_z', 'T60_max':'T60_max_z', 'H_max':'H_max_z'},inplace=True)
            ##df_connect4.drop_duplicates(subset='date')
            ##df_connect4=df_connect4.set_index('date')
            df_connect4.to_csv(os.path.join(path,str(i),str(j),'shift_'+str(i)+'.csv'),encoding='utf_8',index=True)
#%%--データセットごとにまとめる
csv_cont= []
disease_red=[]
train=['20161115data','20170113data','20170510data','20170731data','20171012data','20171218data','20180322data','20180528data','20180810data','20181026data','20180108data']
test=['20190306data','20190624data','20191017data','20100806data']
for i in number:
    for j in train:
        path3=os.path.join(path,str(j),str(i),'shift_'+str(j)+'.csv')
        if os.path.exists(path3):
            csv_cont.append(pd.read_csv(path3))
        path5=os.path.join(path,str(j),str(i),'disease_red.csv')
        if os.path.exists(path5):
            disease_red.append(pd.read_csv(path5))
    # リスト形式にまとめたCSVファイルを結合
    merge_content = pd.concat(csv_cont)
    csv_cont.clear()
    merge_disease = pd.concat(disease_red)
    if i== 'number1':
        merge_disease=merge_disease.drop(columns='Date/Time')
    merge_disease['date'] = pd.to_datetime(merge_disease['date'])
    disease_red.clear()
    merge_content['date'] = pd.to_datetime(merge_content['date'])
    merge=pd.merge(merge_disease,merge_content,on='date')
    merge=merge.dropna()
    merge=merge.set_index('date')
    merge.to_csv(os.path.join(path_before,'excel','merge_'+str(i)+'_train.csv'),encoding='utf_8',index=True)
#%%
for i in number:
    for j in test:
        path3=os.path.join(path,str(j),str(i),'shift_'+str(j)+'.csv')
        if os.path.exists(path3):
            csv_cont.append(pd.read_csv(path3))
        path5=os.path.join(path,str(j),str(i),'disease_red.csv')
        if os.path.exists(path5):
            disease_red.append(pd.read_csv(path5))
    # リスト形式にまとめたCSVファイルを結合
    merge_content = pd.concat(csv_cont)
    csv_cont.clear()
    merge_disease = pd.concat(disease_red)
    merge_disease['date'] = pd.to_datetime(merge_disease['date'])
    disease_red.clear()
    merge_content['date'] = pd.to_datetime(merge_content['date'])
    merge=pd.merge(merge_disease,merge_content,on='date')
    merge=merge.dropna()
    merge=merge.set_index('date')
    merge.to_csv(os.path.join(path_before,'excel','merge_'+str(i)+'_test.csv'),encoding='utf_8',index=True)

#%%
## データセットall作成
csv_cont_all= []
for i in number:
    path4=os.path.join(path_before,'excel','merge_'+str(i)+'_train.csv')
    if os.path.exists(path4):
        csv_cont_all.append(pd.read_csv(path4))
merge_content_all = pd.concat(csv_cont_all)
merge_content_all['date'] = pd.to_datetime(merge_content_all['date'])
merge_content_all.to_csv(os.path.join(path_before,'excel','merge_all_train.csv'), encoding='utf_8',index=False)

csv_cont_all.clear()
for i in number:
    path4=os.path.join(path_before,'excel','merge_'+str(i)+'_test.csv')
    if os.path.exists(path4):
        csv_cont_all.append(pd.read_csv(path4))
merge_content_all = pd.concat(csv_cont_all)
merge_content_all['date'] = pd.to_datetime(merge_content_all['date'])
merge_content_all.to_csv(os.path.join(path_before,'excel','merge_all_test.csv'), encoding='utf_8',index=False)


# %%--テストデータ内の病気データをカウント
dataset=['number1','number2','number3','all']
for i in dataset:
    df=pd.read_csv(os.path.join(path_before,'excel','merge_'+str(i)+'_train.csv'))
    vc = df['disease'].value_counts()
    print(vc)
for i in dataset:
    df=pd.read_csv(os.path.join(path_before,'excel','merge_'+str(i)+'_test.csv'))
    vc = df['disease'].value_counts()
    print(vc)



# %%

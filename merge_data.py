#%%
import os
import pandas as pd
import numpy as np
import glob
from datetime import datetime
import os
#%%--pathを取得
current_path=os.path.dirname(os.path.abspath("__file__"))
path=os.path.join(current_path,'data')
path_before=os.path.join(current_path)
file_name=os.listdir(path)
# %%--平均値、最大値、最小値を出す
number=['number1','number2','number3']
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
            print(connect3.columns)
            connect3.to_csv(os.path.join(path,str(i),str(j),'environment_data.csv'))
#%%--データセットごとにまとめる
csv_cont= []
disease_yellow=[]
for i in number:
    for j in file_name:
        path3=os.path.join(path,str(j),str(i),'environment_data.csv')
        if os.path.exists(path3):
            csv_cont.append(pd.read_csv(path3))
        path5=os.path.join(path,str(j),str(i),'disease_yellow.csv')
        if os.path.exists(path5):
            disease_yellow.append(pd.read_csv(path5))
    # リスト形式にまとめたCSVファイルを結合
    merge_content = pd.concat(csv_cont)
    #merge_content.info()
    csv_cont.clear()
    merge_disease = pd.concat(disease_yellow)
    if i== 'number1':
        merge_disease=merge_disease.drop(columns='Date/Time')
    #if i== 'number3':
      #  merge_disease=merge_disease.drop(columns='Date/Time')
    merge_disease['date'] = pd.to_datetime(merge_disease['date'])
    disease_yellow.clear()
    merge_content['date'] = pd.to_datetime(merge_content['date'])
    merge=pd.merge(merge_disease,merge_content,on='date')
    merge=merge.dropna()
    merge.info()
    merge.to_csv('merge_'+str(i)+'.csv')
## データセットall作成
#%%
csv_cont_all= []
for i in number:
    path4=os.path.join(path_before,'merge_'+str(i)+'.csv')
    if os.path.exists(path4):
        csv_cont_all.append(pd.read_csv(path4))
merge_content_all = pd.concat(csv_cont_all)
merge_content_all['date'] = pd.to_datetime(merge_content_all['date'])
merge_content_all.info()
merge_content_all.to_csv('merge_all.csv', encoding='utf_8',index=False)


# %%

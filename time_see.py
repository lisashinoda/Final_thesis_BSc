#%%
import os
import pandas as pd
import numpy as np
import glob
from datetime import datetime
from datetime import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#%%--pathを取得
current_path=os.path.dirname(os.path.abspath("__file__"))
path=os.path.join(current_path,'data')
path_before=os.path.join(current_path)
file_name=os.listdir(path)
#%%-　環境データ
today_data=[]
number=['number1','number2','number3']
for i in number:
    for j in file_name:
        path2=os.path.join(path,str(j),str(i),'environment_data.csv')
        if os.path.exists(path2):
            today_data.append(pd.read_csv(path2))
    today_data2 = pd.concat(today_data)
    today_data2.to_csv(os.path.join(path_before,'excel','today_'+str(i)+'.csv'),encoding='utf_8',index=True)

# %%
df = pd.read_csv(os.path.join(path_before,'excel','today_'+str(i)+'.csv'))
df.head()
save_path=os.path.join(current_path,'graph_time')
# %%
for i in number:
    df = pd.read_csv(os.path.join(path_before,'excel','today_'+str(i)+'.csv'))
    #pandas.Seriesに変換
    series = pd.Series(df['T20_x'], dtype='float') 
    series.index = pd.to_datetime(df['date']) 
    plt.figure()
    plt.ylabel("T20_mean")
    series.plot()
    image_path=os.path.join(save_path,''+str(i)+'_meantemp20_time.png')
    plt.savefig(image_path)

    series = pd.Series(df['T20_y'], dtype='float') 
    series.index = pd.to_datetime(df['date']) 
    plt.figure()
    plt.ylabel("T20_min")
    series.plot()
    image_path=os.path.join(save_path,''+str(i)+'_mintemp20_time.png')
    plt.savefig(image_path)

    series = pd.Series(df['T20'], dtype='float') 
    series.index = pd.to_datetime(df['date']) 
    plt.figure()
    plt.ylabel("T20_max")
    series.plot()
    image_path=os.path.join(save_path,''+str(i)+'_maxtemp20_time.png')
    plt.savefig(image_path)

    series = pd.Series(df['T60_x'], dtype='float') 
    series.index = pd.to_datetime(df['date']) 
    plt.figure()
    plt.ylabel("T60_mean")
    series.plot()
    image_path=os.path.join(save_path,''+str(i)+'_meantemp60_time.png')
    plt.savefig(image_path)

    series = pd.Series(df['T60_y'], dtype='float') 
    series.index = pd.to_datetime(df['date']) 
    plt.figure()
    plt.ylabel("T60_min")
    series.plot()
    image_path=os.path.join(save_path,''+str(i)+'_mintemp60_time.png')
    plt.savefig(image_path)

    series = pd.Series(df['T60'], dtype='float') 
    series.index = pd.to_datetime(df['date']) 
    plt.figure()
    plt.ylabel("T60_max")
    series.plot()
    image_path=os.path.join(save_path,''+str(i)+'_maxtemp60_time.png')
    plt.savefig(image_path)

    series = pd.Series(df['ST_x'], dtype='float') 
    series.index = pd.to_datetime(df['date']) 
    plt.figure()
    plt.ylabel("ST_mean")
    series.plot()
    image_path=os.path.join(save_path,''+str(i)+'_meanST_time.png')
    plt.savefig(image_path)

    series = pd.Series(df['ST_y'], dtype='float') 
    series.index = pd.to_datetime(df['date']) 
    plt.figure()
    plt.ylabel("ST_min")
    series.plot()
    image_path=os.path.join(save_path,''+str(i)+'_minST_time.png')
    plt.savefig(image_path)

    series = pd.Series(df['ST'], dtype='float') 
    series.index = pd.to_datetime(df['date']) 
    plt.figure()
    plt.ylabel("ST_max")
    series.plot()
    image_path=os.path.join(save_path,''+str(i)+'_maxST_time.png')
    plt.savefig(image_path)

    series = pd.Series(df['H_x'], dtype='float') 
    series.index = pd.to_datetime(df['date']) 
    plt.figure()
    plt.ylabel("H_mean")
    series.plot()
    image_path=os.path.join(save_path,''+str(i)+'_meanH_time.png')
    plt.savefig(image_path)

    series = pd.Series(df['H_y'], dtype='float') 
    series.index = pd.to_datetime(df['date']) 
    plt.figure()
    plt.ylabel("H_min")
    series.plot()
    image_path=os.path.join(save_path,''+str(i)+'_minH_time.png')
    plt.savefig(image_path)

    series = pd.Series(df['H'], dtype='float') 
    series.index = pd.to_datetime(df['date']) 
    plt.figure()
    plt.ylabel("H_max")
    series.plot()
    image_path=os.path.join(save_path,''+str(i)+'_maxH_time.png')
    plt.savefig(image_path)
#%%--病気データ
for i in number:
    df2 = pd.read_csv(os.path.join(path_before,'excel','merge_'+str(i)+'.csv'),usecols=[0,1])
    df2['date']=pd.to_datetime(df2['date'])
    df2.info()
    df2=df2.set_index('date')
    df2 = df2['disease'].resample('M').sum() 
    df2.head()
    # 描画
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.ylabel("Disease")
    ax.plot(df2)
    image_path=os.path.join(save_path,''+str(i)+'_disease_time.png')
    plt.savefig(image_path)
# %%--ペアプロット
df3 = pd.read_csv(os.path.join(path_before,'excel','merge_number2.csv'),usecols=[0,1])

# %%
df3.info()

# %%
df4=pd.read_csv(os.path.join(path_before,'excel','today_number2.csv'),usecols=[1,2,3,5])

# %%
df4.head()

# %%
df5=pd.merge(df3,df4,on='date')

# %%
df5.head()

# %%
import seaborn as sns
figure=sns.pairplot(df5, hue="disease", size=3.0)
image_path=os.path.join(save_path,'number2_pair.png')
figure.savefig(image_path)
# %%

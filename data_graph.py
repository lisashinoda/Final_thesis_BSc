#%%
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
#%%
number=["x","y","z"]
day=['1','2','3']
def data_see(df):
    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('Set1')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(df.query('disease=="1"')['days'], bins=50, alpha=0.6)
    ax.hist(df.query('disease=="0"')['days'], bins=50, alpha=0.6)
    ax.set_xlabel('days')
    ax.set_ylabel('Count')
    if not j == 'merge_all':
        ax.set(xlim=(0,120), ylim=(0,10))
    if  j == 'merge_all':
        ax.set(xlim=(0,120))
    plt.legend(['Disease', 'Not disease'])
        #plt.savefig(''+str(j)+'_meantemp20(t-'+str(k)+').png')
    image_path=os.path.join(save_path,''+str(j)+'_days.png')
    plt.savefig(image_path)
#%%
    for i,k in zip (number,day):
        sns.set()
        sns.set_style('whitegrid')
        sns.set_palette('Set1')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(df.query('disease=="1"')['T20_mean_'+str(i)+''], bins=50, alpha=0.6)
        ax.hist(df.query('disease=="0"')['T20_mean_'+str(i)+''], bins=50, alpha=0.6)
        ax.set_xlabel('Mean Temperature 20cm(°c)')
        ax.set_ylabel('Count')
        if not j == 'merge_all':
            ax.set(xlim=(10,35), ylim=(0,28))
        if  j == 'merge_all':
            ax.set(xlim=(10,35))
        plt.legend(['Disease', 'Not disease'])
        #plt.savefig(''+str(j)+'_meantemp20(t-'+str(k)+').png')
        image_path=os.path.join(save_path,''+str(j)+'_meantemp20(t-'+str(k)+').png')
        plt.savefig(image_path)

    for i,k in zip (number,day):
        sns.set()
        sns.set_style('whitegrid')
        sns.set_palette('Set1')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(df.query('disease=="1"')['T20_min_'+str(i)+''], bins=50, alpha=0.6)
        ax.hist(df.query('disease=="0"')['T20_min_'+str(i)+''], bins=50, alpha=0.6)
        ax.set_xlabel('Minimum Temperature 20cm(°c)')
        ax.set_ylabel('Count')
        plt.legend(['Disease', 'Not disease'])
        if not j=='merge_all':
            ax.set(xlim=(3,29), ylim=(0,52))
        if j=='merge_all':
            ax.set(xlim=(3,29))
        image_path=os.path.join(save_path,''+str(j)+'_mintemp20(t-'+str(k)+').png')
        plt.savefig(image_path)

    for i,k in zip (number,day):
        sns.set()
        sns.set_style('whitegrid')
        sns.set_palette('Set1')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(df.query('disease=="1"')['T20_max_'+str(i)+''], bins=50, alpha=0.6)
        ax.hist(df.query('disease=="0"')['T20_max_'+str(i)+''], bins=50, alpha=0.6)
        ax.set_xlabel('Maximum Temperature 20cm(°c)')
        ax.set_ylabel('Count')
        plt.legend(['Disease', 'Not disease'])
        if not j=='merge_all':
            ax.set(xlim=(13,45), ylim=(0,23))
        if j=='merge_all':
            ax.set(xlim=(13,45))
        image_path=os.path.join(save_path,''+str(j)+'_maxtemp20(t-'+str(k)+').png')
        plt.savefig(image_path)

    for i,k in zip (number,day):
        sns.set()
        sns.set_style('whitegrid')
        sns.set_palette('Set1')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(df.query('disease=="1"')['T60_mean_'+str(i)+''], bins=50, alpha=0.6)
        ax.hist(df.query('disease=="0"')['T60_mean_'+str(i)+''], bins=50, alpha=0.6)
        ax.set_xlabel('Mean Temperature 60cm(°c)')
        ax.set_ylabel('Count')
        plt.legend(['Disease', 'Not disease'])
        if not j=='merge_all':
            ax.set(xlim=(3,32), ylim=(0,33))
        if j=='merge_all':
            ax.set(xlim=(3,32))
        #plt.savefig(''+str(j)+'_meantemp60(t-'+str(k)+').png')
        image_path=os.path.join(save_path,''+str(j)+'_meantemp60(t-'+str(k)+').png')
        plt.savefig(image_path)

    for i,k in zip (number,day):
        sns.set()
        sns.set_style('whitegrid')
        sns.set_palette('Set1')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(df.query('disease=="1"')['T60_min_'+str(i)+''], bins=50, alpha=0.6)
        ax.hist(df.query('disease=="0"')['T60_min_'+str(i)+''], bins=50, alpha=0.6)
        ax.set_xlabel('Minimum Temperature 60cm(°c)')
        ax.set_ylabel('Count')
        plt.legend(['Disease', 'Not disease'])
        if not j=='merge_all':
            ax.set(xlim=(2,29), ylim=(0,33))
        if j=='merge_all':
            ax.set(xlim=(2,29))
        image_path=os.path.join(save_path,''+str(j)+'_mintemp60(t-'+str(k)+').png')
        plt.savefig(image_path)

    for i,k in zip (number,day):
        sns.set()
        sns.set_style('whitegrid')
        sns.set_palette('Set1')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(df.query('disease=="1"')['T60_max_'+str(i)+''], bins=50, alpha=0.6)
        ax.hist(df.query('disease=="0"')['T60_max_'+str(i)+''], bins=50, alpha=0.6)
        ax.set_xlabel('Maximum Temperature 60cm(°c)')
        ax.set_ylabel('Count')
        plt.legend(['Disease', 'Not disease'])
        if not j=='merge_all':
            ax.set(xlim=(9,53), ylim=(0,37))
        if not j=='merge_all':
            ax.set(xlim=(9,53))
        image_path=os.path.join(save_path,''+str(j)+'_maxtemp60(t-'+str(k)+').png')
        plt.savefig(image_path)

    for i,k in zip (number,day):
        sns.set()
        sns.set_style('whitegrid')
        sns.set_palette('Set1')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(df.query('disease=="1"')['ST_mean_'+str(i)+''], bins=50, alpha=0.6)
        ax.hist(df.query('disease=="0"')['ST_mean_'+str(i)+''], bins=50, alpha=0.6)
        ax.set_xlabel('Mean Soil Temperature(°c)')
        ax.set_ylabel('Count')
        plt.legend(['Disease', 'Not disease'])
        if not j=='merge_all':
            ax.set(xlim=(10,33), ylim=(0,23))
        if j=='merge_all':
            ax.set(xlim=(10,33))
        image_path=os.path.join(save_path,''+str(j)+'_meanST(t-'+str(k)+').png')
        plt.savefig(image_path)

    for i,k in zip (number,day):
        sns.set()
        sns.set_style('whitegrid')
        sns.set_palette('Set1')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(df.query('disease=="1"')['ST_min_'+str(i)+''], bins=50, alpha=0.6)
        ax.hist(df.query('disease=="0"')['ST_min_'+str(i)+''], bins=50, alpha=0.6)
        ax.set_xlabel('Minimum Soil Temperature(°c)')
        ax.set_ylabel('Count')
        plt.legend(['Disease', 'Not disease'])
        if not j=='merge_all':
            ax.set(xlim=(6,29), ylim=(0,39))
        if j=='merge_all':
            ax.set(xlim=(6,29))
        image_path=os.path.join(save_path,''+str(j)+'_minST(t-'+str(k)+').png')
        plt.savefig(image_path)

    for i,k in zip (number,day):
        sns.set()
        sns.set_style('whitegrid')
        sns.set_palette('Set1')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(df.query('disease=="1"')['ST_max_'+str(i)+''], bins=50, alpha=0.6)
        ax.hist(df.query('disease=="0"')['ST_max_'+str(i)+''], bins=50, alpha=0.6)
        ax.set_xlabel('Maximum Soil Temperature(°c)')
        ax.set_ylabel('Count')
        plt.legend(['Disease', 'Not disease'])
        if not j=='merge_all':
            ax.set(xlim=(12,38), ylim=(0,24))
        if j=='merge_all':
            ax.set(xlim=(12,38))
        image_path=os.path.join(save_path,''+str(j)+'_maxST(t-'+str(k)+').png')
        plt.savefig(image_path)

    for i,k in zip (number,day):
        sns.set()
        sns.set_style('whitegrid')
        sns.set_palette('Set1')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(df.query('disease=="1"')['H_mean_'+str(i)+''], bins=50, alpha=0.6)
        ax.hist(df.query('disease=="0"')['H_mean_'+str(i)+''], bins=50, alpha=0.6)
        ax.set_xlabel('Mean Humid(%)')
        ax.set_ylabel('Count')
        plt.legend(['Disease', 'Not disease'])
        if not j=='merge_all':
            ax.set(xlim=(9,103), ylim=(0,30))
        if j=='merge_all':
            ax.set(xlim=(9,103))
        image_path=os.path.join(save_path,''+str(j)+'_meanH(t-'+str(k)+').png')
        plt.savefig(image_path)
    
    for i,k in zip (number,day):
        sns.set()
        sns.set_style('whitegrid')
        sns.set_palette('Set1')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(df.query('disease=="1"')['H_min_'+str(i)+''], bins=50, alpha=0.6)
        ax.hist(df.query('disease=="0"')['H_min_'+str(i)+''], bins=50, alpha=0.6)
        ax.set_xlabel('Minimum Humid(%)')
        ax.set_ylabel('Count')
        plt.legend(['Disease', 'Not disease'])
        if not j=='merge_all':
            ax.set(xlim=(15,103), ylim=(0,24))
        if j=='merge_all':
            ax.set(xlim=(15,103))
        image_path=os.path.join(save_path,''+str(j)+'_minH(t-'+str(k)+').png')
        plt.savefig(image_path)

    for i,k in zip (number,day):
        sns.set()
        sns.set_style('whitegrid')
        sns.set_palette('Set1')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(df.query('disease=="1"')['H_max_'+str(i)+''], bins=50, alpha=0.6)
        ax.hist(df.query('disease=="0"')['H_max_'+str(i)+''], bins=50, alpha=0.6)
        ax.set_xlabel('Maximum Humid(%)')
        ax.set_ylabel('Count')
        plt.legend(['Disease', 'Not disease'])
        if not j=='merge_all':
            ax.set(xlim=(65,103), ylim=(0,150))
        if j=='merge_all':
            ax.set(xlim=(65,103))
        image_path=os.path.join(save_path,''+str(j)+'_maxH(t-'+str(k)+').png')
        plt.savefig(image_path)
# %%
import os
current_path=os.path.dirname(os.path.abspath("__file__"))
path=os.path.join(current_path,'')
save_path=os.path.join(current_path,'graph')
data=['merge_number1','merge_number2','merge_number3','merge_all']
if not os.path.isdir(save_path):
    os.makedirs(save_path)
for j in data:
    data=os.path.join(current_path,'excel',''+str(j)+'_train_2.csv')
    df=pd.read_csv(data)
    df.rename(columns={'T20_mean': 'T20_mean_z', 'ST_mean': 'ST_mean_z', 'T60_mean':'T60_mean_z','H_mean':'H_mean_z', 'T20_min':'T20_min_z', 'ST_min':'ST_min_z', 'T60_min':'T60_min_z', 'H_min':'H_min_z', 'T20_max':'T20_max_z','ST_max':'ST_max_z', 'T60_max':'T60_max_z', 'H_max':'H_max_z'},inplace=True)
    data_see(df)



# %%

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
path_excel=os.path.join(current_path,'excel')
number=['number1','number2','number3','all']
dataset=['train','test']
for z in dataset:
    for i in number:
        path2=os.path.join(path_excel,'merge_'+str(i)+'_'+str(z)+'.csv')
        #data=pd.read_csv(path2,usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13])
        data=pd.read_csv(path2)
        alpha=['x','y','z','a','b']
        #alpha=['x']
        for j in alpha:
            hpres=6.1078*10**(7.5*data['T60_mean_'+str(j)+'']/(data['T60_mean_'+str(j)+'']+273.3))
            houwa=(217*hpres)/(data['T60_mean_'+str(j)+'']+273.15)
            housa=(100-data['H_mean_'+str(j)+''])*houwa/100
            housa=housa.to_frame()
            housa.columns=['housa_'+str(j)+'']
            data=pd.concat([data,housa],axis=1)
            #data=pd.merge(data,housa,left_index=True,right_index=True)
        data.to_csv(os.path.join(path_excel,'merge_'+str(i)+'_'+str(z)+'.csv'), encoding='utf_8',index=False)
# %%

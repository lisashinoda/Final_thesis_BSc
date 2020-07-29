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
number=['number1','number2','number3']
for i in number:
    path2=os.path.join(path_before,'merge_'+str(i)+'.csv')
    data=pd.read_csv(path2,usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13])
    hpres=6.1078*10**(7.5*data["T60_mean"]/(data["T60_mean"]+273.3))
    houwa=(217*hpres)/(data["T60_mean"]+273.15)
    housa=(100-data["H_mean"])*houwa/100
    housa=housa.to_frame()
    housa.columns=['housa']
    merge_content_all=pd.concat([data,housa],axis=1)
    merge_content_all=pd.merge(data,housa,left_index=True,right_index=True)
    merge_content_all.to_csv('merge_'+str(i)+'.csv', encoding='utf_8',index=False)



# %%

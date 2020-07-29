#%%
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def randomforest(path_train,path_test):
    from sklearn.preprocessing import MinMaxScaler
    mmsc = MinMaxScaler()
    # 訓練用のデータを正規化する
    x_train = pd.DataFrame(mmsc.fit_transform(pd.read_csv(path_train).iloc[:,2:]))
    # 訓練用データを基準にテストデータも正規化
    x_test = pd.DataFrame(mmsc.transform(pd.read_csv(path_test).iloc[:,2:]))
    y_train=pd.read_csv(path_train,usecols=['disease'])
    y_test=pd.read_csv(path_test,usecols=['disease'])
    search_params = {
        'n_estimators'      : [5, 10, 20, 30, 50, 100, 300],
        'max_features'      : [3, 5, 9],
        'random_state'      : [2525],
        'n_jobs'            : [1],
        'min_samples_split' : [3, 5, 10, 15, 20, 25, 30, 40, 50, 100],
        'max_depth'         : [3, 5, 10, 15, 20, 25, 30, 40, 50, 100]
    }
    gs = GridSearchCV(RFC(),         # 対象の機械学習モデル
                    search_params,   # 探索パラメタ辞書
                    cv=5,            # クロスバリデーションの分割数
                    verbose=True,    # ログ表示
                    n_jobs=-1)       # 並列処理
    gs.fit(x_train, y_train)
    best_estimator=gs.best_estimator_
    from sklearn.metrics import classification_report
    score=gs.score(x_test, y_test) 
    y_true, y_pred = y_test, gs.predict(x_test)
    report=classification_report(y_true, y_pred)
    return best_estimator,score,report,confusion_matrix(y_true, y_pred)
# %%
import os
current_path=os.path.dirname(os.path.abspath("__file__"))
path=os.path.join(current_path,'')
number=['number1','number2','number3','all']


#%%
##特徴量選択前
for i in number:
    path_train=os.path.join(current_path,'excel','merge_'+str(i)+'_train.csv')
    path_test=os.path.join(current_path,'excel','merge_'+str(i)+'_test.csv')
    print(randomforest(path_train,path_test))
#%%
##方法A
for i in number:
    path_train=os.path.join(current_path,'excel','select_'+str(i)+'_train.csv')
    path_test=os.path.join(current_path,'excel','select_'+str(i)+'_test.csv')
    print(randomforest(path_train,path_test))
#%%
##方法B
import os
for i in number:
    path_train=os.path.join(current_path,'excel','crr_p_'+str(i)+'_train.csv')
    path_test=os.path.join(current_path,'excel','crr_p_'+str(i)+'_test.csv')
    print(randomforest(path_train,path_test))


# %%

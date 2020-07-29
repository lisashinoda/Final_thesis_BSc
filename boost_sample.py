#%%
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#%%
def main(path_train,path_test):
    from sklearn.preprocessing import MinMaxScaler
    mmsc = MinMaxScaler()
    # 訓練用のデータを正規化する
    x_train = pd.DataFrame(mmsc.fit_transform(pd.read_csv(path_train).iloc[:,2:]))
    # 訓練用データを基準にテストデータも正規化
    x_test = pd.DataFrame(mmsc.transform(pd.read_csv(path_test).iloc[:,2:]))
    y_train=pd.read_csv(path_train,usecols=['disease'])
    y_test=pd.read_csv(path_test,usecols=['disease'])
    # XGBoost が扱うデータセットの形式に直す
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    # 学習用のパラメータ
    xgb_params = {
        # 二値分類問題
        'objective': 'binary:logistic',
        # 評価指標
        'eval_metric': 'logloss',
    }
    # モデルを学習する
    bst = xgb.train(xgb_params,
                    dtrain,
                    num_boost_round=100,  # 学習ラウンド数は適当
                    )
    # 検証用データが各クラスに分類される確率を計算する
    y_pred_proba = bst.predict(dtest)
    # しきい値 0.5 で 0, 1 に丸める
    y_pred = np.where(y_pred_proba > 0.5, 1, 0)
    # 精度 (Accuracy) を検証する
    acc = accuracy_score(y_test, y_pred)
    print('Accuracy:', acc)
    from sklearn.metrics import classification_report 
    from sklearn.metrics import confusion_matrix
    report=classification_report(y_test, y_pred)
    return report,confusion_matrix(y_test, y_pred)
# %%
import os
current_path=os.path.dirname(os.path.abspath("__file__"))
path=os.path.join(current_path,'')
number=['number1','number2','number3','all']
#%%
##特徴量選択前
for i in number:
    path_train=os.path.join(current_path,'excel','merge_'+str(i)+'_train_2.csv')
    path_test=os.path.join(current_path,'excel','merge_'+str(i)+'_test_2.csv')
    print(main(path_train,path_test))
#%%
##方法A
for i in number:
    path_train=os.path.join(current_path,'excel','select_'+str(i)+'_train.csv')
    path_test=os.path.join(current_path,'excel','select_'+str(i)+'_test.csv')
    print(main(path_train,path_test))
#%%
##方法B
import os
for i in number:
    path_train=os.path.join(current_path,'excel','crr_p_'+str(i)+'_train.csv')
    path_test=os.path.join(current_path,'excel','crr_p_'+str(i)+'_test.csv')
    print(main(path_train,path_test))






# %%

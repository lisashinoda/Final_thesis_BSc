#%%
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# %%
def SVM(path_train,path_test):
    from sklearn.preprocessing import MinMaxScaler
    mmsc = MinMaxScaler()
    # 訓練用のデータを正規化する
    x_train = pd.DataFrame(mmsc.fit_transform(pd.read_csv(path_train).iloc[:,2:]))
    # 訓練用データを基準にテストデータも正規化
    x_test = pd.DataFrame(mmsc.transform(pd.read_csv(path_test).iloc[:,2:]))
    y_train=pd.read_csv(path_train,usecols=['disease'])
    y_test=pd.read_csv(path_test,usecols=['disease'])
    search_params = [
        {
            "kernel"          : ["rbf","linear","sigmoid"],
            "C"               : [10**i for i in range(-10,10)],
            "random_state"    : [2525],
        }
    ]
    gs = GridSearchCV(SVC(), 
                    search_params, 
                    cv = 3,
                    verbose=True,
                    n_jobs=-1)
    gs.fit(x_train, y_train.values.ravel())
    y_true, y_pred = y_test, gs.predict(x_test)
    return gs.best_estimator_,gs.best_estimator_.score(x_test,y_test), classification_report(y_true, y_pred),confusion_matrix(y_true, y_pred)
# %%
import os
current_path=os.path.dirname(os.path.abspath("__file__"))
path=os.path.join(current_path,'')
number=['all']
#%%
##特徴量選択前
for i in number:
    path_train=os.path.join(current_path,'excel','merge_'+str(i)+'_train.csv')
    path_test=os.path.join(current_path,'excel','merge_'+str(i)+'_test.csv')
    print(SVM(path_train,path_test))

##方法A
for i in number:
    path_train=os.path.join(current_path,'excel','select_'+str(i)+'_train.csv')
    path_test=os.path.join(current_path,'excel','select_'+str(i)+'_test.csv')
    print(SVM(path_train,path_test))
##方法B
import os
for i in number:
    path_train=os.path.join(current_path,'excel','crr_p_'+str(i)+'_train.csv')
    path_test=os.path.join(current_path,'excel','crr_p_'+str(i)+'_test.csv')
    print(SVM(path_train,path_test))


# %%

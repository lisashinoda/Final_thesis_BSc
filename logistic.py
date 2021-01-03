#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns   
from sklearn import metrics
# %%
##Logistic回帰モデル
def Logistic(path_train,path_test):
    from sklearn.preprocessing import MinMaxScaler
    mmsc = MinMaxScaler()
    # 訓練用のデータを正規化する
    x_train=pd.read_csv(path_train,index_col=0)
    x_test=pd.read_csv(path_test,index_col=0)
    columns=x_train.iloc[:,2:].columns.values
    x_train = pd.DataFrame(mmsc.fit_transform(x_train.iloc[:,2:]),columns=columns)
    # 訓練用データを基準にテストデータも正規化
    x_test =pd.DataFrame(mmsc.transform(x_test.iloc[:,2:]),columns=columns)
    y_train=pd.read_csv(path_train,usecols=['disease'])
    y_test=pd.read_csv(path_test,usecols=['disease'])
    clf = LR() 
    clf.fit(x_train, y_train) 
    y_true, y_pred = y_test, clf.predict(x_test)
    # coefficient=pd.DataFrame(clf.coef_,columns=x_train.keys())
    # plt.figure(figsize=(20, 11)) 
    # sns.set_palette("hls",24)
    # g=sns.barplot(data=coefficient)
    # plt.setp(g.get_xticklabels(), rotation=90)
    # img_path=os.path.join(save_path,''+str(k)+'crr_p_'+str(i)+'_logistic.png')
    # plt.savefig(img_path)

    y_pred_prob=clf.predict_proba(x_test)[:,1]
    # save the probability
    pd.DataFrame(y_pred_prob).to_csv(os.path.join(current_path,'prob','csv','prob_'+str(i)+'_logistic_'+str(k)+'.csv'),encoding='utf_8',index=True)
    plt.figure(figsize=(20, 11)) 
    fpr, tpr,thresholds = metrics.roc_curve(y_test, y_pred_prob)
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc)
    plt.plot(np.linspace(1, 0, len(fpr)), np.linspace(1, 0, len(fpr)), label='Random ROC curve (area = %.2f)'%0.5, linestyle = '--', color = 'gray')
    plt.legend()
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    img_path2=os.path.join(save_path2,''+str(i)+'_logistic_'+str(k)+'.png')
    plt.savefig(img_path2)
    return clf.score(x_test,y_test),classification_report(y_true, y_pred),confusion_matrix(y_true, y_pred)
# %%
import os
current_path=os.path.dirname(os.path.abspath("__file__"))
path=os.path.join(current_path,'')
save_path=os.path.join(current_path,'logistic_graph')
save_path2=os.path.join(current_path,'prob','details')
number=['number1','number2','number3','all']
if not os.path.isdir(save_path):
    os.makedirs(save_path)
#%%
##特徴量選択前
for i in number:
    k='normal'
    path_train=os.path.join(current_path,'excel','merge_'+str(i)+'_train.csv')
    path_test=os.path.join(current_path,'excel','merge_'+str(i)+'_test.csv')
    print(Logistic(path_train,path_test))
#%%
##方法A
for i in number:
    k='selectA'
    path_train=os.path.join(current_path,'excel','select_'+str(i)+'_train.csv')
    path_test=os.path.join(current_path,'excel','select_'+str(i)+'_test.csv')
    print(Logistic(path_train,path_test))
#%%
##方法B
import os
for i in number:
    k='selectB'
    path_train=os.path.join(current_path,'excel','crr_p_'+str(i)+'_train.csv')
    path_test=os.path.join(current_path,'excel','crr_p_'+str(i)+'_test.csv')
    print(Logistic(path_train,path_test))
#%%


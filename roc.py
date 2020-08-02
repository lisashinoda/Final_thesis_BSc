# %%
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
import os
    
# %%
def ROC (model):
    prob1=[]
    prob2=[]
    prob3=[]
    prob4=[]
    method=['normal','selectA','selectB']
    number=['number1','number2','number3','all']

    current_path=os.path.dirname(os.path.abspath("__file__"))
    path=os.path.join(current_path,'prob')

    for i  in number:
        path_test=os.path.join(current_path,'excel','merge_'+str(i)+'_test.csv')
        for k in method:
            y_pred_prob=pd.read_csv(os.path.join(path,'csv','prob_'+str(i)+'_'+str(model)+'_'+str(k)+'.csv'),usecols=[1])
            if i=='number1':
                    prob1.append(y_pred_prob)
                    prob=prob1
            if i=='number2':
                    prob2.append(y_pred_prob)
                    prob=prob2
            if i=='number3':
                    prob3.append(y_pred_prob)
                    prob=prob3
            if i=='all':
                    prob4.append(y_pred_prob)
                    prob=prob4
        y_test=pd.read_csv(path_test,usecols=['disease'])
        plt.figure(figsize=(20, 11)) 
        fpr1, tpr1,thresholds1 = metrics.roc_curve(y_test, prob[0][:])
        auc = metrics.auc(fpr1, tpr1)
        plt.plot(fpr1, tpr1, label='ROC curve1  (area = %.2f)'%auc)
        fpr2, tpr2,thresholds2 = metrics.roc_curve(y_test, prob[1][:])
        auc = metrics.auc(fpr2, tpr2)
        plt.plot(fpr2, tpr2, label='ROC curve2 methodA (area = %.2f)'%auc)
        fpr3, tpr3,thresholds3 = metrics.roc_curve(y_test, prob[2][:])
        auc = metrics.auc(fpr3, tpr3)
        plt.plot(fpr3, tpr3, label='ROC curve3 methodB (area = %.2f)'%auc)
        plt.plot(np.linspace(1, 0, len(fpr1)), np.linspace(1, 0, len(fpr1)), label='Random ROC curve (area = %.2f)'%0.5, linestyle = '--', color = 'gray')
        plt.legend()
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True)
        img_path2=os.path.join(path,''+str(i)+'_'+str(model)+'.png')
        plt.savefig(img_path2)
        prob.clear()
#%%
ROC(logistic)
#%%
ROC(svm)
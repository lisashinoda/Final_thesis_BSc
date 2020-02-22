#%%
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.layers import LSTM
from sklearn.metrics import confusion_matrix
#%%
def activate(train_x,train_y,test_x,test_y):
    for i in number:
        # design network
        model = Sequential()
        model.add(LSTM(4, input_shape=(train_x.shape[1], train_x.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam',metrics=['accuracy'])
        # fit network
        history = model.fit(train_x, train_y, epochs=50, batch_size=72, validation_data=(test_x, test_y), verbose=2, shuffle=False)
        # plot history
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()
        y_pred=model.predict_classes(test_x)
        y_true=test_y
        labels = np.unique(y_true)
        a =  confusion_matrix(y_true, y_pred, labels=labels)
        a2=pd.DataFrame(a, index=labels, columns=labels)
        return a2
#%%
import os
current_path=os.path.dirname(os.path.abspath("__file__"))
path=os.path.join(current_path,'')
number=['number1','number2','number3','all']
#%%-特長量選択前
for i in number:
    path_train=os.path.join(current_path,'excel','merge_'+str(i)+'_train.csv')
    path_test=os.path.join(current_path,'excel','merge_'+str(i)+'_test.csv')
    train_x=pd.read_csv(path_train,usecols=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37])
    train_y=pd.read_csv(path_train,usecols=[1])
    test_x=pd.read_csv(path_test,usecols=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37])
    test_y=pd.read_csv(path_test,usecols=[1])
    train_x=train_x.ix[:,[0,3,6,9,12,15,18,21,24,27,30,33,1,4,7,10,13,16,19,22,25,28,31,34,2,5,8,11,14,17,20,23,26,29,32,35]]
    test_x=test_x.ix[:,[0,3,6,9,12,15,18,21,24,27,30,33,1,4,7,10,13,16,19,22,25,28,31,34,2,5,8,11,14,17,20,23,26,29,32,35]]
    train_x= train_x.values.reshape((train_x.shape[0], 3, 12))
    test_x = test_x.values.reshape((test_x.shape[0], 3, 12))
    print(activate(train_x,train_y,test_x,test_y))
#%%--方法A
for i in number:
    path_train=os.path.join(current_path,'excel','select_'+str(i)+'_train.csv')
    path_test=os.path.join(current_path,'excel','select_'+str(i)+'_test.csv')
    train_y=pd.read_csv(path_train,usecols=[1])
    test_y=pd.read_csv(path_test,usecols=[1])
    train_x=pd.read_csv(path_train,usecols=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
    test_x=pd.read_csv(path_test,usecols=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
    train_x=train_x.ix[:,[0,3,6,9,12,15,1,4,7,10,13,16,2,5,8,11,14,17]]
    test_x=test_x.ix[:,[0,3,6,9,12,15,1,4,7,10,13,16,2,5,8,11,14,17]]
    train_x= train_x.values.reshape((train_x.shape[0], 3, 6))
    test_x = test_x.values.reshape((test_x.shape[0], 3, 6))
    print(activate(train_x,train_y,test_x,test_y))
# %%--方法B
for i in number:
    path_train=os.path.join(current_path,'excel','merge_'+str(i)+'_train.csv')
    path_test=os.path.join(current_path,'excel','merge_'+str(i)+'_test.csv')
    
    if i=='number1':
        train_y=pd.read_csv(path_train,usecols=[1])
        test_y=pd.read_csv(path_test,usecols=[1])
        train_x=pd.read_csv(path_train,usecols=['T20_mean_x','T20_mean_y','T20_mean','T20_max_x','T20_max_y','T20_max','T60_mean_x','T60_mean_y','T60_mean','T60_max_x','T60_max_y','T60_max','ST_mean_x','ST_mean_y','ST_mean','ST_max_x','ST_max_y','ST_max'])
        test_x=pd.read_csv(path_test,usecols=['T20_mean_x','T20_mean_y','T20_mean','T20_max_x','T20_max_y','T20_max','T60_mean_x','T60_mean_y','T60_mean','T60_max_x','T60_max_y','T60_max','ST_mean_x','ST_mean_y','ST_mean','ST_max_x','ST_max_y','ST_max'])
        train_x=train_x.ix[:,[0,3,6,9,12,15,1,4,7,10,13,16,2,5,8,11,14,17]]
        test_x=test_x.ix[:,[0,3,6,9,12,15,1,4,7,10,13,16,2,5,8,11,14,17]]
        train_x= train_x.values.reshape((train_x.shape[0], 3, 6))
        test_x = test_x.values.reshape((test_x.shape[0], 3, 6))
        print(activate(train_x,train_y,test_x,test_y))

    if i=='number2':
        train_y=pd.read_csv(path_train,usecols=[1])
        test_y=pd.read_csv(path_test,usecols=[1])
        train_x=pd.read_csv(path_train,usecols=['T20_mean_x','T20_mean_y','T20_mean','T20_max_x','T20_max_y','T20_max','T60_mean_x','T60_mean_y','T60_mean','T60_max_x','T60_max_y','T60_max','ST_max_x','ST_max_y','ST_max','ST_min_x','ST_min_y','ST_min'])
        test_x=pd.read_csv(path_test,usecols=['T20_mean_x','T20_mean_y','T20_mean','T20_max_x','T20_max_y','T20_max','T60_mean_x','T60_mean_y','T60_mean','T60_max_x','T60_max_y','T60_max','ST_max_x','ST_max_y','ST_max','ST_min_x','ST_min_y','ST_min'])
        train_x=train_x.ix[:,[0,3,6,9,12,1,4,7,10,13,2,5,8,11,14]]
        test_x=test_x.ix[:,[0,3,6,9,12,1,4,7,10,13,2,5,8,11,14]]
        train_x= train_x.values.reshape((train_x.shape[0], 3, 5))
        test_x = test_x.values.reshape((test_x.shape[0], 3, 5))
        print(activate(train_x,train_y,test_x,test_y))

    if i=='number3':
        train_y=pd.read_csv(path_train,usecols=[1])
        test_y=pd.read_csv(path_test,usecols=[1])
        train_x=pd.read_csv(path_train,usecols=['T20_mean_x','T20_mean_y','T20_mean','T20_max_x','T20_max_y','T20_max','T60_mean_x','T60_mean_y','T60_mean','T60_max_x','T60_max_y','T60_max','ST_mean_x','ST_mean_y','ST_mean','ST_max_x','ST_max_y','ST_max','ST_min_x','ST_min_y','ST_min'])
        test_x=pd.read_csv(path_test,usecols=['T20_mean_x','T20_mean_y','T20_mean','T20_max_x','T20_max_y','T20_max','T60_mean_x','T60_mean_y','T60_mean','T60_max_x','T60_max_y','T60_max','ST_mean_x','ST_mean_y','ST_mean','ST_max_x','ST_max_y','ST_max','ST_max','ST_min_x','ST_min_y','ST_min'])
        train_x=train_x.ix[:,[0,3,6,9,12,15,18,1,4,7,10,13,16,19,2,5,8,11,14,17,20]]
        test_x=test_x.ix[:,[0,3,6,9,12,15,18,1,4,7,10,13,16,19,2,5,8,11,14,17,20]]
        train_x= train_x.values.reshape((train_x.shape[0], 3, 7))
        test_x = test_x.values.reshape((test_x.shape[0], 3, 7))
        print(activate(train_x,train_y,test_x,test_y))

    if i=='all':
        train_y=pd.read_csv(path_train,usecols=[1])
        test_y=pd.read_csv(path_test,usecols=[1])
        train_x=pd.read_csv(path_train,usecols=['T20_mean_x','T20_mean_y','T20_mean','T20_max_x','T20_max_y','T20_max','T60_mean_x','T60_mean_y','T60_mean','T60_max_x','T60_max_y','T60_max','ST_mean_x','ST_mean_y','ST_mean','ST_max_x','ST_max_y','ST_max'])
        test_x=pd.read_csv(path_test,usecols=['T20_mean_x','T20_mean_y','T20_mean','T20_max_x','T20_max_y','T20_max','T60_mean_x','T60_mean_y','T60_mean','T60_max_x','T60_max_y','T60_max','ST_mean_x','ST_mean_y','ST_mean','ST_max_x','ST_max_y','ST_max'])
        train_x=train_x.ix[:,[0,3,6,9,12,15,1,4,7,10,13,16,2,5,8,11,14,17]]
        test_x=test_x.ix[:,[0,3,6,9,12,15,1,4,7,10,13,16,2,5,8,11,14,17]]
        train_x= train_x.values.reshape((train_x.shape[0], 3, 6))
        test_x = test_x.values.reshape((test_x.shape[0], 3, 6))
        print(activate(train_x,train_y,test_x,test_y))

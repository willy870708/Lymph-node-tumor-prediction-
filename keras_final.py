    
#函式庫
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py
import subprocess as sp
from scipy import io
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten,MaxPooling2D
from sklearn import preprocessing
from keras.callbacks import ReduceLROnPlateau

def show_result():
    #模型預測參數
    y_predict = np.argmax(model.predict(testing_data),axis=1) #測試資料
    y_map_predict = np.argmax(model.predict(mapping_data),axis=1) #進行colormapping
    y_true = []
    y_pred = []
    y_map = []
    
    #將正確職與預測值放入陣列
    for i in range(len(y_predict)): 
        y_pred.append(int(y_predict[i]))
        y_true.append(int(testing_target[i]))
        
    #將預測的colormapping圖片放入陣列
    for i in range(len(y_map_predict)):
        y_map.append(int(y_map_predict[i]))
    conf = confusion_matrix(y_true,y_pred) #混淆矩陣，用於計算 sensitivity & specificity
    sensitivity = conf[0][0]/(conf[0][0]+conf[1][0])
    specificity = conf[1][1]/(conf[1][1]+conf[0][1])
    
    #模型結果 & 找出ROI
    loss, accuracy = model.evaluate(testing_data,testing_target) #測試資料準確率
    print('\nLoss:','%.2f'%(loss))
    print('Accuracy:',accuracy*100,'%')
    print('Sensitivity : '+str(sensitivity*100)+'%')
    print('Specificity : '+str(specificity*100)+'%')
    print('TP:'+str(conf[0][0])+'\n'+'TN:'+str(conf[1][1])+'\n'+'FP:'+str(conf[0][1])+'\n'+'FN:'+str(conf[1][0]))
    
    #將驗證結果圖片印出
    show_train_history(train_history,'acc','val_acc')
    
    #將colormapping圖片存成.mat格式
    colormapping_data = np.insert(mapping_data_flattern,400,y_map,axis=1) #加入元素到陣列(原始陣列,對應的內部陣列數量,欲加入的陣列,axis=插入的維度)
    io.savemat('finded_data.mat', {'colormapping_data':colormapping_data})  

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel('train')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='center right' )
    plt.show()

#資料放入方式
mat_data=io.loadmat('C:/Users/dl/Desktop/code/data/system_mix.mat') #訓練資料 or 驗證資料
mat_data_test=h5py.File('C:/Users/dl/Desktop/code_final/data/system1_row.mat') #測試資料
mat_data_map = io.loadmat('C:/Users/dl/Desktop/code/data/colormapping2_data.mat') #colormapping 資料

#將.mat轉換成numpy格式
train_data = np.float32(np.transpose(mat_data['system_mix'])).T 
test_data = np.float32(np.transpose(mat_data_test['system1_row'])).T
map_data = np.float32(np.transpose(mat_data_map['colormapping2_data'])).T

#設變數
x_input = 20 #寬
y_input = 20 #高
epochs = 20 #所有資料run完一次，算1個epoch
batch_size = 128 #每次進入模型的圖片數量(1個迭代的資料量)
test = True #標籤，用來記錄是否有自訂測試資料
model_type = 2 #模型選擇
input_size = x_input*y_input
input_label = x_input*y_input+1

#打亂資料順序
for i in range(10):
    np.random.shuffle(train_data) #亂數
    if (test == True):
        np.random.shuffle(test_data)

if (test == False):
    #將資料分成訓練集、驗證集、測試集，比值為8:2
    training_part=int(len(train_data)*0.8) 
    validation_part=training_part+1+int(len(train_data)*0.1)
    testing_part=len(train_data)
    training_data=train_data[0:training_part, 0:input_size];
    validating_data=train_data[training_part:validation_part, 0:input_size];
    testing_data=train_data[validation_part:testing_part, 0:input_size];
    training_target=train_data[0:training_part, x_input*y_input:input_label];
    validating_target=train_data[training_part:validation_part, x_input*y_input:input_label];
    testing_target=train_data[validation_part:testing_part, x_input*y_input:input_label];
elif (test == True):
    training_part=int(len(train_data)*0.8)
    validation_part=len(train_data)
    #將測試資料獨立開來
    training_data=train_data[0:training_part, 0:input_size];
    validating_data=train_data[training_part:validation_part, 0:input_size];
    training_target=train_data[0:training_part, x_input*y_input:input_label];
    validating_target=train_data[training_part:validation_part, x_input*y_input:input_label];
    testing_data = test_data[:,0:input_size];
    testing_target = test_data[:,input_size:input_label];
    

#將colormapping存入陣列
mapping_data = map_data[:,:]; #用於預測
mapping_data_flattern = map_data[:,:]; #用於加入標籤存取

#將資料轉換成20*20*1的格式以進入模型做測試
training_data = training_data.reshape(training_data.shape[0], x_input, y_input, 1).astype('float32')
validating_data = validating_data.reshape(validating_data.shape[0], x_input, y_input, 1).astype('float32')
testing_data = testing_data.reshape(testing_data.shape[0], x_input, y_input, 1).astype('float32')  
mapping_data = mapping_data.reshape(mapping_data.shape[0], x_input, y_input, 1).astype('float32') 

#模型架構
if model_type == 1: #keras CNN-有無腫瘤   
    model = Sequential()
    model.add(Conv2D(filters = 96,kernel_size = (5,5),padding = 'same' ,input_shape = (x_input,y_input,1),activation = 'relu')) #卷積層
    model.add(Conv2D(filters = 96,kernel_size = (5,5),padding = 'same',activation = 'relu')) #卷積層
    model.add(MaxPooling2D(pool_size=(2,2))) #池化層
    model.add(Conv2D(filters = 96,kernel_size = (5,5),padding = 'same',activation = 'relu')) #卷積層
    model.add(Flatten()) #將數據展平，以進入全連接層
    model.add(Dense(units=1024,activation=tf.nn.relu)) #全連接層
    model.add(Dense(units=512,activation=tf.nn.relu)) #全連階層
    model.add(Dropout(0.5)) #隨機丟棄神經元，0.5=50%的神經元被丟棄
    model.add(Dense(2,activation=tf.nn.softmax)) #輸出層
    model.summary() #印出模型各層狀況
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.save('my_model.h5') #儲存模型
elif(model_type == 2): #keras CNN-有無惡性腫瘤
    model = Sequential()
    model.add(Conv2D(filters = 96,kernel_size = (5,5),padding = 'same' ,input_shape = (x_input,y_input,1),activation = 'relu'))
    model.add(Conv2D(filters = 96,kernel_size = (5,5),padding = 'same' ,activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=1024,activation=tf.nn.relu,input_dim=x_input*y_input))
    model.add(Dropout(0.5))
    model.add(Dense(2,activation=tf.nn.softmax))
    model.summary()
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.save('my_model.h5')
#為了防止發生震盪，每三次準確率提升實作一次學習率的調整
learning_rate_function = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

#將資料寫入模型
train_history=model.fit(training_data, training_target,validation_data=(validating_data,validating_target),epochs=epochs,batch_size=batch_size,verbose=2)
show_result()

        
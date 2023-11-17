# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 00:18:39 2023

@author: Jahirul
"""

import pandas as pd
import os
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten, Reshape, Conv2D, MaxPooling2D
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Dense, Dropout
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
import seaborn as sns
import os
from tensorflow.keras.callbacks import ModelCheckpoint
import time
from tqdm import tqdm
from keras.utils import plot_model
import os
import random
from einops import rearrange

#getting current working directory
cwd= os.getcwd()

#fetching MNIST dataset
df = pd.read_csv(cwd+'\\MNIST_HW4.csv')
save_dir=cwd+"\\Output"

#creating output ditrectory
os.makedirs(save_dir, exist_ok=True)
print(df.isnull().any())
print(df.head())
#df=df[:-1]
len(df)

#Function to calculate Symmetric Mean Absolute Percentage Error
def smape(a, f):
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)

skf = StratifiedKFold(n_splits=5,random_state=10, shuffle=True)
#reading ground truth
y=df[['label']]
#encoding the feature value
enc = OneHotEncoder() 
X=df.drop(columns=['label'])/255
skf.get_n_splits(X, y)
print(skf)
#creating a dataframe for storing result
result = pd.DataFrame()
#starting StratifiedKFold for 5 fold crossing
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"Fold {i}:")
    #storing indices of the dataset, which rows will be taken for valdiation and which rows will be taken for testing
    val_index = np.array(random.sample(test_index.tolist(), int(0.2*len(test_index))))
    test_index = np.array(list(set(test_index) - set(val_index)))
    
    #raranging the Xtraing data shape for CNN
    X_train = rearrange(X.values[train_index, :], 's (c h w) -> s h w c', h=28, w=28, c=1)
    print(X_train.shape)
    #taking the rows from the ground truth for training
    Y_train = y.values[train_index, :]
    #fit transforming
    Y_train=enc.fit_transform(Y_train).toarray()
    X_val = rearrange(X.values[val_index, :], 's (c h w) -> s h w c', h=28, w=28, c=1)
    Y_val = y.values[val_index]
    Y_val=enc.transform(Y_val).toarray()
    X_test = rearrange(X.values[test_index, :], 's (c h w) -> s h w c', h=28, w=28, c=1)
    Y_test = y.values[test_index]
    Y_test=enc.transform(Y_test).toarray()
    
    #printing the value
    print("X_train.shape = "+str(X_train.shape))
    print("Y_train = "+str(Y_train.shape))
    print( "X_val.shape = "+str(X_val.shape))
    print( "Y_val = "+str(Y_val.shape))
    print("X_test.shape = "+str(X_test.shape))
    print("Y_test.shape = "+str(Y_test.shape))
    
    #creating the model backbone
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(rate=0.3))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10))
    model.add(Softmax())
    model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    #ploting the model architechture
    plot_model(model, to_file=save_dir+'\model_architecture.png',show_shapes=True)
    
    #saving the checkpoint
    checkpoint_path = os.path.join(save_dir,  f'Fold {i} model.ckpt')
    # Create a callback that saves the model's weights
    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                            monitor = 'val_loss',
                                            mode = 'auto',
                                            save_best_only = True,
                                            save_weights_only=True,
                                            verbose=1)
    start = time.time()
    # fit the model
    validation=[X_val,Y_val]
    history = model.fit(X_train, Y_train,
                        epochs=10,
                        batch_size=8,
                        validation_data=validation,
                        verbose=1,
                        callbacks=[cp_callback])
    Total_Time = time.time() - start
    #calculating the model running time
    print("Total time: ", Total_Time, "seconds")

    print("val loss:",min(history.history['val_loss']))
    print("val accuaray:",max(history.history['accuracy']))
    
    # plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['accuracy'], label='Validation accuracy')
    plt.plot(history.history['val_loss'], label='Validation loss')
    if i==0:
        plt.legend()
    #saving the plot
    plt.savefig(os.path.join(save_dir, f'Train_val_loss.png'), dpi=300)
    
    #predicting
    y_pred = model.predict(X_test)
    print(y_pred.shape, Y_test.shape)
    
    #calculating the validation loss, accuracy, mean square error, RMSE, MAE, Smape, PCC and stored them in a dictionary
    import math
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score , confusion_matrix
    from scipy.stats import pearsonr
    results = {}
    results['dataset_name'] = f"Fold {i}"
    results['loss'] = min(history.history['val_loss'])
    results['Accuaray'] = max(history.history['accuracy'])
    results['mean_squared_error'] = round(mean_squared_error(Y_test[:,-1], y_pred[:,-1]),3)
    results['root_mean_squared_error'] = round(math.sqrt(mean_squared_error(Y_test[:,-1], y_pred[:,-1])),3)
    results['mean_absolute_error'] = round(mean_absolute_error(Y_test[:,-1], y_pred[:,-1]),3)
    results['r2_score'] = round(r2_score(Y_test[:,-1], y_pred[:,-1]),3)
    results['smape'] = round(smape(Y_test[:,-1], y_pred[:,-1]),3)
    pcc, _ = pearsonr(Y_test[:,-1], y_pred[:,-1])
    results['pcc']= round(pcc,3)
    results['Total Time (sec)'] = round(Total_Time,3)
    print(results)
    result_df = pd.DataFrame(results, index=[0])
    result=pd.concat([result,result_df],axis=0)
try:
    os.mkdir(save_dir)
except:
    pass
#converting the dictionary to a csv and save
result.to_csv(os.path.join(save_dir, f'data_prediction_metrics.csv'), index = False, header=True)























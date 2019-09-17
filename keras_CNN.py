
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sys
import gc
import tensorflow as tf
import keras.backend as K

from keras import regularizers
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Lambda
from keras.layers import GRU, Dense, Activation, Dropout, concatenate, Input, BatchNormalization
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from keras.models import Sequential, Model
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import matplotlib.pyplot as plt
import warnings
import os
import pickle
import time
from tensorflow.python.client import timeline
import re
import time
import os

# Based on a public kernel https://www.kaggle.com/higepon/keras-rnn-gru-to-handle-passbands-as-timeseries


# In[3]:


# Read CSV files
train = pd.read_csv('/modules/cs342/Assignment2/training_set.csv')
meta_train = pd.read_csv('/modules/cs342/Assignment2/training_set_metadata.csv')

# Scale mjd flux and flux error of time series
ss1 = StandardScaler()
train[['mjd', 'flux', 'flux_err']] = ss1.fit_transform(train[['mjd', 'flux', 'flux_err']])

# Sort them by id, passband and time
train = train.sort_values(['object_id', 'passband', 'mjd'])

# Convert them into bins based on fluxband, flux err and detected
train_timeseries = train.groupby(['object_id', 'passband'])['flux', 'flux_err', 'detected'].apply(lambda df: df.reset_index(drop=True)).unstack()
train_timeseries.fillna(0, inplace=True)
train_timeseries.columns = ['_'.join(map(str,tup)).rstrip('_') for tup in train_timeseries.columns.values]

# Get the number of columns or the sum of bins
num_columns = len(train_timeseries.columns)

# Reshape them to (X, 6, 216) where X is the number of rows
X_train = train_timeseries.values.reshape(-1, 6, num_columns).transpose(0, 2, 1)

# Get unique classes
classes = sorted(meta_train.target.unique())
class_map = dict()
for i,val in enumerate(classes):
    class_map[val] = i

# Get merged_meta_train based on train_timeseries0
train_timeseries0 = train_timeseries.reset_index()
train_timeseries0 = train_timeseries0[train_timeseries0.passband == 0]
merged_meta_train = train_timeseries0.merge(meta_train, on='object_id', how='left')
merged_meta_train.fillna(0, inplace=True)

y = merged_meta_train.target
classes = sorted(y.unique())

# Taken from Giba's topic : https://www.kaggle.com/titericz
# https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
# with Kyle Boone's post https://www.kaggle.com/kyleboone
class_weight = {
    c: 1 for c in classes
}
for c in [64, 15]:
    class_weight[c] = 2

print('Unique classes : ', classes)

# Convert y labels to one hot encoding categorical values
targets = merged_meta_train.target
target_map = np.zeros((targets.shape[0],))
target_map = np.array([class_map[val] for val in targets])
Y = to_categorical(target_map)


# In[4]:


def multi_weighted_logloss(y_ohe, y_p):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1-1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos
    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss

def plot_loss_acc(history):
    plt.plot(history.history['loss'][1:])
    plt.plot(history.history['val_loss'][1:])
    plt.title('model loss')
    plt.ylabel('val_loss')
    plt.xlabel('epoch')
    plt.legend(['train','Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['acc'][1:])
    plt.plot(history.history['val_acc'][1:])
    plt.title('model Accuracy')
    plt.ylabel('val_acc')
    plt.xlabel('epoch')
    plt.legend(['train','Validation'], loc='upper left')
    plt.show()


# In[5]:


batch_size = 512

def weight_variable(shape, name=None):
    return np.random.normal(scale=.01, size=shape)

def build_model():
    """
    Build Convolutional Neural Network model here.
    There are 3 different convolutional modules.
    Each module has:
    1) Convolution 1D layer of filter size 256.
    2) Batch Normalization layer to improve accuracy.
    3) MaxPooling 1D layer of filter size 4.
    4) Dropout Rate of 0.25 to reduce overfitting.
    """
    input = Input(shape=(X_train.shape[1], 6), dtype='float32', name='input0')

    output = Conv1D(256, kernel_size=80, strides=4, padding='same')(input)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = MaxPooling1D(pool_size=4, strides=None)(output)
    output = Dropout(0.25)(output)

    output = Conv1D(256, kernel_size=80, strides=4, padding='same')(input)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = MaxPooling1D(pool_size=4, strides=None)(output)
    output = Dropout(0.25)(output)

    output = Conv1D(256, kernel_size=3, strides=1, padding='same')(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = MaxPooling1D(pool_size=4, strides=None)(output)
    output = Dropout(0.25)(output)

    output = Lambda(lambda x: K.mean(x, axis=1))(output) # Same as GAP for 1D Conv Layer
    output = Dense(len(classes), activation='softmax')(output)

    model = Model(inputs=input, outputs=output)
    return model

# https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69795
def mywloss(y_true,y_pred):
    yc=tf.clip_by_value(y_pred,1e-15,1-1e-15)
    loss=-(tf.reduce_mean(tf.reduce_mean(y_true*tf.log(yc),axis=0)/wtable))
    return loss


# In[7]:


epochs = 100
y_count = Counter(target_map)
y_map = target_map

wtable = np.zeros((len(classes),))
for i in range(len(classes)):
    wtable[i] = float(y_count[i])/float(y_map.shape[0]) # BUG HERE

y_categorical = Y
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
start = time.time()
clfs = []
oof_preds = np.zeros((len(X_train), len(classes)))

model_file = "model.weights"


for fold_, (trn_, val_) in enumerate(folds.split(y_map, y_map)):
    checkPoint = ModelCheckpoint(model_file, monitor='val_loss',mode = 'min', save_best_only=True, verbose=0)

    # Split train and test
    x_train, y_train = X_train[trn_], Y[trn_]
    x_valid, y_valid = X_train[val_], Y[val_]

    # Build model
    model = build_model()
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    stopping = EarlyStopping(monitor='val_loss', patience=60, verbose=0, mode='auto')

    model.compile(loss=mywloss, optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                    validation_data=[x_valid, y_valid],
                    epochs=epochs,
                        batch_size=batch_size,
                    shuffle=False,verbose=1,callbacks=[checkPoint, stopping])
    plot_loss_acc(history)

    print('Loading Best Model')
    model.load_weights(model_file)

    # Get predicted probabilities for each class
    oof_preds[val_, :] = model.predict(x_valid,batch_size=batch_size)
    print(multi_weighted_logloss(y_valid, model.predict(x_valid,batch_size=batch_size)))
    clfs.append(model)

print('MULTI WEIGHTED LOG LOSS : %.5f ' % multi_weighted_logloss(Y,oof_preds))

elapsed_time = time.time() - start
print("elapsed_time:", elapsed_time)


# In[ ]:


# Get class names from sample submission
sample_sub = pd.read_csv('/modules/cs342/Assignment2/sample_submission.csv')
class_names = list(sample_sub.columns[1:-1])
del sample_sub;gc.collect()


# ### Make Predictions

# In[ ]:


"""
meta_test = pd.read_csv('/modules/cs342/Assignment2/test_set_metadata.csv')

import time

start = time.time()
chunks = 10000
for i_c, df in enumerate(pd.read_csv('/modules/cs342/Assignment2/test_set.csv', chunksize=chunks, iterator=True)):

    ss1 = StandardScaler()
    df[['mjd', 'flux', 'flux_err']] = ss1.fit_transform(df[['mjd', 'flux', 'flux_err']])

    df = df.sort_values(['object_id', 'passband', 'mjd'])

    df_timeseries = df.groupby(['object_id', 'passband'])['flux', 'flux_err', 'detected'].apply(lambda df: df.reset_index(drop=True)).unstack()
    df_timeseries.fillna(0, inplace=True)

    df_timeseries.columns = ['_'.join(map(str,tup)).rstrip('_') for tup in df_timeseries.columns.values]

    num_columns = len(df_timeseries.columns)

    print(num_columns)

    X_test = df_timeseries.values.reshape(-1, 6, num_columns).transpose(0, 2, 1)

    # Make predictions
    preds = None
    for clf in clfs:
        if preds is None:
            preds = clf.predict(X_test) / folds.n_splits
        else:
            preds += clf.predict(X_test) / folds.n_splits

    # Compute preds_99 as the proba of class not being any of the others
    # preds_99 = 0.1 gives 1.769
    preds_99 = np.ones(preds.shape[0])
    for i in range(preds.shape[1]):
        preds_99 *= (1 - preds[:, i])

    # Store predictions
    preds_df = pd.DataFrame(preds, columns=class_names)
    preds_df['object_id'] = meta_test['object_id']
    preds_df['class_99'] = 0.14 * preds_99 / np.mean(preds_99)

    if i_c == 0:
        preds_df.to_csv('predictions.csv',  header=True, index=False)
    else:
        preds_df.to_csv('predictions.csv',  header=False, mode='a', index=False)

    del preds_df, preds

    if (i_c + 1) % 10 == 0:
        print('%15d done in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))
"""

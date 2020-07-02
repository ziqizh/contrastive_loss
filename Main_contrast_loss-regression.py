import datetime
import numpy as np
import tensorflow as tf
print(tf.__version__)
import tensorflow_addons as tfa

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from model import *
from contrast_loss_utils import *




SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# global configs
EPOCHS = 20
DATA = 'mnist-regression'
NORMALIZE_EMBEDDING = True
# NORMALIZE_EMBEDDING = False
N_DATA_TRAIN = 60000
# N_DATA_TRAIN = 10000
BATCH_SIZE = 32
PROJECTION_DIM = 128
WRITE_SUMMARY = True
ACTIVATION = 'leaky_relu'


optimizer = tf.keras.optimizers.Adam()


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)
print(x_train.shape, x_test.shape)


# simulate low data regime for training
n_train = x_train.shape[0]
shuffle_idx = np.arange(n_train)
np.random.shuffle(shuffle_idx)

x_train = x_train[shuffle_idx][:N_DATA_TRAIN]
y_train = y_train[shuffle_idx][:N_DATA_TRAIN]
print(x_train.shape, y_train.shape)


train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(5000).batch(BATCH_SIZE)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)


# # Train baseline model

mlp = MLP(normalize=NORMALIZE_EMBEDDING, regress=True, activation=ACTIVATION)

mse_loss_obj = tf.keras.losses.MeanSquaredError()
# mse_loss_obj = tf.keras.losses.MeanSquaredLogarithmicError()


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_mae = tf.keras.metrics.MeanAbsoluteError(name='train_MAE')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_mae = tf.keras.metrics.MeanAbsoluteError(name='test_MAE')


@tf.function
def train_step_baseline(x, y):
    with tf.GradientTape() as tape:
        y_preds = mlp(x, training=True)
        loss = mse_loss_obj(y, y_preds)

    gradients = tape.gradient(loss, 
                              mlp.trainable_variables)
    optimizer.apply_gradients(zip(gradients, 
                                  mlp.trainable_variables))

    train_loss(loss)
    train_mae(y, y_preds)



@tf.function
def test_step_baseline(x, y):
    y_preds = mlp(x, training=False)
    t_loss = mse_loss_obj(y, y_preds)
    test_loss(t_loss)
    test_mae(y, y_preds)


model_name = 'baseline'
if WRITE_SUMMARY:
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/%s/%s/%s/train' % (model_name, DATA, current_time)
    test_log_dir = 'logs/%s/%s/%s/test' % (model_name, DATA, current_time)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)




for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_mae.reset_states()
    test_loss.reset_states()
    test_mae.reset_states()

    for x, y in train_ds:
        train_step_baseline(x, y)
        
    if WRITE_SUMMARY:
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('MAE', train_mae.result(), step=epoch)        

    for x_te, y_te in test_ds:
        test_step_baseline(x_te, y_te)

    if WRITE_SUMMARY:
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
            tf.summary.scalar('MAE', test_mae.result(), step=epoch)        

    template = 'Epoch {}, Loss: {}, MAE: {}, Test Loss: {}, Test MAE: {}'
    print(template.format(epoch + 1,
                        train_loss.result(),
                        train_mae.result(),
                        test_loss.result(),
                        test_mae.result()))


# # Stage 1: train encoder with contrastive loss



encoder = Encoder(normalize=NORMALIZE_EMBEDDING, activation=ACTIVATION)




projector = Projector(PROJECTION_DIM, normalize=NORMALIZE_EMBEDDING, activation=ACTIVATION)



# Select metrics to measure the loss and the accuracy of the model. 
# These metrics accumulate the values over epochs and then print the overall result.

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_mae = tf.keras.metrics.MeanAbsoluteError(name='train_contrast_MAE')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_mae = tf.keras.metrics.MeanAbsoluteError(name='test_contrast_MAE')



@tf.function
# train step for the contrast loss
def train_step_contrast(x, y):
    '''
    x: data tensor, shape: (batch_size, data_dim)
    y: data labels, shape: (batch_size, )
    '''
    with tf.GradientTape() as tape:
        y_contrasts = get_contrast_batch_labels_regression(y)
        
        r = encoder(x, training=True)
        z = projector(r, training=True)
        
        D = pdist_euclidean(z)
        d_vec = square_to_vec(D)
        loss = tf.losses.mean_absolute_error(y_contrasts, d_vec)
#         loss = tf.losses.mean_squared_error(y_contrasts, d_vec)
#         loss = mse_loss_obj(y_contrasts, d_vec)

    gradients = tape.gradient(loss, 
                              encoder.trainable_variables + projector.trainable_variables)
    optimizer.apply_gradients(zip(gradients, 
                                  encoder.trainable_variables + projector.trainable_variables))

    train_loss(loss)
    train_mae(y_contrasts, d_vec)



@tf.function
def test_step_contrast(x, y):
    y_contrasts = get_contrast_batch_labels_regression(y)

    r = encoder(x, training=False)
    z = projector(r, training=False)

    D = pdist_euclidean(z)
    d_vec = square_to_vec(D)
    t_loss = tf.losses.mean_absolute_error(y_contrasts, d_vec)
#     t_loss = tf.losses.mean_squared_error(y_contrasts, d_vec)
#     t_loss = mse_loss_obj(y_contrasts, d_vec)

    test_loss(t_loss)
    test_mae(y_contrasts, d_vec)




for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_mae.reset_states()
    test_loss.reset_states()
    test_mae.reset_states()

    for x, y in train_ds:
        train_step_contrast(x, y)

    for x_te, y_te in test_ds:
        test_step_contrast(x_te, y_te)

    template = 'Epoch {}, Loss: {}, MAE: {}, Test Loss: {}, Test MAE: {}'
    print(template.format(epoch + 1,
                        train_loss.result(),
                        train_mae.result(),
                        test_loss.result(),
                        test_mae.result()))


# In[55]:


x_tr_proj = projector(encoder(x_train))
x_tr_proj.shape


# In[56]:


x_te_proj = projector(encoder(x_test))
x_te_proj.shape


# In[57]:


# convert tensor to np.array
x_tr_proj = x_tr_proj.numpy()
x_te_proj = x_te_proj.numpy()
print(x_tr_proj.shape, x_te_proj.shape)


# In[58]:


x_test.shape


# ## Check learned embedding

# In[59]:


from sklearn.decomposition import PCA


# In[60]:


# do PCA for the projected data
pca = PCA(n_components=2)
pca.fit(x_tr_proj)
x_te_proj_pca = pca.transform(x_te_proj)
x_te_proj_pca.shape


# In[61]:


# do PCA for original data
pca = PCA(n_components=2)
pca.fit(x_train)
x_te_pca = pca.transform(x_test)
x_te_pca.shape


# In[62]:


x_te_proj_df = pd.DataFrame(x_te_proj[:, :2], columns=['Proj1', 'Proj2'])
x_te_proj_df['label'] = y_test
# x_te_proj_df.head()


# In[63]:


ax = sns.scatterplot('Proj1', 'Proj2', data=x_te_proj_df,
                palette='tab10',
                hue='label',
                linewidth=0,
                alpha=0.6
               )
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5));


# In[64]:


sns.jointplot('Proj1', 'Proj2', data=x_te_proj_df,
              kind="hex"
             );


# In[65]:


x_te_proj_pca_df = pd.DataFrame(x_te_proj_pca, columns=['PC1', 'PC2'])
x_te_proj_pca_df['label'] = y_test


# In[78]:


ax = sns.scatterplot('PC1', 'PC2', data=x_te_proj_pca_df,
#                 palette='tab10',
                     palette='RdBu_r',
                     legend='full',
                hue='label',
                linewidth=0,
                alpha=0.6
               );

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5));
title = 'Data: %s; Embedding: contrastive'% DATA
if NORMALIZE_EMBEDDING:
    title = 'Data: %s; Embedding: contrastive (normed)'% DATA
ax.set_title(title);


# In[67]:


sns.jointplot('PC1', 'PC2', data=x_te_proj_pca_df,
              kind="hex"
             )


# In[68]:


x_te_pca_df = pd.DataFrame(x_te_pca, columns=['PC1', 'PC2'])
x_te_pca_df['label'] = y_test


# In[40]:


ax = sns.scatterplot('PC1', 'PC2', data=x_te_pca_df,
                palette='tab10',
                hue='label',
                linewidth=0,
                alpha=0.6
               )
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5));


# In[41]:


sns.jointplot('PC1', 'PC2', data=x_te_pca_df,
              kind="hex"
             );


# # Stage 2: freeze the learned representations and then learn a classifier on a linear layer using a softmax loss

# In[69]:


dense1 = tf.keras.layers.Dense(1)


# In[70]:


# Select metrics to measure the loss and the accuracy of the model. 
# These metrics accumulate the values over epochs and then print the overall result.
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_mae = tf.keras.metrics.MeanAbsoluteError(name='train_MAE')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_mae = tf.keras.metrics.MeanAbsoluteError(name='test_MAE')


# In[71]:


mse_loss_obj = tf.keras.losses.MeanSquaredError()
# mse_loss_obj = tf.keras.losses.MeanSquaredLogarithmicError()


# In[72]:


@tf.function
# train step for the 2nd stage
def train_step(x, y):
    '''
    x: data tensor, shape: (batch_size, data_dim)
    y: data labels, shape: (batch_size, )
    '''
    with tf.GradientTape() as tape:        
        r = encoder(x, training=False)
        y_preds = dense1(r, training=True)
        loss = mse_loss_obj(y, y_preds)

    # freeze the encoder, only train the softmax layer
    gradients = tape.gradient(loss, 
                              dense1.trainable_variables) 
    optimizer.apply_gradients(zip(gradients, 
                                  dense1.trainable_variables))

    train_loss(loss)
    train_mae(y, y_preds)


# In[73]:


@tf.function
def test_step(x, y):
    r = encoder(x, training=False)
    y_preds = dense1(r, training=False)

    t_loss = mse_loss_obj(y, y_preds)

    test_loss(t_loss)
    test_mae(y, y_preds)


# In[74]:


model_name = 'contrast_loss_model'
if WRITE_SUMMARY:
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/%s/%s/%s/train' % (model_name, DATA, current_time)
    test_log_dir = 'logs/%s/%s/%s/test' % (model_name, DATA, current_time)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)


# In[75]:


for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_mae.reset_states()
    test_loss.reset_states()
    test_mae.reset_states()

    for x, y in train_ds:
        train_step(x, y)

    if WRITE_SUMMARY:
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_mae.result(), step=epoch)        
        
    for x_te, y_te in test_ds:
        test_step(x_te, y_te)

    if WRITE_SUMMARY:
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', test_mae.result(), step=epoch)        
        
    template = 'Epoch {}, Loss: {}, MAE: {}, Test Loss: {}, Test MAE: {}'
    print(template.format(epoch + 1,
                        train_loss.result(),
                        train_mae.result(),
                        test_loss.result(),
                        test_mae.result()))


# In[46]:


test_mae.result().numpy()


# In[ ]:





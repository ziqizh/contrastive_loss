#!/usr/bin/env python
# coding: utf-8

# # This notebook will use the two-stage framework by Khosla et al. 2020 to compare the following contrastive loss functions for the first stage:
# 
# ## Supervised contrastive loss functions
# - Max margin contrastive loss (Hadsell et al. 2006) 
#     + Euclidean distance
#     + tfa.losses.contrastive_loss
# - Multiclass N-pair loss (Sohn 2016) 
#     + inner product distance
#     + tfa.losses.npairs_loss
# - Supervised NT-Xent (the normalized temperature-scaled cross entropy loss) (equation 4 in Khosla et al. 2020)
#     + inner product distance
# 
# ## Other params
# ### Distances:
# - inner product 
# - Euclidean
# 
# ### Normalize projections
# 


import datetime
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns



from utils import *
from model import *
import losses



SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# global configs
EPOCHS = 20
# DATA = 'mnist'
DATA = 'mnist'
NORMALIZE_EMBEDDING = True
# NORMALIZE_EMBEDDING = False
N_DATA_TRAIN = 60000
# BATCH_SIZE = 32
BATCH_SIZE = 1024 # batch_size for pre-training
PROJECTION_DIM = 128
# WRITE_SUMMARY = True
WRITE_SUMMARY = False
ACTIVATION = 'leaky_relu'
LR = 0.5 # learning rate for pre-training



optimizer = tf.keras.optimizers.Adam(lr=LR)
optimizer2 = tf.keras.optimizers.Adam() # with default lr

@tf.function
# train step for the multiclass N-pair loss
def train_step_sup_nt_xent(x, y):
    '''
    x: data tensor, shape: (batch_size, data_dim)
    y: data labels, shape: (batch_size, )
    '''
    with tf.GradientTape() as tape:        
        r = encoder(x, training=True)
        z = projector(r, training=True)
        loss = losses.supervised_nt_xent_loss(z, y, temperature=0.1)

    gradients = tape.gradient(loss, 
                              encoder.trainable_variables + projector.trainable_variables)
    optimizer.apply_gradients(zip(gradients, 
                                  encoder.trainable_variables + projector.trainable_variables))
    train_loss(loss)


@tf.function
# train step for the multiclass N-pair loss
def train_step_npair(x, y, train_loss):
    '''
    x: data tensor, shape: (batch_size, data_dim)
    y: data labels, shape: (batch_size, )
    '''
    with tf.GradientTape() as tape:        
        r = encoder(x, training=True)
        z = projector(r, training=True)
        loss = losses.multiclass_npairs_loss(z, y)

    gradients = tape.gradient(loss, 
                              encoder.trainable_variables + projector.trainable_variables)
    optimizer.apply_gradients(zip(gradients, 
                                  encoder.trainable_variables + projector.trainable_variables))
    train_loss(loss)


@tf.function
def test_step_npair(x, y, test_loss):
    r = encoder(x, training=False)
    z = projector(r, training=False)
    t_loss = losses.multiclass_npairs_loss(z, y)
    test_loss(t_loss)


@tf.function
# train step for the 2nd stage
def train_step(x, y):
    '''
    x: data tensor, shape: (batch_size, data_dim)
    y: data labels, shape: (batch_size, )
    '''
    with tf.GradientTape() as tape:        
        r = encoder(x, training=False)
        y_preds = softmax(r, training=True)
        loss = cce_loss_obj(y, y_preds)

    # freeze the encoder, only train the softmax layer
    gradients = tape.gradient(loss, 
                              softmax.trainable_variables) 
    optimizer2.apply_gradients(zip(gradients, 
                                  softmax.trainable_variables))

    train_loss(loss)
    train_acc(y, y_preds)


@tf.function
def test_step(x, y):
    r = encoder(x, training=False)
    y_preds = softmax(r, training=False)
    t_loss = cce_loss_obj(y, y_preds)
    test_loss(t_loss)
    test_acc(y, y_preds)

@tf.function
def test_step_sup_nt_xent(x, y):
    r = encoder(x, training=False)
    z = projector(r, training=False)
    t_loss = losses.supervised_nt_xent_loss(z, y, temperature=0.1)
    test_loss(t_loss)


# 0 Load MNIST data

if DATA == 'mnist':
    mnist = tf.keras.datasets.mnist
elif DATA == 'fashion_mnist':
    mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 28*28).astype(np.float32)
x_test = x_test.reshape(-1, 28*28).astype(np.float32)
print("Loading ", DATA, " :", x_train.shape, x_test.shape)


# simulate low data regime for training
n_train = x_train.shape[0]
shuffle_idx = np.arange(n_train)
np.random.shuffle(shuffle_idx)

x_train = x_train[shuffle_idx][:N_DATA_TRAIN]
y_train = y_train[shuffle_idx][:N_DATA_TRAIN]
print("Loading low data regime", DATA, " :", x_train.shape, y_train.shape)

# Prepare dataset
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(5000).batch(BATCH_SIZE)

train_ds2 = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(5000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)


# # 1. Multi-class N-pairs loss
# 
# ## Stage 1: train encoder with multiclass N-pair loss

encoder = Encoder(normalize=NORMALIZE_EMBEDDING, activation=ACTIVATION)
projector = Projector(PROJECTION_DIM, normalize=NORMALIZE_EMBEDDING, activation=ACTIVATION)

def train_encoder(train_ds, test_ds):
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    train_mae = tf.keras.metrics.MeanAbsoluteError(name='train_MAE')
    test_mae = tf.keras.metrics.MeanAbsoluteError(name='test_MAE')

    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        test_loss.reset_states()

        for x, y in train_ds:
            train_step_npair(x, y, train_loss)

        for x_te, y_te in test_ds:
            test_step_npair(x_te, y_te, test_loss)

        template = 'Epoch {}, Loss: {}, Test Loss: {}'
        print(template.format(epoch + 1,
                            train_loss.result(),
                            test_loss.result()))


    x_tr_proj = projector(encoder(x_train))
    x_te_proj = projector(encoder(x_test))

    # convert tensor to np.array
    x_tr_proj = x_tr_proj.numpy()
    x_te_proj = x_te_proj.numpy()
    print(x_tr_proj.shape, x_te_proj.shape)

    return x_tr_proj, x_te_proj


# ## Check learned embedding

# do PCA for the projected data
pca = PCA(n_components=2)
pca.fit(x_tr_proj)
x_te_proj_pca = pca.transform(x_te_proj)
x_te_proj_pca.shape

# do PCA for original data
pca = PCA(n_components=2)
pca.fit(x_train)
x_te_pca = pca.transform(x_test)
x_te_pca.shape

x_te_proj_df = pd.DataFrame(x_te_proj[:, :2], columns=['Proj1', 'Proj2'])
x_te_proj_df['label'] = y_test

ax = sns.scatterplot('Proj1', 'Proj2', data=x_te_proj_df,
                palette='tab10',
                hue='label',
                linewidth=0,
                alpha=0.6
               )
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5));

sns.jointplot('Proj1', 'Proj2', data=x_te_proj_df,
              kind="hex"
             );


x_te_proj_pca_df = pd.DataFrame(x_te_proj_pca, columns=['PC1', 'PC2'])
x_te_proj_pca_df['label'] = y_test


fig, ax = plt.subplots()
ax = sns.scatterplot('PC1', 'PC2', 
                     data=x_te_proj_pca_df,
                     palette='tab10',
                     hue='label',
                     linewidth=0,
                     alpha=0.6,
                     ax=ax
               );

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5));
title = 'Data: %s; Embedding: contrastive'% DATA
if NORMALIZE_EMBEDDING:
    title = 'Data: %s; Embedding: contrastive (normed)'% DATA
ax.set_title(title);
fig.savefig('figs/PCA_plot_%s_contrastive_embed.png' % DATA)


g = sns.jointplot('PC1', 'PC2', data=x_te_proj_pca_df,
              kind="hex"
             )
plt.subplots_adjust(top=0.95)
g.fig.suptitle(title);
g.savefig('figs/Joint_PCA_plot_%s_contrastive_embed.png' % DATA)


x_te_pca_df = pd.DataFrame(x_te_pca, columns=['PC1', 'PC2'])
x_te_pca_df['label'] = y_test


fig, ax = plt.subplots()
ax = sns.scatterplot('PC1', 'PC2', data=x_te_pca_df,
                palette='tab10',
                hue='label',
                linewidth=0,
                alpha=0.6,
                     ax=ax
               )
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5));

title = 'Data: %s; Embedding: None'% DATA
ax.set_title(title);
fig.savefig('figs/PCA_plot_%s_no_embed.png' % DATA)


g = sns.jointplot('PC1', 'PC2', data=x_te_pca_df,
              kind="hex"
             );

plt.subplots_adjust(top=0.95)
g.fig.suptitle(title);
g.savefig('figs/Joint_PCA_plot_%s_no_embed.png' % DATA)

def train_classifier():
    # ## Stage 2: freeze the learned representations and then learn a classifier on a linear layer using a softmax loss
    softmax = SoftmaxPred()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_ACC')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_ACC')


    cce_loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)



    model_name = 'contrast_loss_model'
    if not NORMALIZE_EMBEDDING:
        model_name = 'contrast_loss_model-no_norm'
    if WRITE_SUMMARY:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/%s/%s/%s/train' % (model_name, DATA, current_time)
        test_log_dir = 'logs/%s/%s/%s/test' % (model_name, DATA, current_time)
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)


    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_acc.reset_states()
        test_loss.reset_states()
        test_acc.reset_states()

        for x, y in train_ds2:
            train_step(x, y)

        if WRITE_SUMMARY:
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', train_acc.result(), step=epoch)        
            
        for x_te, y_te in test_ds:
            test_step(x_te, y_te)

        if WRITE_SUMMARY:
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', test_acc.result(), step=epoch)        
            
        template = 'Epoch {}, Loss: {}, Acc: {}, Test Loss: {}, Test Acc: {}'
        print(template.format(epoch + 1,
                            train_loss.result(),
                            train_acc.result() * 100,
                            test_loss.result(),
                            test_acc.result() * 100))



# # 2. Supervised NT-Xent


encoder = Encoder(normalize=NORMALIZE_EMBEDDING, activation=ACTIVATION)
projector = Projector(PROJECTION_DIM, normalize=NORMALIZE_EMBEDDING, activation=ACTIVATION)


train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    test_loss.reset_states()

    for x, y in train_ds:
        train_step_sup_nt_xent(x, y)

    for x_te, y_te in test_ds:
        test_step_sup_nt_xent(x_te, y_te)

    template = 'Epoch {}, Loss: {}, Test Loss: {}'
    print(template.format(epoch + 1,
                        train_loss.result(),
                        test_loss.result()))

x_tr_proj = projector(encoder(x_train))
x_tr_proj.shape


x_te_proj = projector(encoder(x_test))
x_te_proj.shape




# convert tensor to np.array
x_tr_proj = x_tr_proj.numpy()
x_te_proj = x_te_proj.numpy()
print(x_tr_proj.shape, x_te_proj.shape)


# ## Check learned embedding



def plot_data(x_train, x_test, x_tr_proj, x_te_proj, format=''):
    # do PCA for the projected data
    pca = PCA(n_components=2)
    pca.fit(x_tr_proj)
    x_te_proj_pca = pca.transform(x_te_proj)
    x_te_proj_pca.shape


    # do PCA for original data
    pca = PCA(n_components=2)
    pca.fit(x_train)
    x_te_pca = pca.transform(x_test)
    x_te_pca.shape

    x_te_proj_df = pd.DataFrame(x_te_proj[:, :2], columns=['Proj1', 'Proj2'])
    x_te_proj_df['label'] = y_test


    ax = sns.scatterplot('Proj1', 'Proj2', data=x_te_proj_df,
                    palette='tab10',
                    hue='label',
                    linewidth=0,
                    alpha=0.6
                )
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5));


    sns.jointplot('Proj1', 'Proj2', data=x_te_proj_df,
                kind="hex"
                );



    x_te_proj_pca_df = pd.DataFrame(x_te_proj_pca, columns=['PC1', 'PC2'])
    x_te_proj_pca_df['label'] = y_test



    fig, ax = plt.subplots()
    ax = sns.scatterplot('PC1', 'PC2', 
                        data=x_te_proj_pca_df,
                        palette='tab10',
                        hue='label',
                        linewidth=0,
                        alpha=0.6,
                        ax=ax
                );

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5));
    title = 'Data: %s; Embedding: contrastive'% DATA
    if NORMALIZE_EMBEDDING:
        title = 'Data: %s; Embedding: contrastive (normed)'% DATA
    ax.set_title(title);
    fig.savefig('figs/PCA_plot_{}_contrastive_embed_{}.png'.format(DATA, format))



    g = sns.jointplot('PC1', 'PC2', data=x_te_proj_pca_df,
                kind="hex"
                )
    plt.subplots_adjust(top=0.95)
    g.fig.suptitle(title);
    g.savefig('figs/Joint_PCA_plot_{}_contrastive_embed_{}.png'.format(DATA, format))

    x_te_pca_df = pd.DataFrame(x_te_pca, columns=['PC1', 'PC2'])
    x_te_pca_df['label'] = y_test



    fig, ax = plt.subplots()
    ax = sns.scatterplot('PC1', 'PC2', data=x_te_pca_df,
                    palette='tab10',
                    hue='label',
                    linewidth=0,
                    alpha=0.6,
                        ax=ax
                )
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5));

    title = 'Data: %s; Embedding: None'% DATA
    ax.set_title(title);
    fig.savefig('figs/PCA_plot_{}_no_embed_{}.png'.format(DATA, format))


    g = sns.jointplot('PC1', 'PC2', data=x_te_pca_df,
                kind="hex"
                );

    plt.subplots_adjust(top=0.95)
    g.fig.suptitle(title);
    g.savefig('figs/Joint_PCA_plot_{}_no_embed_{}.png'.format(DATA, format))




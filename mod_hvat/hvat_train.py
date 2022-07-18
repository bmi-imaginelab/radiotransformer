import os
import random
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import wandb
# from wandb.keras import WandbCallback

import tensorflow as tf
import tensorflow_addons as tfa

os.environ['PYTHONHASHSEED']=str(1111)
random.seed(1111)
np.random.seed(1111)
tf.random.set_seed(1111)

from hvat_model import *
from hvat_dataset import *

def train(model, train_dataset, test_dataset):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.2)

    model.compile(loss=[[tf.keras.losses.CategoricalCrossentropy()], [tfa.losses.GIoULoss(), tf.keras.losses.MeanSquaredError()]], loss_weights=[1, 0.01], optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), metrics=[[tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy'), tf.keras.metrics.AUC(curve='ROC', name='auc'), tfa.metrics.F1Score(num_classes=3, name='f1'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')], [tf.keras.metrics.MeanSquaredError()]]) #, [tf.keras.metrics.MeanAbsoluteError(name='mae')]])
    
    model.summary()
    model.fit(train_dataset, epochs=100, shuffle=True, validation_data=test_dataset, verbose=1) #, callbacks=[WandbCallback()]) #[checkpoint_cb, early_stopping_cb])

    return model

if __name__ == '__main__':
    # resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='node-1', zone='us-central1-f') #, project='midyear-tempo-333208 ')
    # tf.config.experimental_connect_to_cluster(resolver)
    # tf.tpu.experimental.initialize_tpu_system(resolver)
    # print("All devices: ", tf.config.list_logical_devices('TPU'))
    # strategy = tf.distribute.TPUStrategy(resolver)

    # with strategy.scope():
    train_dataset, test_dataset = physionet1_data()
    model = hvat_teacher_model()
    model = train(model, train_dataset, test_dataset)
    #model.save_weights('/home/moinakbhattacharya/radiotransformer/hvat_saved_checkpoint/main_pretrain_best_60.hdf5')
    
    
    
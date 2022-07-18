import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
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

from loss import *
from rsna import *
from radiotransformer import *

def train(model, train_dataset, val_dataset):
    # es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, mode='min')
    # cp_callback = tf.keras.callbacks.ModelCheckpoint('saved_checkpoint/main_finetune_best_2.hdf5', verbose=1, monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min')
    model.compile(loss=[[tf.keras.losses.CategoricalCrossentropy()], [visual_attention_loss]],  loss_weights=[1, 0.01], optimizer=tfa.optimizers.AdamW(weight_decay=0.001), metrics=[[tf.keras.metrics.CategoricalAccuracy(name="accuracy"), tf.keras.metrics.AUC(curve='ROC', name='auc'), tfa.metrics.F1Score(num_classes=2, name='f1'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')], []])
    model.fit(train_dataset, shuffle=True, epochs=100, validation_data=val_dataset, verbose=1) #, callbacks=[es_callback, cp_callback, WandbCallback()])

    return model

if __name__ == '__main__':
    # resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='node-4', zone='us-central1-f') #, project='secret-outpost-250816')
    # tf.config.experimental_connect_to_cluster(resolver)
    # tf.tpu.experimental.initialize_tpu_system(resolver)
    # print("All devices: ", tf.config.list_logical_devices('TPU'))
    # strategy = tf.distribute.TPUStrategy(resolver)

    # with strategy.scope():
    tf.debugging.set_log_device_placement(True)
    # gpus = tf.config.list_logical_devices('GPU')
    gpus = tf.config.list_physical_devices("GPU")
    print(gpus)
    # mirrored_strategy = tf.distribute.MirroredStrategy(devices= ["/gpu:0"], cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    # print(tf.device('/gpu:0'))
    # strategy = tf.distribute.MirroredStrategy(gpus)
    # strategy = tf.distribute.MirroredStrategy(devices= ["/gpu:0","/gpu:1"], cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    # with strategy.scope():
    # with tf.device('/gpu:0'):
    #     train_dataset, val_dataset, test_dataset = rsna_pneumonia_data()
    #     model = student_teacher_model()
    #     model = train(model, train_dataset, val_dataset)

import tensorflow as tf
import time
import os
import numpy as np


def get_unique_name(name):
    unique_name=time.asctime().replace(" ","_").replace(":","_")
    unique_name_=f"{name}_at_{unique_name}"
    return unique_name_


def get_callback(config,X_train):
    logs=config["logs"]
    unique_dir_name=get_unique_name("tb_log")
    Tensorboard_root_log_dir=os.path.join(logs["log_dir"],logs["tensorboard_dir"],unique_dir_name)
    os.makedirs(Tensorboard_root_log_dir,exist_ok=True)
    file_writer = tf.summary.create_file_writer(logdir=Tensorboard_root_log_dir)
    with file_writer.as_default():
        images = np.reshape(X_train[10:30],(-1,28,28,1))
        tf.summary.image("20 handwritten digit sampels", images,max_outputs=25,step=0 )
    tensorboard_cb=tf.keras.callbacks.TensorBoard(log_dir=Tensorboard_root_log_dir)
    early_stopping_cb=tf.keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)

    artifacts=config["artifacts"]
    CKPT_dir=os.path.join(artifacts["artifact_dir"],artifacts["model_check_points_dir"])
    os.makedirs(CKPT_dir,exist_ok=True)
    CKPT_path=os.path.join(CKPT_dir,"model_ckpt.h5")
    checkpointing_cb=tf.keras.callbacks.ModelCheckpoint(CKPT_path,save_best_only=True)
    return [tensorboard_cb,early_stopping_cb,checkpointing_cb]

  
    

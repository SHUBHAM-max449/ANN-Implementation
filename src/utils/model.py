import os
import tensorflow as tf
import time
import matplotlib.pyplot as plt


def create_model(loss_function,optimizer,metrics,no_classes):
    Layers=[
        tf.keras.layers.Flatten(input_shape=[28,28],name="input_layer"),
        tf.keras.layers.Dense(300,activation="relu",name="hidddenlayer1"),
        tf.keras.layers.Dense(100,activation="relu",name="hidddenlayer2"),
        tf.keras.layers.Dense(no_classes,activation="softmax",name="output_layer")
        ]
    untrained_model=tf.keras.Sequential(Layers)
    untrained_model.summary()
    untrained_model.compile(loss=loss_function,optimizer=optimizer,metrics=metrics)
    return untrained_model      ## <<< untrained model


def get_unique_file_name(file_name):
    unique_filename = time.strftime(f"%Y%m%d_%H%M%S_{file_name}")
    return unique_filename

def get_unique_file_name_for_plot(file_name,epochs):
    unique_filename = time.strftime(f"%Y%m%d_%H%M%S_{file_name}_for_{epochs}_epochs")
    return unique_filename


def save_model(model,model_name,model_dir):
    unique_filename = get_unique_file_name(model_name)
    path_to_model = os.path.join(model_dir,unique_filename)
    model.save(path_to_model)

def save_plot(data,plot_name,plot_dir,epochs):
    unique_filename = get_unique_file_name_for_plot(plot_name,epochs)
    data.plot(figsize=(10, 7))
    plot_path=os.path.join(plot_dir,unique_filename)
    plt.savefig(plot_path)
    plt.show()






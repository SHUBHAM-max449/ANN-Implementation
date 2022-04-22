import tensorflow as tf

def create_model(loss_function,optimizer,metrics,no_classes):
    Layers=[
        tf.keras.layers.Flatten(input_shape=[28,28],name="input_layer"),
        tf.keras.layers.Dense(300,activation="relu",name="hidddenlayer1"),
        tf.keras.layers.Dense(100,activation="relu",name="hidddenlayer2"),
        tf.keras.layers.Dense(no_classes,activation="softmax",name="output_layer")
        ]
    model=tf.keras.Sequential(Layers)
    model.summary()
    model.compile(loss=loss_function,optimizer=optimizer,metrics=metrics)
    return model ## <<< untrained model

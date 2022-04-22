
from utils.common import read_config
from utils.data_managment import get_data
from utils.model import create_model
import argparse

def training(config_path):
    config = read_config(config_path)
    validation_datasize=config["params"]["validation_datasize"]
    (X_train,y_train),(X_valid,y_valid),(X_test,y_test)=get_data(validation_datasize)
    loss_function=config["params"]["loss_function"]
    optimizer=config["params"]["optimizer"]
    metrics=config["params"]["metrics"]
    no_classes=config["params"]["no_classes"]
    model=create_model(loss_function,optimizer,metrics,no_classes)
    #Default batch size is 32
    EPOCS=config["params"]["epochs"]
    VALIDATION=(X_valid,y_valid)
    Trained_model=model.fit(X_train,y_train,epochs=EPOCS,validation_data=VALIDATION)


   

if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c",default="config.yaml")
    parsed_args = args.parse_args()
    training(parsed_args.config)



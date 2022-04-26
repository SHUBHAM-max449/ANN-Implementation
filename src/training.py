import os
import pandas as pd
from utils.common import read_config
from utils.data_managment import get_data
from utils.model import create_model
from utils.model import save_model
from utils.model import save_plot
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
    # Default batch size is 32
    EPOCS = config["params"]["epochs"]
    VALIDATION = (X_valid,y_valid)
    Trained_model = model.fit(X_train,y_train,epochs=EPOCS,validation_data=VALIDATION)
    model_name=config["artifacts"]["model_name"]
    model_dir = config["artifacts"]["models_dir"]
    artifacts_dir = config["artifacts"]["artifact_dir"]
    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path,exist_ok=True)
    save_model(model, model_name, model_dir_path)
    plot_name = config["artifacts"]["plot_name"]
    plot_dir=config["artifacts"]["plot_dir"]
    plot_dir_path=os.path.join(artifacts_dir,plot_dir)
    save_plot(pd.DataFrame(Trained_model.history),plot_name,plot_dir_path,EPOCS)


   

if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c",default="config.yaml")
    parsed_args = args.parse_args()
    training(parsed_args.config)



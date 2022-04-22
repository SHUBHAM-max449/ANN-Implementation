
from utils.common import read_config
from utils.data_managment import get_data
import argparse

def training(config_path):
    config = read_config(config_path)
    validation_datasize=config["params"]["validation_datasize"]
    (X_train,y_train),(X_valid,y_valid),(X_test,y_test)=get_data(validation_datasize)
    print(X_train.shape)
    print(X_valid.shape)
    print(X_test.shape)

if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c",default="config.yaml")
    parsed_args = args.parse_args()

    training(parsed_args.config)



import os
import yaml
import argparse
from logger import logging
from get_data import get_data, read_params
import pandas as pd



def load_and_save(config_path):
    config = read_params(config_path)
    df = get_data(config_path)
    raw_data_path = config['load_data']['raw_data_path']
    df.to_csv(raw_data_path)

    






if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    parsed_args = args.parse_args()
    load_and_save(config_path=parsed_args.config)
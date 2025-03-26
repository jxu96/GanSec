#!/usr/bin/env python3
import argparse
import logging
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()

def get_datasets(path_to_dir):
    df_train = pd.read_csv(f'{path_to_dir}/train.csv')


    df_valid = pd.read_csv(f'{path_to_dir}/valid.csv')

def main():
    args = parse_args()


    
if __name__ == "__main__":
    main()

#!/usr/bin/env python
# coding: utf-8

"""
Pass arguments that are needed to process files
argument 1: Dictionary file
argument 2: path of files that needs to be processed
"""

import argparse
from preprocessing import DataPreprocessing
import json

with open('data-config.json') as f:
  config = json.load(f)

parser = argparse.ArgumentParser(description='Run preprocessing script')
parser.add_argument("-dp", "--data_path",required=True)
parser.add_argument("-wf", "--walk_forward",required=True)
args = vars(parser.parse_args())

data_path = args["data_path"]


not_include_be_list = config['not_include_be_list']
overall_be = config['overall_be']
aggregate_choice = config['aggregate_choice']
target_file = config['target_file']

target_dates = config['target_dates']
target_value = config['target_value']
target_items = config['target_item']
other_target_value = config['other_target_value']
minor_holidays=config['minor_holidays']
major_holidays=config['major_holidays']
n_periods=int(config['n_periods'])

dp = DataPreprocessing(data_path,not_include_be_list,overall_be,target_dates,
                       target_value,target_items,target_file,other_target_value,
                       aggregate_choice,minor_holidays,major_holidays,n_periods,args["walk_forward"]) 
dp.save_transformed_files()


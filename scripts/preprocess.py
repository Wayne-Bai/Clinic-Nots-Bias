import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import tqdm
import json
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import tiktoken

def load_data(path):
    data = pd.read_csv(path)
    return data

def collect_token_num(string, encoding_name):
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def process_data(data):
    encoding_name = 'gpt-4'
    token_num_list = []
    index_list = data.index.values.tolist()
    for i in tqdm.tqdm(range(len(index_list))):
        index = index_list[i]
        string = data['MASK_SEX'][index]
        num_tokens = collect_token_num(string, encoding_name)
        token_num_list.append(num_tokens)
    return token_num_list

def mask_sex(data):
    index_list = data.index.values.tolist()
    for i in tqdm.tqdm(range(len(index_list))):
        index = index_list[i]
        string = data['TEXT'][index]
        if "Sex:" in string:
            pattern = r"Sex:  [MF]"
            repalcement = "Sex: [Redacted]"
            data.loc[index, "MASK_SEX"] = re.sub(pattern, repalcement, string)
        else:
            data.loc[index, "MASK_SEX"] = string
    return data

def multi_mask_sex(data):
    div = len(data) // 100
    data_list = [data[i*div:(i+1)*div] for i in range(100)]
    with ProcessPoolExecutor(100) as executor:
        results = executor.map(mask_sex, data_list)
    
    new_data = pd.concat(results)
    return new_data

def multi_process_data(data, mask=None):
    
    div = len(data) // 100
    data_list = [data[i*div:(i+1)*div] for i in range(100)]
    
    if mask:
        pass
    else:
        with ProcessPoolExecutor(100) as executor:
            print(executor._max_workers)
            results = list(executor.map(process_data, data_list))

    return results

def plot_token_distribution(token_list): 
    # plt.hist(token_list, bins=16, range=(0,8000), alpha=0.3, histtype='stepfilled', color='steelblue', edgecolor='none') # 500
    plt.hist(token_list, bins=8, range=(0,8000), alpha=0.3, histtype='stepfilled', color='steelblue', edgecolor='none') # 1000
    plt.savefig('token_distribute.png')
    plt.show()

if __name__ == '__main__':
    
    data = load_data('dataset/NOTEEVENTS-MIMIC3.csv')
    data.insert(data.shape[1], 'MASK_SEX', '')
    print(len(data))
    print(data.head())

    # data = data[:100]
    mask_data = multi_mask_sex(data)
    # print(mask_data.head())

    token_num_list = multi_process_data(mask_data)
    totol_token_list = []
    for i in token_num_list:
        totol_token_list += i
    # print(totol_token_list)
    plot_token_distribution(totol_token_list)
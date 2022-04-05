import pandas as pd
import argparse
import numpy as np


def get_data(param):
    '''
    データを抽出して出力する関数
    :return: df
    '''

    train_file = param['train_file']
    test_file = param['test_file']

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    return train_df, test_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='./data/train.csv')
    parser.add_argument('--test_file', type=str, default='./data/test.csv')
    args = parser.parse_args()  # 引数解析
    param = vars(args)  # コマンドライン引数を取り込み

    get_data(param)
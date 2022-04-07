import os
import datetime
import argparse
from data_input import get_data
from train import train
from utils import get_device, set_logging
import logging


def main():
    timestamp = "{0:%Y%m%d-%H%M%S}".format(datetime.datetime.now())  # タイムスタンプ
    result_dir = './result/' + timestamp + '/'  # 結果を出力するディレクトリ名
    os.mkdir(result_dir)  # 結果を出力するディレクトリを作成

    # 実行時パラメータ群
    set_logging(result_dir)  # ログを標準出力とファイルに出力するよう設定
    logger = logging.getLogger(__name__)
    logger.info('Setting parameters... ')
    param = vars(args)  # コマンドライン引数を取り込み
    param.update({
        # デバイス
        'device': str(get_device()),
        # 出力ディレクトリ
        'result_dir': result_dir,
        # 学習パラメータ
        'epoch_num': 100
    })  # 追加パラメータ

    # データを取得
    train_df, test_df = get_data(param)

    # 学習
    train(param, train_df, test_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='./data/train.csv')
    parser.add_argument('--test_file', type=str, default='./data/test.csv')
    parser.add_argument('-l', '--load', type=str, default=None)
    args = parser.parse_args()  # 引数解析
    main()

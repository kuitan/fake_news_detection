import os
import datetime
import argparse
from data_input import get_data


def main():
    timestamp = "{0:%Y%m%d-%H%M%S}".format(datetime.datetime.now())  # タイムスタンプ
    result_dir = './result/' + timestamp + '/'  # 結果を出力するディレクトリ名
    os.mkdir(result_dir)  # 結果を出力するディレクトリを作成
    param = vars(args)  # コマンドライン引数を取り込み
    param.update({
        'result_dir': result_dir
    })  # 追加パラメータ

    # データを取得
    get_data(param)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='./data/train.csv')
    parser.add_argument('--test_file', type=str, default='./data/test.csv')
    args = parser.parse_args()  # 引数解析
    main()

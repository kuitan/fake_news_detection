import torch
import logging


logger = logging.getLogger(__name__)


def get_device():
    """
    実行環境のデバイス(GPU or CPU) を取得
    :return: デバイス (Device)
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def highlight(word, attn):
    html_color = '#%02X%02X%02X' % (255, int(255*(1 - attn)), int(255*(1 - attn)))
    return '<span style="background-color: {}">{}</span>'.format(html_color, word)


def mk_html(index, batch, preds, attention_weight, tokenizer, id2cat):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sentence = batch.Text[0][index]
    label = batch.Label[index].item()
    pred = preds[index].item()

    label_str = id2cat[label]
    pred_str = id2cat[pred]

    html = "正解カテゴリ: {}<br>予測カテゴリ: {}<br>".format(label_str, pred_str)

    # 文章の長さ分のzero tensorを宣言
    seq_len = attention_weight.size()[2]
    all_attens = torch.zeros(seq_len).to(device)

    for i in range(12):
        all_attens += attention_weight[index, i, 0, :]

    for word, attn in zip(sentence, all_attens):
        if tokenizer.convert_ids_to_tokens([word.tolist()])[0] == "[SEP]":
            break
        html += highlight(tokenizer.convert_ids_to_tokens([word.numpy().tolist()])[0], attn)
    html += "<br><br>"

    return html


def set_logging(result_dir):
    """
    ログを標準出力とファイルに書き出すよう設定する関数．
    :param result_dir: ログの出力先
    :return: 設定済みのrootのlogger
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # ログレベル
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # ログのフォーマット
    # 標準出力へのログ出力設定
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)  # 出力ログレベル
    handler.setFormatter(formatter)  # フォーマットを指定
    logger.addHandler(handler)
    # ファイル出力へのログ出力設定
    file_handler = logging.FileHandler(f'{result_dir}log.log', 'w')  # ログ出力ファイル
    file_handler.setLevel(logging.DEBUG)  # 出力ログレベル
    file_handler.setFormatter(formatter)  # フォーマットを指定
    logger.addHandler(file_handler)
    return logger
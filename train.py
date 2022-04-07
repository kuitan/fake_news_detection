from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import torchtext
import torch
from torch import nn
import torch.optim as optim
import numpy as np
from transformers import AutoTokenizer, BertModel
from neural_net import BertClassifier
from utils import mk_html
import logging


logger = logging.getLogger(__name__)
# 日本語BERT用のtokenizer
tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')


def train(param, train_dataset, test_dataset):
    result_dir = param['result_dir']
    device = param['device']
    epoch_num = param['epoch_num']
    load_dir = param['load']

    train_data = train_dataset.drop('id', axis=1)
    train_data = train_data.reindex(columns=['text', 'isFake'])
    test_data = test_dataset.drop('id', axis=1)

    train_df, valid_df = train_test_split(train_data, train_size=0.8)
    logger.info(f'train_size: {train_df.shape[0]}')
    logger.info(f'valid_size: {valid_df.shape[0]}')

    train_df.to_csv('./data/train.tsv', sep='\t', index=False, header=None)
    valid_df.to_csv('./data/valid.tsv', sep='\t', index=False, header=None)

    # 分かち書きのテスト
    # text = train_df['text'].iat[0]
    # text_sample = tokenizer.encode(text, return_tensors='pt', max_length=512)
    # logger.info(tokenizer.convert_ids_to_tokens(text_sample[0].tolist()))
    # logger.info(text_sample)

    # 文章のlengthの分布を可視化
    text_length = train_df['text'].map(tokenizer.encode).map(len)
    # logger.info(max(text_length))
    plt.figure()
    sns.distplot(text_length)
    plt.savefig(f'{result_dir}text_length.png')
    plt.close()

    # イテレータを作成
    text = torchtext.data.Field(sequential=True, tokenize=bert_tokenizer, use_vocab=False, lower=False,
                                include_lengths=True, batch_first=True, pad_token=0)
    label = torchtext.data.Field(sequential=False, use_vocab=False)

    train_data, valid_data = torchtext.data.TabularDataset.splits(
        path='./data/', train='train.tsv', test='valid.tsv', format='tsv', fields=[('Text', text), ('Label', label)])

    batch_size = 16
    train_iter, valid_iter = torchtext.data.Iterator.splits((train_data, valid_data),
                                                           batch_sizes=(batch_size, batch_size), repeat=False,
                                                           sort=False)

    # モデルを作成
    # model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    # print(model)

    # batch = next(iter(valid_iter))
    # logger.info(batch.Text[0].size())

    # output_attentions=TrueでAttention weightを取得
    # output = model(batch.Text[0], output_attentions=True)
    # logger.info(output['last_hidden_state'].size())
    # logger.info(output['pooler_output'].size())
    # logger.info('{}, {}'.format(len(output['attentions']), output['attentions'][-1].size()))

    # インスタンスを作成
    classifier = BertClassifier()

    # ファインチューニングの設定
    # 勾配計算を最後のBertLayerモジュールと追加した分類アダプターのみ実行

    # まずは全部OFF
    for param in classifier.parameters():
        param.requires_grad = False

    # BERTの最後の層だけ更新ON
    for param in classifier.bert.encoder.layer[-1].parameters():
        param.requires_grad = True

    # クラス分類のところもON
    for param in classifier.linear.parameters():
        param.requires_grad = True

    # 事前学習済の箇所は学習率小さめ，最後の全結合層は大きめに設定
    optimizer = optim.Adam([
        {'params': classifier.bert.encoder.layer[-1].parameters(), 'lr': 5e-5},
        {'params': classifier.linear.parameters(), 'lr': 1e-4}
    ])

    # 損失関数の設定
    loss_function = nn.NLLLoss()

    # GPUの設定
    classifier.to(device)
    loss_list = []
    loss_valid_list = []

    # 学習
    if load_dir is not None:  # ロードオプションが指定されていたらロード
        classifier.load_state_dict(torch.load(f'{load_dir}transformer_bert.nn', map_location=device))  # モデルを読み込み
    else:
        for epoch in range(epoch_num):
            all_loss = 0
            for idx, batch in enumerate(train_iter):
                classifier.zero_grad()
                input_ids = batch.Text[0].to(device)
                label_ids = batch.Label.to(device)
                out, _ = classifier(input_ids)
                batch_loss = loss_function(out, label_ids)
                batch_loss.backward()
                optimizer.step()
                all_loss += batch_loss.item()
            loss_list.append(all_loss)

            # テスト
            all_valid_loss = 0
            with torch.no_grad():
                for valid_idx, valid_batch in enumerate(valid_iter):
                    input_valid_ids = valid_batch.Text[0].to(device)
                    label_valid_ids = valid_batch.Label.to(device)
                    out, _ = classifier(input_valid_ids)
                    batch_valid_loss = loss_function(out, label_valid_ids)
                    all_valid_loss += batch_valid_loss.item()
            loss_valid_list.append(all_valid_loss)

            logger.info('epoch: %d, traning loss: %.5f, valid loss: %.5f' % (epoch + 1, all_loss, all_valid_loss))

    # モデルの保存
    torch.save(classifier.state_dict(), f'{result_dir}transformer_bert.nn')

    # lossの可視化
    plt.figure()
    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.savefig(f'{result_dir}train_loss.png')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.close()

    plt.figure()
    plt.plot(np.arange(len(loss_valid_list)), loss_valid_list)
    plt.savefig(f'{result_dir}valid_loss.png')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.close()

    # 検証
    answer = []
    prediction = []
    categories = ['Real', 'Fake']
    # カテゴリーのID辞書を作成
    id2cat = dict(zip(list(range(len(categories))), categories))

    with torch.no_grad():
        for batch in valid_iter:
            text_tensor = batch.Text[0].to(device)
            label_tensor = batch.Label.to(device)

            score, _ = classifier(text_tensor)
            _, pred = torch.max(score, 1)

            prediction += list(pred.cpu().numpy())
            answer += list(label_tensor.cpu().numpy())
    print(classification_report(prediction, answer, target_names=categories))

    # Attentionの可視化
    batch = next(iter(valid_iter))
    score, attentions = classifier(batch.Text[0].to(device))
    # 最後の層のAttention weightだけ取得して、サイズを確認
    # logger.info(attentions[-1].size())
    _, pred = torch.max(score, 1)

    with open(f'{result_dir}attention.html', mode='w') as f:
        for i in range(batch_size):
            html_output = mk_html(i, batch, pred, attentions[-1], tokenizer, id2cat)
            f.write(html_output)


def bert_tokenizer(text):
    return tokenizer.encode(text, return_tensors='pt', max_length=512)[0]


if __name__ == '__main__':
    train()
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, DistilBertForSequenceClassification, DistilBertModel


class BertClassifier(nn.Module):
    '''
    BERTにより文書のクラス分類をするクラス
    '''
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        # self.bert = DistilBertModel.from_pretrained('laboro-ai/distilbert-base-japanese')
        # BERTの隠れ層の次元数: 768, カテゴリ数: 2(True or False)
        self.linear = nn.Linear(768, 2)
        # 重み初期化処理
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, input_ids):
        # last_hidden_stateとattentionsを受け取る
        output = self.bert(input_ids, output_attentions=True)
        vec = output['last_hidden_state']
        attentions = output['attentions']
        # 先頭トークンclsのベクトルだけ取得
        vec = vec[:,0,:]
        vec = vec.view(-1, 768)
        # 全結合層でクラス分類用に次元を変換
        out = self.linear(vec)

        return F.log_softmax(out), attentions
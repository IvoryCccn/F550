from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

sentences = [
    "there is a shortage of capital, and we need extra financing",
    "growth is strong and we have plenty of liquidity",
    "there are doubts about our finances",
    "profits are flat"
]

inputs = tokenizer(sentences, return_tensors="pt", padding=True)
outputs = finbert(**inputs)[0]

from scipy.special import softmax
np.around(softmax(outputs.detach().numpy(), axis=1), 4)

sentences = [
    "there is a shortage of capital, and we need extra financing",
    "growth is strong and we have plenty of liquidity",
    "there are doubts about our finances",
    "profits are flat"
]

inputs = tokenizer(sentences, return_tensors="pt", padding=True)
outputs = finbert(**inputs)[0]

from scipy.special import softmax
np.around(softmax(outputs.detach().numpy(), axis=1), 4)

labels = {0: 'neutral', 1: 'positive', 2: 'negative'}
for idx, sent in enumerate(sentences):
  print(sent, '----', labels[np.argmax(outputs.detach().numpy()[idx])])
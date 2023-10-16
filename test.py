
from transformers import AutoTokenizer

import jieba.posseg as psg

#tokenizer = AutoTokenizer.from_pretrained("./bert/bert-large-uncased-whole-word-masking")
tokenizer = AutoTokenizer.from_pretrained("./bert/chinese-roberta-wwm-ext-large")

text = "全民制作人们 大家好 我是练习时长两年半的个人练习生蔡徐坤喜欢唱跳rap篮球NBAmusic和NBA"
text = "蔡徐坤喜欢 唱 跳 rap 篮球 music和NBA"

words = tokenizer.tokenize(text)

token = tokenizer(text, return_tensors="pt")

print(words)
print(token["input_ids"].shape)

seg_cut = psg.lcut(text)

print(seg_cut)

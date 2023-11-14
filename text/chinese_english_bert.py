import torch
import sys
from transformers import AutoTokenizer, AutoModelForMaskedLM
import re

tokenizer = AutoTokenizer.from_pretrained("./bert/chinese-roberta-wwm-ext-large")

models = dict()

class ContinuousHyphenChecker:
    def __init__(self):
        self.hyphen_count = 0
        self.wenhao_count = 0
        self.gantan_count = 0

    def is_second_hyphen(self, item):
        if item == "-":
            self.hyphen_count += 1
            if self.hyphen_count == 2:
                self.hyphen_count = 0  # 重置计数器
                return True
        else:
            self.hyphen_count = 0  # 如果不是"-"，则重置计数器
        return False

    def is_second_gantan(self, item):
        if item == "!":
            self.gantan_count += 1
            if self.gantan_count == 2:
                self.gantan_count = 0  # 重置计数器
                return True
        else:
            self.gantan_count = 0  # 如果不是"-"，则重置计数器
        return False
    
    def is_second_wenhao(self, item):
        if item == "?":
            self.wenhao_count += 1
            if self.wenhao_count == 2:
                self.wenhao_count = 0  # 重置计数器
                return True
        else:
            self.wenhao_count = 0  # 如果不是"-"，则重置计数器
        return False

def get_bert_feature(text, word2ph, device=None):
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"
    if device not in models.keys():
        models[device] = AutoModelForMaskedLM.from_pretrained(
            "./bert/chinese-roberta-wwm-ext-large"
        ).to(device)

    with torch.no_grad():
        tokens = tokenizer.tokenize(text)
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = models[device](**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
    
    # 增加##合并功能
    tokens = ["_"] + tokens + ["_"]
    assert inputs["input_ids"].shape[-1] == len(tokens)

    word_level_feature = []
    for i, token in enumerate(tokens):
        if bool(re.match('^[a-zA-Z#\s]+$', token)):
            res[i].zero_()
        if token.startswith("##"): 
            word_level_feature[-1] = [word_level_feature[-1][0] + res[i], word_level_feature[-1][1] + 1]
        else:
            word_level_feature.append([res[i], 1])
    word_level_feature = [res[0]/res[1] for res in word_level_feature] 

    assert len(word_level_feature) == len(word2ph), "{} | {} | {}".format(text, len(word_level_feature), len(word2ph))
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = word_level_feature[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T


if __name__ == "__main__":
    import torch

    word_level_feature = torch.rand(38, 1024)  # 12个词,每个词1024维特征
    word2phone = [
        1,
        2,
        1,
        2,
        2,
        1,
        2,
        2,
        1,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        1,
    ]

    # 计算总帧数
    total_frames = sum(word2phone)
    print(word_level_feature.shape)
    print(word2phone, total_frames, len(word2phone))
    phone_level_feature = []
    for i in range(len(word2phone)):
        print(word_level_feature[i].shape)

        # 对每个词重复word2phone[i]次
        repeat_feature = word_level_feature[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    print(phone_level_feature.shape)  # torch.Size([36, 1024])

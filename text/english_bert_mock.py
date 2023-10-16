import torch


def get_bert_feature(norm_text, word2ph, device):
    return torch.zeros(1024, sum(word2ph))

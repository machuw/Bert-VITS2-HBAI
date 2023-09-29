import json
from collections import defaultdict
from random import shuffle
from typing import Optional

from tqdm import tqdm
import click
from text.cleaner import clean_text
import re
import os
import random

def keep_english_chinese_japanese(s):
    return re.sub(r"[^a-zA-Z\u4e00-\u9fa5\u3040-\u30FF\u31F0-\u31FF]", "", s)

def extract_files_content(base_path, cleaned_path, language='ZH', spk_pos=2):
    file_data = {}
    out_file = open(cleaned_path, "w", encoding="utf-8")

    # 遍历base_path下的所有文件
    for root, dirs, files in os.walk(base_path):
        for file in files:
            try:
                file_path = os.path.join(root, file)

                if ".lab" not in file_path:
                    continue

                # 获取音频路径
                utt = file_path.replace(".lab", ".wav")
                if not os.path.isfile(utt):
                    continue

                # 获取人物名字
                spk = file_path.strip().split("/")[spk_pos]
                spk = keep_english_chinese_japanese(spk)

                # 读取文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                norm_text, phones, tones, word2ph = clean_text(content, language)

                out_file.write(
                    "{}|{}|{}|{}|{}|{}|{}\n".format(
                        utt,
                        spk,
                        language,
                        norm_text,
                        " ".join(phones),
                        " ".join([str(i) for i in tones]),
                        " ".join([str(i) for i in word2ph]),
                    )
                )
            except Exception as error:
                print("extract_files_content err!", file_path, error)


def create_train_and_val_list(
        path: str, 
        train_path: str,
        val_path: str,
        spk_utt_map: dict, 
        spk_id_map: dict,
        max_text_len: int,
        sample_rate: float):
    current_sid = 0
    
    with open(path, encoding="utf-8") as f:
        for line in f.readlines():
            try:
                utt, spk, language, text, phones, tones, word2ph = line.strip().split("|")

                # 过滤太长的语音，最好的办法应该是去做分割
                if len(text) > max_text_len:
                    continue

                spk_utt_map[spk].append(line)
                if spk not in spk_id_map.keys():
                    spk_id_map[spk] = current_sid
                    current_sid += 1
            except Exception as error:
                print("create_train_and_val_list err!", line, error)

    train_list = []
    val_list = []

    for spk, utts in spk_utt_map.items():
        shuffle(utts)
        val_size = int(len(utts) * sample_rate)

        val_list += random.sample(utts, val_size)
        train_list += [sample for sample in utts if sample not in val_list]

    with open(train_path, "w", encoding="utf-8") as f:
        for line in train_list:
            f.write(line)

    with open(val_path, "w", encoding="utf-8") as f:
        for line in val_list:
            f.write(line)



@click.command()
@click.option("--base-path", default="./Chinese")
@click.option("--cleaned-path", default="filelists/zh.list.cleaned")
@click.option("--train-path", default="filelists/train.zh.list")
@click.option("--val-path", default="filelists/val.zh.list")
@click.option(
    "--config-path",
    default="configs/config.zh.json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--max-text-len", default=500)
@click.option("--language", default="ZH")
@click.option("--sample-rate", default=0.01)
@click.option("--clean/--no-clean", default=True)
def main(
    base_path: str,
    cleaned_path: Optional[str],
    train_path: str,
    val_path: str,
    config_path: str,
    max_text_len: int,
    language: str,
    sample_rate: float,
    clean: bool,
):
    if clean:
        extract_files_content(base_path, cleaned_path, language)

    spk_utt_map = defaultdict(list)
    spk_id_map = {}

    create_train_and_val_list(
        cleaned_path,
        train_path,
        val_path,
        spk_utt_map,
        spk_id_map,
        max_text_len,
        sample_rate)

    config = json.load(open(config_path, encoding="utf-8"))
    config["data"]["spk2id"] = spk_id_map
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

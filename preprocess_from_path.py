import json
from collections import defaultdict
from random import shuffle
from typing import Optional

from tqdm import tqdm
import click
from text.cleaner import clean_text
import re
import os

def keep_english_chinese(s):
    return re.sub(r"[^a-zA-Z\u4e00-\u9fa5]", "", s)

def extract_files_content(base_path, cleaned_path, spk_pos=2):
    file_data = {}
    out_file = open(cleaned_path, "w", encoding="utf-8")

    # 遍历base_path下的所有文件
    try:
        for root, dirs, files in os.walk(base_path):
            for file in files:
                file_path = os.path.join(root, file)

                if ".lab" not in file_path:
                    continue

                # 获取音频路径
                utt = file_path.replace(".lab", ".wav")
                if not os.path.isfile(utt):
                    continue

                # 获取人物名字
                spk = file_path.strip().split("/")[spk_pos]
                spk = keep_english_chinese(spk)

                # 读取文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                language = "ZH"

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
        print("err!", file_path, error)


def create_train_and_val_list(
        path: str, 
        train_path: str,
        val_path: str,
        spk_utt_map: dict, 
        spk_id_map: dict,
        val_per_spk: int,
        max_val_total: int):
    current_sid = 0
    
    with open(path, encoding="utf-8") as f:
        for line in f.readlines():
            try:
                utt, spk, language, text, phones, tones, word2ph = line.strip().split("|")
                spk_utt_map[spk].append(line)

                if spk not in spk_id_map.keys():
                    spk_id_map[spk] = current_sid
                    current_sid += 1
            except Exception as error:
                print("err!", line, error)

    train_list = []
    val_list = []

    for spk, utts in spk_utt_map.items():
        shuffle(utts)
        val_list += utts[:val_per_spk]
        train_list += utts[val_per_spk:]

    if len(val_list) > max_val_total:
        train_list += val_list[max_val_total:]
        val_list = val_list[:max_val_total]

    with open(train_path, "w", encoding="utf-8") as f:
        for line in train_list:
            f.write(line)

    with open(val_path, "w", encoding="utf-8") as f:
        for line in val_list:
            f.write(line)



@click.command()
@click.option("--base-path", default="./Chinese")
@click.option("--cleaned-path", default="filelists/temp.list.cleaned")
@click.option("--train-path", default="filelists/train.list")
@click.option("--val-path", default="filelists/val.list")
@click.option(
    "--config-path",
    default="configs/config.json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--val-per-spk", default=4)
@click.option("--max-val-total", default=8)
@click.option("--clean/--no-clean", default=True)
def main(
    base_path: str,
    cleaned_path: Optional[str],
    train_path: str,
    val_path: str,
    config_path: str,
    val_per_spk: int,
    max_val_total: int,
    clean: bool,
):
    if clean:
        extract_files_content(base_path, cleaned_path)

    spk_utt_map = defaultdict(list)
    spk_id_map = {}

    create_train_and_val_list(
        cleaned_path,
        train_path,
        val_path,
        spk_utt_map,
        spk_id_map,
        val_per_spk,
        max_val_total)

    config = json.load(open(config_path, encoding="utf-8"))
    config["data"]["spk2id"] = spk_id_map
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

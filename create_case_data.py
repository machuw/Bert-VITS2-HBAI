import json
from collections import defaultdict
from random import shuffle
from typing import Optional
from multiprocessing import Pool

from tqdm import tqdm
import click
from text.cleaner import clean_text
import re
import os
import random

def parse_case(case_path: str):
    res = []
    with open(case_path, encoding="utf-8") as f:
        for line in f.readlines():
            text = line.strip().split("text:")[-1].strip()
            res.append(text)
    return res
            

def create_train_and_val_list(
        case_list,
        path: str, 
        train_path: str,
        spk_utt_map: dict, 
        spk_id_map: dict,
        max_text_len: int):
    current_sid = 0
    res = []
    with open(path, encoding="utf-8") as f:
        for line in f.readlines():
            try:
                utt, spk, language, text, phones, tones, word2ph = line.strip().split("|")

                if text not in case_list:
                    continue

                res.append(line.strip())
            except Exception as error:
                print("create_train_and_val_list err!", line, error)

    train_list = []

    with open(train_path, "w", encoding="utf-8") as f:
        for line in res:
            f.write(line)
            f.write("\n")


@click.command()
@click.option("--case-path", default="/root/autodl-tmp/models/Bert-VITS2-HBAI/case/honkai_enzh.txt")
@click.option("--cleaned-path", default="filelists/honkai.cleaned.enzh.list")
@click.option("--train-path", default="filelists/honkai.case.train.enzh.list")
@click.option(
    "--config-path",
    default="configs/config.case.json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--max-text-len", default=500)
def main(
    case_path: str,
    cleaned_path: Optional[str],
    train_path: str,
    config_path: str,
    max_text_len: int,
):
    
    case_list = parse_case(case_path)

    spk_utt_map = defaultdict(list)
    spk_id_map = {}

    create_train_and_val_list(
        case_list,
        cleaned_path,
        train_path,
        spk_utt_map,
        spk_id_map,
        max_text_len)

    config = json.load(open(config_path, encoding="utf-8"))
    config["data"]["spk2id"] = spk_id_map
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
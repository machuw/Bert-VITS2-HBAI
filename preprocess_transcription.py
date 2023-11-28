import json
from collections import defaultdict
from random import shuffle
from typing import Optional
import os

from tqdm import tqdm
import click
from text.cleaner import clean_text
from config import config
from infer import latest_version
import re

preprocess_text_config = config.preprocess_text_config

def keep_english_chinese_japanese(s):
    return re.sub(r"[^a-zA-Z\u4e00-\u9fa5\u3040-\u30FF\u31F0-\u31FF\-]", "", s)


dataset_path = "/root/autodl-tmp/datasets/Honkai/Japanese/"

@click.command()
@click.option("--base-path",default=dataset_path)
@click.option("--transcription-path",default=dataset_path+"esd.list")
@click.option("--language", default="JP")
@click.option("--spk-pos", default=6)
@click.option("-y", "--yml_config")
def preprocess(
    base_path: str,
    transcription_path: str,
    language: str,
    spk_pos: int,
    yml_config: str,  # 这个不要删
):
    out_file = open(transcription_path, "w", encoding="utf-8")

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
                game = file_path.strip().split("/")[spk_pos-2]
                spk = file_path.strip().split("/")[spk_pos] + "-" + game + "-" + language
                spk = keep_english_chinese_japanese(spk)

                # 读取文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                out_file.write(
                    "{}|{}|{}|{}\n".format(
                        utt,
                        spk,
                        language,
                        content
                    )
                )
            except Exception as error:
                print("extract_files_content err!", file_path, error)


if __name__ == "__main__":
    preprocess()

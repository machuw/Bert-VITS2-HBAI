import torch
from multiprocessing import Pool
import commons
import utils
from tqdm import tqdm
from text import check_bert_models, cleaned_text_to_sequence, get_bert
import argparse
import torch.multiprocessing as mp
from config import config
import os


def process_line(item):
    line, args = item
    device = config.bert_gen_config.device
    if config.bert_gen_config.use_multi_device:
        rank = mp.current_process()._identity
        rank = rank[0] if len(rank) > 0 else 0
        if torch.cuda.is_available():
            gpu_id = rank % torch.cuda.device_count()
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cpu")
    wav_path, _, language_str, text, phones, tone, word2ph = line.strip().split("|")
    phone = phones.split(" ")
    tone = [int(i) for i in tone.split(" ")]
    word2ph = [int(i) for i in word2ph.split(" ")]
    word2ph = [i for i in word2ph]
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    phone = commons.intersperse(phone, 0)
    tone = commons.intersperse(tone, 0)
    language = commons.intersperse(language, 0)
    for i in range(len(word2ph)):
        word2ph[i] = word2ph[i] * 2
    word2ph[0] += 1

    bert_path = wav_path.replace(".WAV", ".wav").replace(".wav", ".bert.pt")

    if not args.rebuild:
        try:
            bert = torch.load(bert_path)
            assert bert.shape[-1] == len(phone)
        except Exception:
            bert = get_bert(text, word2ph, language_str, device)
            assert bert.shape[-1] == len(phone)
            torch.save(bert, bert_path)
    else:
        bert = get_bert(text, word2ph, language_str, device)
        assert bert.shape[-1] == len(phone)
        torch.save(bert, bert_path)

    if args.check:
        print("text:{}".format(text)) 
        print("word2ph sum:{}".format(sum(word2ph)))
        print("{} | {} | {}".format("".join(phones), " ".join([str(i) for i in tone]), " ".join([str(i) for i in word2ph])))
        torch.set_printoptions(edgeitems=5, linewidth=160)
        print(bert)
        print(bert.size())

preprocess_text_config = config.preprocess_text_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default=config.bert_gen_config.config_path
    )
    parser.add_argument(
        "--num_processes", type=int, default=config.bert_gen_config.num_processes
    )
    parser.add_argument("--rebuild", action='store_true', default=False)
    parser.add_argument("--check", action='store_true', default=False)
    parser.add_argument("--linear", action='store_true', default=False)
    args = parser.parse_args()
    config_path = args.config
    hps = utils.get_hparams_from_file(config_path)
    check_bert_models()
    lines = []
    with open(hps.data.training_files, encoding="utf-8") as f:
        lines.extend(f.readlines())

    with open(hps.data.validation_files, encoding="utf-8") as f:
        lines.extend(f.readlines())

    if not args.linear:
        num_processes = args.num_processes
        with Pool(processes=num_processes) as pool:
            for _ in tqdm(pool.imap_unordered(process_line, [(line, args) for line in lines]), total=len(lines)):
                pass
    else:
        for line in lines:
            process_line((line, args))

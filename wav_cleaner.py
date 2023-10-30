import numpy as np
import librosa
from collections import defaultdict
import re
import os
from sklearn.mixture import GaussianMixture
from scipy.spatial import distance
import tqdm
from multiprocessing import Pool

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfccs.T

def train_gmm(features):
    gmm = GaussianMixture(n_components=1, max_iter=200, covariance_type='diag', n_init=3, reg_covar=1e-5)
    gmm.fit(features)
    return gmm

def check_speaker(item):
    main_voice, test_voice, threshold = item
    # 提取特征
    main_features = extract_features(main_voice)
    test_features = extract_features(test_voice)

    # 使用GMM训练说话人模型
    main_gmm = train_gmm(main_features)
    
    # 计算得分
    score = main_gmm.score(test_features)

    # 对比得分
    if score > threshold:
        #print("score[{}] > threshold[{}], utts[{}]".format(score, threshold, test_voice))
        return True
    else:
        print("score[{}] <= threshold[{}], utts[{}]".format(score, threshold, test_voice))
        return False

def keep_english_chinese_japanese(s):
    return re.sub(r"[^a-zA-Z\u4e00-\u9fa5\u3040-\u30FF\u31F0-\u31FF\-]", "", s)

def extract_files_content(base_path, spk_pos=2):
    spk_utt_map = defaultdict(list)

    # 遍历base_path下的所有文件
    for root, dirs, files in os.walk(base_path):
        for file in files:
            try:
                file_path = os.path.join(root, file)

                # 获取音频路径
                if not file_path.endswith(".wav"):
                    continue
                utt = file_path
                if not os.path.isfile(utt):
                    continue

                # 获取人物名字
                game = file_path.strip().split("/")[spk_pos-2]
                spk = file_path.strip().split("/")[spk_pos] + "-" + game
                spk = keep_english_chinese_japanese(spk)

                spk_utt_map[spk].append(utt)
            except Exception as error:
                print("extract_files_content err!", file_path, error)
    return spk_utt_map 

def main():
    filename = "/root/autodl-tmp/datasets/Genshin/Chinese"
    #filename = "./test_wav/派蒙/"
    spk_utt_map = extract_files_content(filename, spk_pos=6)

    for spk, utts in spk_utt_map.items():
        print(spk)
        main_voice = utts[0]
        #for test_voice in utts[1:]:
        #    check_speaker((main_voice, test_voice, -70.0))
        num_processes = 10
        with Pool(processes=num_processes) as pool:
            for _ in pool.imap_unordered(check_speaker, [(main_voice, test_voice, -70.0) for test_voice in utts[1:]]):
                pass


if __name__ == "__main__":
    main()

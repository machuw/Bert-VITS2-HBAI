import os
import re

import cn2an
from pypinyin import lazy_pinyin, Style

from text.symbols import punctuation
from text.tone_sandhi import ToneSandhi
from text.english import g2p_en_word

current_file_path = os.path.dirname(__file__)
pinyin_to_symbol_map = {
    line.split("\t")[0]: line.strip().split("\t")[1]
    for line in open(os.path.join(current_file_path, "opencpop-strict.txt")).readlines()
}

import jieba.posseg as psg


rep_map = {
    ",": ", ",
    ".": ". ",
    ":": ": ",
    "!": "! ",
    "?": "? ",
    "-": "- ",
    "…": "… ",
    "：": ", ",
    "；": ", ",
    "，": ", ",
    "。": ". ",
    "！": "! ",
    "？": "? ",
    "\n": ". ",
    "·": ", ",
    "、": ", ",
    "...": "… ",
    "$": ". ",
    "“": "' ",
    "”": "' ",
    "‘": "' ",
    "’": "' ",
    "（": "' ",
    "）": "' ",
    "(": "' ",
    ")": "' ",
    "《": "' ",
    "》": "' ",
    "【": "' ",
    "】": "' ",
    "[": "' ",
    "]": "' ",
    "—": "- ",
    "～": "- ",
    "~": "- ",
    "「": "' ",
    "」": "' ",
 #   "……": "… ",
 #   "--": "- ",
}

tone_modifier = ToneSandhi()


def replace_punctuation(text):
    text = text.replace("嗯", "恩").replace("呣", "母")
    text = text.replace("——", "-")
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))

    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)
    
    # 这里会剔除非中文字符和特殊符号
    replaced_text = re.sub(
        r"[^a-zA-Z\u4e00-\u9fa5\s\d" + "".join(punctuation) + r"]+", "", replaced_text
    )

    return replaced_text


def g2p_old(text):
    pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
    sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
    phones, tones, word2ph = _g2p(sentences)
    assert sum(word2ph) == len(phones)
    #assert len(word2ph) == len(text)  # Sometimes it will crash,you can add a try-catch.
    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]
    return phones, tones, word2ph

def g2p(text):
    phones_list = []
    tones_list = []
    word2ph_list = []
    seg_cut = psg.lcut(text)
    seg_cut = tone_modifier.pre_merge_for_modify(seg_cut)
    for word, pos in seg_cut:
        if word.isspace():
            continue
        if bool(re.match('^[a-zA-Z\s]+$', word)):
            phone, tone, word2ph = _g2p_en(word, pos)
        else:
            phone, tone, word2ph = _g2p_zh(word, pos)
        phones_list += phone
        tones_list += tone 
        word2ph_list += word2ph
    assert sum(word2ph_list) == len(phones_list)
    phones_list = ["_"] + phones_list + ["_"]
    tones_list = [0] + tones_list + [0]
    word2ph_list = [1] + word2ph_list + [1]
    return phones_list, tones_list, word2ph_list


def _get_initials_finals(word):
    initials = []
    finals = []
    orig_initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
    orig_finals = lazy_pinyin(
        word, neutral_tone_with_five=True, style=Style.FINALS_TONE3
    )
    for c, v in zip(orig_initials, orig_finals):
        initials.append(c)
        finals.append(v)
    return initials, finals


def _g2p_zh(word, pos):
    phones_list = []
    tones_list = []
    word2ph = []

    word = word.strip()
    if len(word) == 0:
        return phones_list, tones_list, word2ph

    initials, finals = _get_initials_finals(word.strip())
    finals = tone_modifier.modified_tone(word, pos, finals)
    
    for c, v in zip(initials, finals):
        raw_pinyin = c + v
        # NOTE: post process for pypinyin outputs
        # we discriminate i, ii and iii
        if c == v:
            assert c in punctuation
            phone = [c]
            tone = "0"
            word2ph.append(1)
        else:
            v_without_tone = v[:-1]
            tone = v[-1]

            pinyin = c + v_without_tone
            assert tone in "12345"

            if c:
                # 多音节
                v_rep_map = {
                    "uei": "ui",
                    "iou": "iu",
                    "uen": "un",
                }
                if v_without_tone in v_rep_map.keys():
                    pinyin = c + v_rep_map[v_without_tone]
            else:
                # 单音节
                pinyin_rep_map = {
                    "ing": "ying",
                    "i": "yi",
                    "in": "yin",
                    "u": "wu",
                }
                if pinyin in pinyin_rep_map.keys():
                    pinyin = pinyin_rep_map[pinyin]
                else:
                    single_rep_map = {
                        "v": "yu",
                        "e": "e",
                        "i": "y",
                        "u": "w",
                    }
                    if pinyin[0] in single_rep_map.keys():
                        pinyin = single_rep_map[pinyin[0]] + pinyin[1:]

            assert pinyin in pinyin_to_symbol_map.keys(), (pinyin, word, raw_pinyin)
            phone = pinyin_to_symbol_map[pinyin].split(" ")
            word2ph.append(len(phone))
        phones_list += phone
        tones_list += [int(tone)] * len(phone)
    return phones_list, tones_list, word2ph

def _g2p_en(word, pos):
    return g2p_en_word(word)


def _g2p(segments):
    phones_list = []
    tones_list = []
    word2ph = []
    for seg in segments:
        # Replace all English words in the sentence
        seg = re.sub("[a-zA-Z]+", "", seg)
        seg_cut = psg.lcut(seg)
        initials = []
        finals = []
        seg_cut = tone_modifier.pre_merge_for_modify(seg_cut)
        for word, pos in seg_cut:
            if pos == "eng":
                continue
            # 用lazy_pinyin接口处理：返回切词后的读音列表<声母，韵母带声调>
            # 为什么分声母和韵母？因为paddle的ToneSandhi的变调接口只处理带声调韵母
            sub_initials, sub_finals = _get_initials_finals(word)

             # 用paddle的ToneSandhi做变调处理，只处理带声调韵母
            sub_finals = tone_modifier.modified_tone(word, pos, sub_finals)
            initials.append(sub_initials)
            finals.append(sub_finals)

            assert len(sub_initials) == len(sub_finals) == len(word)
        initials = sum(initials, [])
        finals = sum(finals, [])
        #
        for c, v in zip(initials, finals):
            raw_pinyin = c + v
            # NOTE: post process for pypinyin outputs
            # we discriminate i, ii and iii
            if c == v:
                assert c in punctuation
                phone = [c]
                tone = "0"
                word2ph.append(1)
            else:
                v_without_tone = v[:-1]
                tone = v[-1]

                pinyin = c + v_without_tone
                assert tone in "12345"

                if c:
                    # 多音节
                    v_rep_map = {
                        "uei": "ui",
                        "iou": "iu",
                        "uen": "un",
                    }
                    if v_without_tone in v_rep_map.keys():
                        pinyin = c + v_rep_map[v_without_tone]
                else:
                    # 单音节
                    pinyin_rep_map = {
                        "ing": "ying",
                        "i": "yi",
                        "in": "yin",
                        "u": "wu",
                    }
                    if pinyin in pinyin_rep_map.keys():
                        pinyin = pinyin_rep_map[pinyin]
                    else:
                        single_rep_map = {
                            "v": "yu",
                            "e": "e",
                            "i": "y",
                            "u": "w",
                        }
                        if pinyin[0] in single_rep_map.keys():
                            pinyin = single_rep_map[pinyin[0]] + pinyin[1:]

                assert pinyin in pinyin_to_symbol_map.keys(), (pinyin, seg, raw_pinyin)
                phone = pinyin_to_symbol_map[pinyin].split(" ")
                word2ph.append(len(phone))

            phones_list += phone
            tones_list += [int(tone)] * len(phone)
    return phones_list, tones_list, word2ph

def remove_color_tag(text):
    cleaned_text = re.sub(r'<color=#([0-9A-Fa-f]{8})>', '', text)
    cleaned_text = re.sub(r'</color>', '', cleaned_text)
    return cleaned_text 


def text_normalize(text):
    text = remove_color_tag(text)
    numbers = re.findall(r"\d+(?:\.?\d+)?", text)
    for number in numbers:
        text = text.replace(number, cn2an.an2cn(number), 1)

    text = replace_punctuation(text)
    return text


def get_bert_feature(text, word2ph):
    from text import chinese_bert

    return chinese_bert.get_bert_feature(text, word2ph)


if __name__ == "__main__":
    from text.chinese_english_bert import get_bert_feature

    texts = []
    #texts += ["*yawn* Dear, dear."]
    texts += ["不…不——！500,不可能！"]
    texts += ["一--"]
    texts += ["不--趴--"]
    texts += ["----"]
    texts += ["那、那是……"]
    texts += ["Ahh!! Uhh, um... I'm not shaking, I'm not... Ohhhh..."]
    texts += ["小心-"]
    texts += ["小心--"]
    texts += ["小心---"]
    texts += ["小心----"]
    texts += ["啊！但是《原神》是由,米哈\游自主，  [研发]的一款全.新开放世界.冒险游戏"]
    texts += ["Don't be so childish! I'm sure heroes like them have important things to do."]
    texts += ["不仅要买,还一人一个,不,一人好几个!"]
    texts += ["蔡徐坤喜欢 唱 跳 rap 篮球 music和NBA"]
    for text in texts:
        text = text_normalize(text)
        phones, tones, word2ph = g2p(text)
        bert = get_bert_feature(text, word2ph)

        print(len(phones), len(tones), len(word2ph), sum(word2ph), bert.shape)


# # 示例用法
# text = "这是一个示例文本：,你好！这是一个测试...."
# print(g2p_paddle(text))  # 输出: 这是一个示例文本你好这是一个测试

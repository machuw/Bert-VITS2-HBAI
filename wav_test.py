import argparse
import copy
import json
import os
import uuid

import librosa
import gradio as gr
import numpy as np
import soundfile
from scipy import signal

from gradio_client import Client
import logging
import sys

logger = logging.getLogger(__name__)

VITS_URL = "http://127.0.0.1:7860"
VITS_URL_EN = "http://127.0.0.1:7860"

vits_client = Client(VITS_URL)
vits_client_en = Client(VITS_URL_EN)

g_json_key_text = ""
g_json_key_path = ""
g_load_file = ""
g_load_format = ""

g_max_json_index = 0
g_index = 0
g_batch = 10
g_text_list = []
g_audio_list = []
g_audio2_list = []
g_checkbox_list = []
g_data_json = []

def text_to_speech(text, file, language="中文", role_name="派蒙", talk_speed=1.2):
    try:
        result = vits_client_en.predict(
            text,
            role_name,
            0.2,  # SDP Ratio
            0.6,    # 感情变化程度
            0.8,    # 音素发音长度
            talk_speed,  # 整体语速
            'ZH',
            fn_index=0
        )
        logger.info(result)
        #os.system(f"mv -v \"{result[1]}\" \"{file}\"")
        target_sample_rate = 48000
        data, sample_rate = soundfile.read(result[1])
        resample_ratio = target_sample_rate / sample_rate
        converted_data = signal.resample(data, int(len(data) * resample_ratio))
        soundfile.write(f"{file}", converted_data, target_sample_rate, format='wav')
        print(file)
        return file
    except Exception as e:
        logger.error(e)
        return None


def format(text):
    return text.replace(" ", "_").replace("\'", "_").replace("\"", "_")\
        .replace(",", "_").replace("，", "_").replace("。", "_").replace(".", "_").replace("？", "_").replace("?", "_")\
        .replace("！", "_").replace("!", "_").replace("；", "_").replace(";", "_")\
        .replace("：", "_").replace(":", "_").replace("“", "_").replace("”", "_")\
        .replace("‘", "_").replace("’", "_").replace("【", "_").replace("】", "_")\
        .replace("（", "_").replace("）", "_").replace("(", "_").replace(")", "_")\
        .replace("《", "_").replace("》", "_").replace("<", "_").replace(">", "_")\
        .replace("、", "_").replace("——", "_").replace("-", "_").replace("—", "_")\
        .replace("……", "_").replace("…", "_").replace("/", "_").replace("\\", "_")\
        .replace("|", "_").replace("～", "_").replace("~", "_").replace("`", "_")\
        .replace("·", "_").replace("￥", "_").replace("$", "_").replace("^", "_")\
        .replace("&", "_").replace("*", "_").replace("%", "_").replace("#", "_")

def b_generate():
    lang = "中文"
    input = "default_text/neko_zh.txt"
    output = "audio/neko_zh"
    role = "可莉-Genshin"
    speed = 1.0
     
    text_list = []
    for line in open(input, "r"):
        line = line.strip()
        if len(line) > 0:
            text_list.append(line)
    output_list = text_list[:] 
    
    audio_list = []
    for text in text_list: 
        outfile = f"{output}/{format(text)}.wav"
        audio_list.append(text_to_speech(text, outfile, lang, role, speed))

    output_list += audio_list
    output_list += audio_list

    for _ in range(len(text_list)):
        output_list.append(False)
    
    print(output_list)
    return output_list


def reload_data(index, batch):
    global g_index
    g_index = index
    global g_batch
    g_batch = batch
    datas = g_data_json[index:index+batch]
    output = []
    for d in datas:
        output.append(
            {
                g_json_key_text: d[g_json_key_text],
                g_json_key_path: d[g_json_key_path]
            }
        )
    return output


def b_change_index():
    index = 0
    batch = 20
    global g_index, g_batch
    g_index, g_batch = index, batch
    datas = reload_data(index, batch)
    output = []
    for i , _ in enumerate(datas):
        output.append(_[g_json_key_text])
        #output.append(
        #    gr.Textbox(
        #        label=f"Text {i+index}",
        #        value=_[g_json_key_text]
        #    )
        #    )
    for _ in range(g_batch - len(datas)):
        output.append(
            gr.Textbox(
                label=f"Text",
                value=""
            )
        )
    for _ in datas:
        output.append(_[g_json_key_path])
    for _ in datas:
        output.append(_[g_json_key_path])
    for _ in range(g_batch - len(datas)):
        output.append(None)
    for _ in range(g_batch):
        output.append(False)
    return output


def b_next_index(index, batch):
    if (index + batch) <= g_max_json_index:
        return index + batch , *b_change_index(index + batch, batch)
    else:
        return index, *b_change_index(index, batch)


def b_previous_index(index, batch):
    if (index - batch) >= 0:
        return index - batch , *b_change_index(index - batch, batch)
    else:
        return 0, *b_change_index(0, batch)


def get_next_path(filename):
    base_dir = os.path.dirname(filename)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    for i in range(100):
        new_path = os.path.join(base_dir, f"{base_name}_{str(i).zfill(2)}.wav")
        if not os.path.exists(new_path) :
            return new_path
    return os.path.join(base_dir, f'{str(uuid.uuid4())}.wav')


def b_load_json():
    global g_data_json, g_max_json_index
    with open(g_load_file, 'r', encoding="utf-8") as file:
        g_data_json = file.readlines()
        g_data_json = [json.loads(line) for line in g_data_json]
        g_max_json_index = len(g_data_json) - 1


def b_load_list():
    global g_data_json, g_max_json_index
    with open(g_load_file, 'r', encoding="utf-8") as source:
        data_list = source.readlines()
        for _ in data_list:
            data = _.split('|')
            if (len(data) == 7):
                wav_path, speaker_name, language, text, phones, tones, word2ph = data
                g_data_json.append(
                        {
                            'wav_path':wav_path,
                            'speaker_name':speaker_name,
                            'language':language,
                            'text':text.strip(),
                            'phones':phones,
                            'tones':tones,
                            'word2ph':word2ph,
                        }
                )
            else:
                print("error line:", data)
        g_max_json_index = len(g_data_json) - 1


def b_save_radio(*checkbox_list):
    #new_list = [item for item in checkbox_list]
    res_dict = {}
    for item in checkbox_list:
        print(item)
        if item not in res_dict:
            res_dict[item] = 1
        else:
            res_dict[item] += 1

    return "Audio1:{}    Audio2:{}".format(res_dict["audio1"], res_dict["audio2"])


def b_load_file():
    if g_load_format == "json":
        b_load_json()
    elif g_load_format == "list":
        b_load_list()


def set_global(load_json, load_list, json_key_text, json_key_path, batch):
    global g_json_key_text, g_json_key_path, g_load_file, g_load_format, g_batch

    g_batch = int(batch)
    
    if (load_json != "None"):
        g_load_format = "json"
        g_load_file = load_json
    elif (load_list != "None"):
        g_load_format = "list"
        g_load_file = load_list
    else:
        g_load_format = "list"
        g_load_file = "demo.list"
        
    g_json_key_text = json_key_text
    g_json_key_path = json_key_path

    b_load_file()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--load_json', default="None", help='source file, like demo.json')
    parser.add_argument('--load_list', default="None", help='source file, like demo.list')
    parser.add_argument('--json_key_text', default="text", help='the text key name in json, Default: text')
    parser.add_argument('--json_key_path', default="wav_path", help='the path key name in json, Default: wav_path')
    parser.add_argument('--g_batch', default=36, help='max number g_batch wav to display, Default: 10')

    args = parser.parse_args()

    set_global(args.load_json, args.load_list, args.json_key_text, args.json_key_path, args.g_batch)
    
    with gr.Blocks() as demo:
        with gr.Row():
            btn_theme_dark = gr.Button("明亮模式", link="?__theme=light", scale=1)
            btn_theme_light = gr.Button("深色模式", link="?__theme=dark", scale=1)

        with gr.Row():
            with gr.Column():
                for _ in range(0,g_batch):
                    with gr.Row():
                        text = gr.Textbox(
                            label = "Text",
                            visible = True,
                            scale=5
                        )
                        audio_output = gr.Audio(
                            label="Audio1",
                            visible = True,
                            scale=3
                        )
                        audio2_output = gr.Audio(
                            label="Audio2",
                            visible = True,
                            scale=3
                        )
                        audio_check = gr.Radio(
                            choices=["audio1", "audio2"],
                            type="value",
                            label="Choose Audio"
                        )
                        g_text_list.append(text)
                        g_audio_list.append(audio_output)
                        g_audio2_list.append(audio2_output)
                        g_checkbox_list.append(audio_check)

        with gr.Row():
            btn_save_json = gr.Button("计算结果", visible=True, scale=1)
            text_output = gr.Textbox(label="Message")

        #with gr.Row():
        #    btn_previous_index = gr.Button("上一页")
        #    btn_next_index = gr.Button("下一页")

        #btn_previous_index.click(
        #    b_previous_index,
        #    inputs=[
        #        index_slider,
        #        batchsize_slider,
        #    ],
        #    outputs=[
        #        index_slider,
        #        *g_text_list,
        #        *g_audio_list,
        #        *g_audio2_list,
        #        *g_checkbox_list
        #    ],
        #)
        
        #btn_next_index.click(
        #    b_next_index,
        #    inputs=[
        #        index_slider,
        #        batchsize_slider,
        #    ],
        #    outputs=[
        #        index_slider,
        #        *g_text_list,
        #        *g_audio_list,
        #        *g_audio2_list,
        #        *g_checkbox_list
        #    ],
        #)

        btn_save_json.click(
            b_save_radio,
            inputs=[
                *g_checkbox_list
            ],
            outputs=[
                text_output
            ]
        )

        demo.load(
            b_generate,
            #inputs=[
            #    index_slider,
            #    batchsize_slider,
            #],
            outputs=[
                *g_text_list,
                *g_audio_list,
                *g_audio2_list,
                *g_checkbox_list
            ],
        )
        
    demo.launch()

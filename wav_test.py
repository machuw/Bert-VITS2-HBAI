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
import random

logger = logging.getLogger(__name__)

VITS_URL = "http://127.0.0.1:7858"
VITS_URL_NEW = "http://192.168.1.11:7860"

vits_client = Client(VITS_URL)
vits_client_name = "VITS_MODEL"
vits_client_new = Client(VITS_URL_NEW)
vits_client_new_name = "BERT_VITS2_MODEL"

g_max_json_index = 0
g_index = 0
g_batch = 10
g_text_list = []
g_role_list = []
g_audio_list = []
g_audio2_list = []
g_checkbox_list = []
g_data_json = []
g_audio_map_list = []

def text_to_speech(text, file, language="中文", role_name="派蒙", talk_speed=1.2, model=1):
    try:
        if model == 1:
            result = vits_client.predict(
                    text,
                    language,
                    role_name,
                    0.7,    # 感情变化程度
                    0.668,  # 音素发音长度
                    talk_speed,    # 整体语速
                    fn_index=0
            )
        else:
            result = vits_client_new.predict(
                text,
                'ZH',
                role_name,
                0.6,    # 感情变化程度
                0.8,    # 音素发音长度
                talk_speed,  # 整体语速
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
    global g_audio_map_list

    lang = "中文"
    input = "default_text/online2.txt"
    output = "audio/online"
    role = "可莉"
    g_audio_map_list = []
     
    text_list = []
    spk_list = []
    for line in open(input, "r"):
        spk, text = line.strip().split("\t")
        if len(text) > 0:
            spk_list.append(spk)
            text_list.append(text)
    output_list = text_list[:]
    output_list += spk_list[:] 
    
    tts_list_1 = []
    os.system(f"mkdir -p \"{output}\"")
    for i, text in enumerate(text_list): 
        outfile = f"{output}/temp_{i}.1.wav"
        tts_list_1.append(text_to_speech(text, outfile, lang, spk_list[i], 1.1, 1))

    tts_list_2 = []
    for i, text in enumerate(text_list): 
        outfile = f"{output}/temp_{i}.2.wav"
        tts_list_2.append(text_to_speech(text, outfile, lang, spk_list[i], 1.0, 2))
    
    audio_list_1 = []
    audio_list_2 = []
    for i, _ in enumerate(text_list):
        random_num = random.random()
        if random_num > 0.5:
            audio_list_1.append(tts_list_1[i])
            audio_list_2.append(tts_list_2[i])
            audio_map = {"audio1":vits_client_name, "audio2":vits_client_new_name}
            g_audio_map_list.append(audio_map)
        else:
            audio_list_1.append(tts_list_2[i])
            audio_list_2.append(tts_list_1[i])
            audio_map = {"audio1":vits_client_new_name, "audio2":vits_client_name}
            g_audio_map_list.append(audio_map)

    print(g_audio_map_list)
    output_list += audio_list_1
    output_list += audio_list_2

    for _ in range(len(text_list)):
        output_list.append(False)
    
    print(output_list)
    return output_list


def b_save_radio(*checkbox_list):
    #new_list = [item for item in checkbox_list]
    print(g_audio_map_list)
    res_dict = {vits_client_name:0, vits_client_new_name:0}
    for i, item in enumerate(checkbox_list):
        if not item:
            continue
        name = g_audio_map_list[i][item]
        res_dict[name] += 1
        print(i, item, name, res_dict[name])
    return "{}:{}    {}:{}".format(vits_client_name, res_dict[vits_client_name], 
                                   vits_client_new_name, res_dict[vits_client_new_name])


def set_global( batch):
    global g_batch

    g_batch = int(batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--load_json', default="None", help='source file, like demo.json')
    parser.add_argument('--load_list', default="None", help='source file, like demo.list')
    parser.add_argument('--json_key_text', default="text", help='the text key name in json, Default: text')
    parser.add_argument('--json_key_path', default="wav_path", help='the path key name in json, Default: wav_path')
    parser.add_argument('--g_batch', default=50, help='max number g_batch wav to display, Default: 10')
    parser.add_argument("-p", "--port", default=7865, help="port")

    args = parser.parse_args()

    set_global(args.g_batch)
    
    with gr.Blocks() as demo:

        with gr.Row():
            with gr.Column():
                for i in range(0,g_batch):
                    with gr.Row():
                        text = gr.Textbox(
                            label = "Text "+str(i),
                            visible = True,
                            scale=8
                        )
                        role = gr.Textbox(
                            label = "Role "+str(i),
                            visible = True,
                            scale=1
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
                        g_role_list.append(role)
                        g_audio_list.append(audio_output)
                        g_audio2_list.append(audio2_output)
                        g_checkbox_list.append(audio_check)

        with gr.Row():
            btn_save_json = gr.Button("计算结果", visible=True, scale=1)
            text_output = gr.Textbox(label="Message")


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
            outputs=[
                *g_text_list,
                *g_role_list,
                *g_audio_list,
                *g_audio2_list,
                *g_checkbox_list
            ],
        )
        
    #demo.launch()
    demo.launch(share=False, server_name="0.0.0.0", server_port=int(args.port))

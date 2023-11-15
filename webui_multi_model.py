# flake8: noqa: E402
import time
import sys, os
import logging
import noisereduce as nr

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO, format="| %(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

import torch
import argparse
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence, get_bert
from text.cleaner import clean_text
import gradio as gr
import webbrowser
from scipy import signal
import soundfile as sf
import numpy as np
import noisereduce as nr


net_g = None
net_g_keli = None
net_g_lynette = None
net_g_zwclr = None
net_g_en = None

g_role_map = {
    "可莉": "可莉-Genshin",
    "莫娜": "莫娜-Genshin",
    "爱莉希雅": "爱莉希雅-Honkai",
    "琪亚娜": "胡桃-Genshin",
    "香菱": "香菱-Genshin",
    "艾伦": "艾伦-Genshin",
    "范二爷": "范二爷-Genshin",
    "北斗": "北斗-Genshin",
    "怪鸟": "怪鸟-Genshin",
    "春丽（乌拉拉）": "虎克-Honkai",
    "虎克": "虎克-Honkai",
    "债务处理人": "债务处理人-Genshin",
    "迪奥娜（猫猫）": "迪奥娜-Genshin",
    "迪奥娜": "迪奥娜-Genshin",
    "狂躁的男人": "盗宝团呱-Genshin",
    "盗宝团呱": "盗宝团呱-Genshin",
    "胡桃": "胡桃-Genshin",
    "奥兹": "奥兹-Genshin",
    "多莉": "多莉-Genshin",
    "蜜汁卫兵": "匹克-Honkai",
    "匹克": "匹克-Honkai",
    "枫原万叶（万叶）": "枫原万叶-Genshin",
    "枫原万叶": "枫原万叶-Genshin",
    "砂糖": "砂糖-Genshin",
    "坎蒂丝": "坎蒂丝-Genshin",
}

if sys.platform == "darwin" and torch.backends.mps.is_available():
    device = "mps"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
else:
    device = "cuda"

def mono_to_stereo(audio_mono):
    # 检查音频是否为单声道
    if len(audio_mono.shape) != 1:
        raise ValueError("The input array should be a mono audio")

    # 创建一个新的ndarray，将单声道音频复制到双声道
    audio_stereo = np.stack([audio_mono, audio_mono])
    return audio_stereo.T

def reduced_noise_audio(file, data, sample_rate):
    #target_sample_rate = 48000
    #resample_ratio = target_sample_rate / sample_rate
    #converted_data = signal.resample(data, int(len(data) * resample_ratio))

    reduced_noise = nr.reduce_noise(audio_clip=data, noise_clip=data)
    
    return reduced_noise


def get_text(text, language_str, hps):
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    logger.info("{}|{}|{}|{}".format(norm_text, " ".join([str(i) for i in phone]), " ".join([str(i) for i in tone]), " ".join([str(i) for i in word2ph])))
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert = get_bert(norm_text, word2ph, language_str, device)
    del word2ph
    assert bert.shape[-1] == len(phone), phone

    if language_str == "ZH":
        bert = bert
        ja_bert = torch.zeros(768, len(phone))
    elif language_str == "JA":
        ja_bert = bert
        bert = torch.zeros(1024, len(phone))
    elif language_str == "EN":
        bert = torch.zeros(1024, len(phone))
        ja_bert = torch.zeros(768, len(phone))
    else:
        bert = torch.zeros(1024, len(phone))
        ja_bert = torch.zeros(768, len(phone))

    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, ja_bert, phone, tone, language


def infer(text, sdp_ratio, noise_scale, noise_scale_w, length_scale, sid, language):
    global net_g,  net_g_keli, net_g_lynette, net_g_zwclr
    bert, ja_bert, phones, tones, lang_ids = get_text(text, language, hps)
    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        del phones

        model = None
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)

        if sid == "可莉-Genshin":
            model = net_g_keli
        elif sid == "莫娜-Genshin":
            model = net_g_lynette
            sdp_ratio = 0.2
            noise_scale = 0.1
            noise_scale_w = 0.8
            length_scale = 1
        elif sid == "债务处理人-Genshin":
            model = net_g_zwclr
            speakers = torch.LongTensor([0]).to(device)
        else:
            model = net_g
        
        audio = (
            model.infer(
                x_tst,
                x_tst_lengths,
                speakers,
                tones,
                lang_ids,
                bert,
                ja_bert,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )
        del x_tst, tones, lang_ids, bert, x_tst_lengths, speakers
        return audio


def tts_fn(
    text, language, speaker, noise_scale, noise_scale_w, length_scale
):
    start_time = time.time()
    logger.info("text: {} speaker: {}".format(text, speaker))

    language = 'ZH'
    
    sdp_ratio = 0.2
    noise_scale = 0.6
    noise_scale_w = 0.8
    length_scale = 1

    if speaker in g_role_map:
        speaker = g_role_map[speaker]

    with torch.no_grad():
        audio = infer(
            text,
            sdp_ratio=sdp_ratio,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
            sid=speaker,
            language=language,
        )
        torch.cuda.empty_cache()
    end_time = time.time()
    latency = end_time - start_time
    logger.info("text: {} speaker: {} lentency: {}".format(text, speaker, latency))
    return "Success", (hps.data.sampling_rate, audio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model 1
    parser.add_argument(
        "-m", "--model", default="./trained_models/ROLE_ZH_ENZH_MODEL/20231113/G_99000.pth", help="path of your model"
    )
    parser.add_argument(
        "-c",
        "--config",
        default="./trained_models/ROLE_ZH_ENZH_MODEL/20231113/config.json",
        help="path of your config file",
    )

    # model 2
    parser.add_argument(
        "-m2", "--model2", default="./trained_models/KELI_ZH_ENZH_MODEL/20231113/G_34000.pth", help="path of your model"
    )
    parser.add_argument(
        "-c2",
        "--config2",
        default="./trained_models/KELI_ZH_ENZH_MODEL/20231113/config.json",
        help="path of your config file",
    )


    # model 3
    parser.add_argument(
        "-m3", "--model3", default="./trained_models/ALL_ZH_ENZH_MODEL/20231108/G_254000.pth", help="path of your model"
    )
    parser.add_argument(
        "-c3",
        "--config3",
        default="./trained_models/ALL_ZH_ENZH_MODEL/20231108/config.json",
        help="path of your config file",
    )

    # model 4
    parser.add_argument(
        "-m4", "--model4", default="./trained_models/ROLE_ZH_ENZH_MODEL/20231104/G_179000.pth", help="path of your model"
    )
    parser.add_argument(
        "-c4",
        "--config4",
        default="./trained_models/ROLE_ZH_ENZH_MODEL/20231104/config.json",
        help="path of your config file",
    )

    ## model 5
    #parser.add_argument(
    #    "-m5", "--model5", default="/home/mayuanchao/workspace/Bert-VITS2-HBAI/logs/EN_MODEL/G_66000.pth", help="path of your model"
    #)
    #parser.add_argument(
    #    "-c5",
    #    "--config5",
    #    default="/home/mayuanchao/workspace/Bert-VITS2-HBAI/configs/config.en.json",
    #    help="path of your config file",
    #)

    parser.add_argument(
        "--share", default=False, help="make link public", action="store_true"
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="enable DEBUG-LEVEL log"
    )
    parser.add_argument(
        "-p", "--port", default=7860, help="port"
    )

    args = parser.parse_args()
    if args.debug:
        logger.info("Enable DEBUG-LEVEL log")
        logging.basicConfig(level=logging.DEBUG)
    hps = utils.get_hparams_from_file(args.config)

    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else (
            "mps"
            if sys.platform == "darwin" and torch.backends.mps.is_available()
            else "cpu"
        )
    )

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)
    _ = net_g.eval()
    _ = utils.load_checkpoint(args.model, net_g, None, skip_optimizer=True)

    net_g_keli = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)
    _ = net_g.eval()
    _ = utils.load_checkpoint(args.model2, net_g_keli, None, skip_optimizer=True)

    net_g_lynette = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)
    _ = net_g.eval()
    _ = utils.load_checkpoint(args.model3, net_g_lynette, None, skip_optimizer=True)

    net_g_zwclr = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=2500,
        **hps.model,
    ).to(device)
    _ = net_g.eval()
    _ = utils.load_checkpoint(args.model4, net_g_zwclr, None, skip_optimizer=True)


    speaker_ids = hps.data.spk2id
    speakers = list(speaker_ids.keys())
    languages = ["ZH", "JP", "EN"]
    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column():
                text = gr.TextArea(
                    label="Text",
                    placeholder="Input Text Here",
                    value="吃葡萄不吐葡萄皮，不吃葡萄倒吐葡萄皮。",
                )
                speaker = gr.Dropdown(
                    choices=speakers, value=speakers[0], label="Speaker"
                )
                sdp_ratio = gr.Slider(
                    minimum=0, maximum=1, value=0.2, step=0.1, label="SDP Ratio"
                )
                noise_scale = gr.Slider(
                    minimum=0.1, maximum=2, value=0.5, step=0.1, label="Noise Scale"
                )
                noise_scale_w = gr.Slider(
                    minimum=0.1, maximum=2, value=0.9, step=0.1, label="Noise Scale W"
                )
                length_scale = gr.Slider(
                    minimum=0.1, maximum=2, value=1, step=0.1, label="Length Scale"
                )
                language = gr.Dropdown(
                    choices=languages, value=languages[0], label="Language"
                )
                btn = gr.Button("Generate!", variant="primary")
            with gr.Column():
                text_output = gr.Textbox(label="Message")
                audio_output = gr.Audio(label="Output Audio")

        btn.click(
            tts_fn,
            inputs=[
                text,
                language,
                speaker,
                noise_scale,
                noise_scale_w,
                length_scale,
            ],
            outputs=[text_output, audio_output],
        )

    #webbrowser.open("http://127.0.0.1:7860")
    #app.launch(share=args.share)
    app.launch(share=False, server_name="0.0.0.0", server_port=int(args.port))

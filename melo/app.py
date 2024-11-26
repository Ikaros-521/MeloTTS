# WebUI by mrfakename <X @realmrfakename / HF @mrfakename>
# Demo also available on HF Spaces: https://huggingface.co/spaces/mrfakename/MeloTTS
import gradio as gr
import os, torch, io
# os.system('python -m unidic download')
print("Make sure you've downloaded unidic (python -m unidic download) for this WebUI to work.")
from melo.api import TTS
speed = 1.0
import tempfile
import click
device = 'auto'
# 如有需要，请替换成自己的模型路径
config_path=None
ckpt_path=None
# 整合包内置的屠夫模型测试用代码
# config_path="melo/logs/config/config.json"
# ckpt_path="melo/logs/config/G_1000.pth"
models = {
    'EN': TTS(language='EN', device=device, config_path=config_path, ckpt_path=ckpt_path),
    'ES': TTS(language='ES', device=device, config_path=config_path, ckpt_path=ckpt_path),
    'FR': TTS(language='FR', device=device, config_path=config_path, ckpt_path=ckpt_path),
    'ZH': TTS(language='ZH', device=device, config_path=config_path, ckpt_path=ckpt_path),
    'JP': TTS(language='JP', device=device, config_path=config_path, ckpt_path=ckpt_path),
    'KR': TTS(language='KR', device=device, config_path=config_path, ckpt_path=ckpt_path),
}
speaker_ids = models['EN'].hps.data.spk2id

default_text_dict = {
    'EN': 'The field of text-to-speech has seen rapid development recently.',
    'ES': 'El campo de la conversión de texto a voz ha experimentado un rápido desarrollo recientemente.',
    'FR': 'Le domaine de la synthèse vocale a connu un développement rapide récemment',
    'ZH': 'text-to-speech 领域近年来发展迅速',
    'JP': 'テキスト読み上げの分野は最近急速な発展を遂げています',
    'KR': '최근 텍스트 음성 변환 분야가 급속도로 발전하고 있습니다.',    
}
    
def synthesize(speaker, text, speed, language, progress=gr.Progress()):
    bio = io.BytesIO()
    models[language].tts_to_file(text, models[language].hps.data.spk2id[speaker], bio, speed=speed, pbar=progress.tqdm, format='wav')
    return bio.getvalue()
def load_speakers(language, text):
    if text in list(default_text_dict.values()):
        newtext = default_text_dict[language]
    else:
        newtext = text
    return gr.update(value=list(models[language].hps.data.spk2id.keys())[0], choices=list(models[language].hps.data.spk2id.keys())), newtext
with gr.Blocks() as demo:
    gr.Markdown('# MeloTTS WebUI\n\nA WebUI for MeloTTS.')
    with gr.Group():
        speaker = gr.Dropdown(speaker_ids.keys(), interactive=True, value='EN-US', label='说话人')
        language = gr.Radio(['EN', 'ES', 'FR', 'ZH', 'JP', 'KR'], label='语言', value='EN')
        speed = gr.Slider(label='语速', minimum=0.1, maximum=10.0, value=1.0, interactive=True, step=0.1)
        text = gr.Textbox(label="文本输入框", value=default_text_dict['EN'])
        language.input(load_speakers, inputs=[language, text], outputs=[speaker, text])
    btn = gr.Button('合成', variant='primary')
    aud = gr.Audio(interactive=False)
    btn.click(synthesize, inputs=[speaker, text, speed, language], outputs=[aud])
    gr.Markdown('WebUI by [mrfakename](https://twitter.com/realmrfakename).')
@click.command()
@click.option('--share', '-s', is_flag=True, show_default=True, default=False, help="Expose a publicly-accessible shared Gradio link usable by anyone with the link. Only share the link with people you trust.")
@click.option('--host', '-h', default=None)
@click.option('--port', '-p', type=int, default=None)
def main(share, host, port):
    demo.queue(api_open=False).launch(show_api=False, share=share, server_name=host, server_port=port)

if __name__ == "__main__":
    main()

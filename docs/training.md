## 训练

在训练之前，请以开发模式安装MeloTTS并进入`melo`文件夹。
```
pip install -e .
cd melo
```

### 数据准备
要训练一个TTS模型，我们需要准备音频文件和元数据文件。我们建议使用44100Hz的音频文件，而元数据文件应具有以下格式：

```
path/to/audio_001.wav |<speaker_name>|<language_code>|<text_001>
path/to/audio_002.wav |<speaker_name>|<language_code>|<text_002>
```
转录文本可以通过ASR模型（例如，[whisper](https://github.com/openai/whisper)）获得。可以在`data/example/metadata.list`中找到示例元数据。

然后我们可以运行预处理代码：
```
python preprocess_text.py --metadata data/example/metadata.list 
```
将生成一个配置文件`data/example/config.json`。您可以自由编辑该配置文件中的某些超参数（例如，如果您遇到了CUDA内存不足的问题，可以减少批处理大小）。

### 训练
可以通过以下命令启动训练：
```
bash train.sh <path/to/config.json> <num_of_gpus>
```

我们发现对于某些机器，由于gloo的一个[问题](https://github.com/pytorch/pytorch/issues/2530)，训练有时会崩溃。因此，在`train.sh`中添加了一个自动恢复包装器。

### 推理
只需运行：
```
python infer.py --text "<some text here>" -m /path/to/checkpoint/G_<iter>.pth -o <output_dir>
```


## Training

Before training, please install MeloTTS in dev mode and go to the `melo` folder. 
```
pip install -e .
cd melo
```

### Data Preparation
To train a TTS model, we need to prepare the audio files and a metadata file. We recommend using 44100Hz audio files and the metadata file should have the following format:

```
path/to/audio_001.wav |<speaker_name>|<language_code>|<text_001>
path/to/audio_002.wav |<speaker_name>|<language_code>|<text_002>
```
The transcribed text can be obtained by ASR model, (e.g., [whisper](https://github.com/openai/whisper)). An example metadata can be found in `data/example/metadata.list`

We can then run the preprocessing code:
```
python preprocess_text.py --metadata data/example/metadata.list 
```
A config file `data/example/config.json` will be generated. Feel free to edit some hyper-parameters in that config file (for example, you may decrease the batch size if you have encountered the CUDA out-of-memory issue).

### Training
The training can be launched by:
```
bash train.sh <path/to/config.json> <num_of_gpus>
```

We have found for some machine the training will sometimes crash due to an [issue](https://github.com/pytorch/pytorch/issues/2530) of gloo. Therefore, we add an auto-resume wrapper in the `train.sh`.

### Inference
Simply run:
```
python infer.py --text "<some text here>" -m /path/to/checkpoint/G_<iter>.pth -o <output_dir>
```


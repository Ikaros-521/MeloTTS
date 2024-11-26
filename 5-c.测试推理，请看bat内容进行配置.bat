@echo off

chcp 65001

SET CONDA_PATH=.\Miniconda3
set HF_ENDPOINT=https://hf-mirror.com
set HF_HOME=%CD%\hf_download


REM 激活base环境
CALL %CONDA_PATH%\Scripts\activate.bat %CONDA_PATH%

echo 进入melo文件夹
cd melo
REM 定义数据集文件夹名变量，注意修改为自己的路径
set DATABASE_NAME=屠夫
echo 定义数据集文件夹名变量，注意修改为自己的路径，当前配置的是%DATABASE_NAME%
echo 根据前面几步生成的内容，配置需要合成的内容、模型路径、输出路径
echo python infer.py -t "this is the text to be synthesized" -l EN -m logs/config/G_步数.pth -o output_dir
python infer.py -t "this is the text to be synthesized" -l EN -m logs/config/G_1000.pth -o out

cmd /k
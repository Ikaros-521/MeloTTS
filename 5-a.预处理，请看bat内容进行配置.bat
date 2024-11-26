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
echo 将生成一个配置文件`data/example/config.json`。您可以自由编辑该配置文件中的某些超参数（例如，如果您遇到了CUDA内存不足的问题，可以减少批处理大小）。
python preprocess_text.py --metadata data/%DATABASE_NAME%/metadata.list 

echo 预处理完成，生成的数据集将保存在`data/%DATABASE_NAME%/`文件夹中。
cmd /k
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
echo train.bat path/to/config.json num_of_gpus  修改脚本可以自定义配置文件路径（训练参数自行修改，如显存不够就减少batch_size等），并指定使用的GPU数量(整合包魔改了 写死的单卡)
echo 默认1000步保存一次模型，可根据自己需求修改eval_interval。默认Epoch：10000
train.bat data/%DATABASE_NAME%/config.json 1

echo 训练完成，模型保存在data/%DATABASE_NAME%/
cmd /k
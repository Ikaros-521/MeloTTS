@echo off

chcp 65001

SET CONDA_PATH=.\Miniconda3
set HF_ENDPOINT=https://hf-mirror.com
set HF_HOME=%CD%\hf_download


REM 激活base环境
CALL %CONDA_PATH%\Scripts\activate.bat %CONDA_PATH%

start http://127.0.0.1:7860/

python melo\app.py

cmd /k
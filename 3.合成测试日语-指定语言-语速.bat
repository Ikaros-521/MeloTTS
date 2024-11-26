@echo off

chcp 65001

SET CONDA_PATH=.\Miniconda3
set HF_ENDPOINT=https://hf-mirror.com
set HF_HOME=%CD%\hf_download


REM 激活base环境
CALL %CONDA_PATH%\Scripts\activate.bat %CONDA_PATH%

melo "彼は毎朝ジョギングをして体を健康に保っています。" jp.wav --language JP --speed 1

cmd /k
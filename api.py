# oip install fastapi
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import io
import os

from melo.api import TTS  # 确保 TTS 类导入正确

COUNT = 0
# 初始化 FastAPI
app = FastAPI()

# 定义 TTS 模型实例
tts_model = None

class TTSConfig(BaseModel):
    language: str = Field('ZH', description="使用的语言，如 'EN', 'ZH', 'JP', 等")
    device: str = Field("auto", description="设备选择，默认为 'auto'，自动选择设备. cpu cuda", example="auto")
    use_hf: bool = Field(True, description="是否使用 Hugging Face 模型，默认为 True", example=True)
    config_path: str = Field(None, description="配置文件路径，可为空", example="melo/logs/config/config.json")
    ckpt_path: str = Field(None, description="检查点文件路径，可为空", example="melo/logs/config/G_1000.pth")

class SynthesizeRequest(BaseModel):
    text: str = Field("要合成的文本", description="要合成的文本")
    speaker_id: int = Field(0, description="说话人 ID")
    sdp_ratio: float = Field(0.2, description="SDP（Source-Dependent Parameter）比例，默认为 0.2", example=0.3)
    noise_scale: float = Field(0.6, description="噪声比例，默认为 0.6", example=0.7)
    noise_scale_w: float = Field(0.8, description="噪声宽度比例，默认为 0.8", example=0.9)
    speed: float = Field(1.0, description="语速，默认为 1.0", example=1.2)

# 不存在则创建out文件夹
if not os.path.exists("out"):
    os.makedirs("out")

@app.post("/init")
async def init_tts_model(config: TTSConfig):
    global tts_model
    try:
        # 初始化 TTS 模型
        tts_model = TTS(
            language=config.language,
            device=config.device,
            use_hf=config.use_hf,
            config_path=config.config_path,
            ckpt_path=config.ckpt_path
        )
        return {"success": True, "message": "TTS model initialized successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts")
async def synthesize_tts(request: SynthesizeRequest):
    global COUNT
    if not tts_model:
        raise HTTPException(status_code=400, detail="TTS model not initialized.")
    
    try:
        COUNT = (COUNT + 1) % 1000
        output_path = f"out/{COUNT}.wav"
        # 使用 TTS 模型合成语音
        tts_model.tts_to_file(
            text=request.text,
            speaker_id=request.speaker_id,
            sdp_ratio=request.sdp_ratio,
            noise_scale=request.noise_scale,
            noise_scale_w=request.noise_scale_w,
            speed=request.speed,
            output_path=output_path
        )

        with open(output_path, "rb") as audio_file:
            audio_data = audio_file.read()
        audio_stream = io.BytesIO(audio_data)

        return StreamingResponse(audio_stream, media_type="audio/wav")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 启动 FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8101)

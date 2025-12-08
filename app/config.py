import os
from dataclasses import dataclass
from typing import Literal

@dataclass
class AppConfig:
    """Конфигурация приложения"""
    # Модель
    model_id: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct" #"HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    model_cache_dir: str = os.getenv("MODEL_CACHE_DIR", "/app/models")
    
    # Оборудование
    device: Literal["cuda", "cpu"] = os.getenv("DEVICE", "cpu")
    model_dtype: Literal["float16", "float32"] = os.getenv("MODEL_DTYPE", "float32")
    
    # Приложение
    port: int = int(os.getenv("PORT", 7860))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Ограничения
    max_image_size: int = 1024
    max_video_duration: int = 10  # секунд

config = AppConfig()
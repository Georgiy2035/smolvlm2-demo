import torch
from transformers import SmolVLMProcessor, AutoModelForImageTextToText
from PIL import Image
import logging
from typing import Optional, Tuple
from .config import config

logger = logging.getLogger(__name__)

class ModelHandler:
    def __init__(self):
        self.device = config.device
        self.model = None
        self.processor = None
        self.is_loaded = False
        
    def load_model(self):
        """Загрузка модели с кэшированием"""
        try:
            logger.info(f"Loading model {config.model_id} on {self.device}")
            
            # Проверяем наличие кэша
            import os
            if os.path.exists(config.model_cache_dir):
                logger.info(f"Using cache from {config.model_cache_dir}")
            
            # Загрузка процессора и модели
            self.processor = SmolVLMProcessor.from_pretrained(
                config.model_id,
                cache_dir=config.model_cache_dir
            )
            
            self.model = AutoModelForImageTextToText.from_pretrained(
                config.model_id,
                torch_dtype=torch.float16 if config.model_dtype == "float16" else torch.float32,
                cache_dir=config.model_cache_dir
            ).to(self.device)
            
            self.is_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    


    def vqa(self, image: Image.Image, question: str) -> str:
        """Visual Question Answering"""
        if not self.is_loaded:
            self.load_model()
        
        # Подготовка промпта
        prompt = f"<|user|>\n<image>\n{question}\n<|assistant|>\n"
        
        # Обработка
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)
        
        # Генерация
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256
            )
        
        # Декодирование
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        # Извлекаем только ответ
        answer = generated_text.split("<|assistant|>\n")[-1].strip()
        return answer
    
    def image_caption(self, image: Image.Image) -> str:
        """Генерация описания изображения"""
        return self.vqa(image, "Describe this image in detail.")
    
    def ocr(self, image: Image.Image) -> str:
        """Оптическое распознавание текста"""
        return self.vqa(image, "Extract all text from this image. Return only the text, no explanations.")
    
    def is_image_valid(self, image_path: str) -> Tuple[bool, str]:
        """Валидация изображения"""
        try:
            img = Image.open(image_path)
            img.verify()
            return True, "Valid image"
        except Exception as e:
            return False, f"Invalid image: {str(e)}"
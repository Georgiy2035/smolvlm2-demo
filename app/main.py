import gradio as gr
from PIL import Image
import tempfile
import os
from .model_handler import ModelHandler
from .utils import validate_input, cleanup_temp_files
from .config import config
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация модели
model_handler = ModelHandler()

def load_model():
    """Загрузка модели при старте"""
    try:
        model_handler.load_model()
        return "✅ Model loaded successfully!"
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return f"❌ Failed to load model: {str(e)}"

# Сценарий 1: VQA + Image Captioning
def vqa_interface(image, question, history=None):
    """Интерфейс для визуального вопроса-ответа"""
    if not image:
        return "Please upload an image first.", None
    
    # Валидация
    is_valid, msg = validate_input(image, "image")
    if not is_valid:
        return msg, None
    
    try:
        # Обработка изображения
        pil_image = Image.fromarray(image)
        
        if question:
            # VQA
            answer = model_handler.vqa(pil_image, question)
        else:
            # Image Captioning
            answer = model_handler.image_caption(pil_image)
        
        # Обновление истории
        if history is None:
            history = []
        
        if question:
            history.append((question, answer))
        
        return answer, history
        
    except Exception as e:
        logger.error(f"VQA error: {e}")
        return f"Error: {str(e)}", history

# Сценарий 2: OCR с загрузкой
def ocr_interface(image):
    """Интерфейс для OCR"""
    if not image:
        return "Please upload an image.", None
    
    # Валидация
    is_valid, msg = validate_input(image, "image")
    if not is_valid:
        return msg, None
    
    try:
        # OCR
        pil_image = Image.fromarray(image)
        text = model_handler.ocr(pil_image)
        
        # Сохранение во временный файл
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(text)
            temp_path = f.name
        
        return text, temp_path
        
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return f"Error: {str(e)}", None

def create_interface():
    """Создание Gradio интерфейса"""
    
    # Сценарий 1: VQA + Captioning
    with gr.Blocks(title="SmolVLM2 Demo - VQA & Captioning") as vqa_tab:
        gr.Markdown("# 🔍 Visual Question Answering & Image Captioning")
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Upload Image", type="numpy")
                question_input = gr.Textbox(
                    label="Ask a question about the image",
                    placeholder="What is in this image? Describe it...",
                    lines=2
                )
                vqa_button = gr.Button("Ask Question", variant="primary")
                caption_button = gr.Button("Generate Caption")
                
            with gr.Column(scale=2):
                answer_output = gr.Textbox(label="Answer", lines=6)
                history_output = gr.Chatbot(label="Conversation History")
        
        # Обработчики
        vqa_button.click(
            fn=vqa_interface,
            inputs=[image_input, question_input, history_output],
            outputs=[answer_output, history_output]
        )
        
        caption_button.click(
            fn=lambda img, hist: vqa_interface(img, "", hist),
            inputs=[image_input, history_output],
            outputs=[answer_output, history_output]
        )
    
    # Сценарий 2: OCR
    with gr.Blocks(title="SmolVLM2 Demo - OCR") as ocr_tab:
        gr.Markdown("# 📝 Optical Character Recognition (OCR)")
        
        with gr.Row():
            with gr.Column(scale=1):
                ocr_image_input = gr.Image(label="Upload Image with Text", type="numpy")
                ocr_button = gr.Button("Extract Text", variant="primary")
                
            with gr.Column(scale=2):
                ocr_text_output = gr.Textbox(label="Extracted Text", lines=10)
                ocr_download = gr.File(label="Download Text File")
        
        # Обработчик
        ocr_button.click(
            fn=ocr_interface,
            inputs=[ocr_image_input],
            outputs=[ocr_text_output, ocr_download]
        )
    
    # Главный интерфейс с табами
    demo = gr.TabbedInterface(
        [vqa_tab, ocr_tab],
        ["VQA & Captioning", "OCR"],
        title="SmolVLM2 Multimodal Demo"
    )
    
    return demo

if __name__ == "__main__":
    # Загрузка модели
    load_status = load_model()
    print(load_status)
    
    # Запуск интерфейса
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=config.port,
        share=False
    )
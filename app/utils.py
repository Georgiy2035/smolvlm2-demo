import os
from PIL import Image
import mimetypes

def validate_input(file_path, expected_type):
    """Валидация входных данных"""
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    # Проверка MIME типа
    mime_type, _ = mimetypes.guess_type(file_path)
    
    if expected_type == "image":
        try:
            img = Image.open(file_path)
            img.verify()
            return True, "Valid image"
        except:
            return False, "Invalid image file"
    
    elif expected_type == "video":
        if mime_type and mime_type.startswith('video/'):
            return True, "Valid video"
        return False, "Invalid video file"
    
    return True, "Valid"

def cleanup_temp_files(temp_paths):
    """Очистка временных файлов"""
    for path in temp_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except:
            pass
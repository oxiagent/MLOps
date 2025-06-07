#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import json

def check_gpu():
    """Перевіряє доступність GPU та повертає базову інформацію"""
    result = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    # Додаємо назву пристрою, якщо GPU доступний
    if result["cuda_available"] and result["device_count"] > 0:
        try:
            result["device_name"] = torch.cuda.get_device_name(0)
        except:
            result["device_name"] = "Unknown GPU"
            
    return result

if __name__ == "__main__":
    result = check_gpu()
    print(json.dumps(result, indent=2))
    
    # Записуємо результат у файл для інших скриптів
    with open("gpu_check_result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    # Завершуємо з кодом успіху, якщо GPU доступний
    exit(0 if result["cuda_available"] else 1) 
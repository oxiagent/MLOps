#!/usr/bin/env python3
"""
Скрипт тренування YOLOv8n з інтеграцією Weights & Biases
Тренує модель YOLOv8n на CPU з повним відстеженням W&B та збереженням моделі
Використовує вбудовану інтеграцію YOLO W&B
"""

import os
import yaml
import wandb
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO
import torch

def load_config(config_path="config.yaml"):
    """Завантажує конфігурацію з YAML файлу"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_wandb_environment():
    """Налаштовує середовище W&B та вмикає інтеграцію YOLO W&B"""
    # Завантажуємо змінні середовища
    load_dotenv()
    
    # Отримуємо API ключ W&B з середовища
    wandb_api_key = os.getenv('WANDB_API_KEY')
    if not wandb_api_key:
        print("Warning: WANDB_API_KEY not found in environment variables")
        print("Please set your W&B API key in .env file")
        return False
    
    # Входимо в W&B
    try:
        wandb.login(key=wandb_api_key)
        print("✅ Successfully logged in to W&B")
        
        # Вмикаємо логування W&B в налаштуваннях YOLO
        from ultralytics.utils import SETTINGS
        SETTINGS['wandb'] = True
        print("✅ W&B logging enabled in YOLO settings")
        
        return True
    except Exception as e:
        print(f"❌ Failed to setup W&B: {e}")
        return False

def train_model(config):
    """Тренує модель YOLOv8n з вбудованим відстеженням W&B"""
    
    # Перевизначаємо run_name змінною середовища, якщо встановлено
    run_name = os.getenv('WANDB_RUN_NAME', config['run_name'])
    
    print("🚀 Starting YOLOv8n training on CPU...")
    print(f"📊 W&B Project: {config['wandb_project']}")
    print(f"🏃 Run Name: {run_name}")
    
    # Ініціалізуємо модель
    model = YOLO(config['model'])
    
    # Параметри тренування - YOLO автоматично обробить інтеграцію W&B
    train_args = {
        'data': config['data'],
        'epochs': config['epochs'],
        'batch': config['batch'],
        'imgsz': config['imgsz'],
        'device': config['device'],
        'workers': config['workers'],
        'optimizer': config['optimizer'],
        'lr0': config['lr0'],
        'momentum': config['momentum'],
        'weight_decay': config['weight_decay'],
        'save': config['save'],
        'save_period': config['save_period'],
        'project': config['wandb_project'],  # Назва проєкту W&B
        'name': run_name,                    # Назва запуску W&B (динамічна)
        'plots': True,
        'verbose': True
    }
    
    print(f"🔧 Training parameters: {train_args}")
    
    # Починаємо тренування - YOLO автоматично логуватиме в W&B
    results = model.train(**train_args)
    
    print("✅ Training completed with built-in W&B logging!")
    
    return model, results

def main():
    """Головна функція тренування"""
    print("=" * 60)
    print("🤖 YOLOv8n CPU Training with W&B Integration")
    print("=" * 60)
    
    # Перевіряємо, чи працюємо на CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    
    try:
        # Завантажуємо конфігурацію
        config = load_config()
        
        # Примусово використовуємо CPU, як зазначено в вимогах
        config['device'] = 'cpu'
        
        # Налаштовуємо середовище W&B (вхід та вмикання інтеграції YOLO)
        if not setup_wandb_environment():
            print("⚠️  Continuing without W&B logging")
        
        # Тренуємо модель з вбудованою інтеграцією W&B
        model, results = train_model(config)
        
        # Отримуємо кінцеву назву запуску (може бути перевизначена середовищем)
        final_run_name = os.getenv('WANDB_RUN_NAME', config['run_name'])
        
        print("✅ Training completed successfully!")
        print(f"📁 Results saved in: {config['wandb_project']}/{final_run_name}/")
        print(f"🌐 Check your W&B dashboard at: https://wandb.ai")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
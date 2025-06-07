#!/usr/bin/env python

import os
import argparse
import shutil
from ultralytics import YOLO
from datetime import datetime
from dotenv import load_dotenv
import wandb
import yaml

# Завантажуємо змінні середовища з .env файлу
load_dotenv()

# Функції для отримання параметрів
def get_wandb_params():
    """Отримує параметри W&B з середовища."""
    return {
        'api_key': os.environ.get("WANDB_API_KEY", ""),
        'entity': os.environ.get("WANDB_ENTITY", "")
    }

def get_training_datetime():
    """Повертає поточну дату та час у форматованому вигляді."""
    return {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'time': datetime.now().strftime('%H%M%S')
    }

def get_model_dataset_names(model_type, data):
    """Витягує назви моделі та датасету з шляхів."""
    # Отримуємо назву моделі без розширення
    model_name = model_type.split('.')[0] if '.' in model_type else model_type
    model_name = model_name.split('/')[-1] if '/' in model_name else model_name
    
    # Визначаємо назву датасету
    dataset_name = data.split('.')[0] if '.' in data else data
    dataset_name = dataset_name.split('/')[-1] if '/' in dataset_name else dataset_name
    
    return {
        'model': model_name,
        'dataset': dataset_name
    }

def load_config(config_path="ray_training_config.yaml"):
    """Завантажує та повертає конфігурацію з файлу."""
    if not os.path.exists(config_path):
        print(f"Warning: Configuration file {config_path} not found")
        return None
        
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Warning: Error loading configuration: {e}")
        return None

def train_yolo_model(
    model_type="yolov8n.pt",
    data="coco8.yaml",
    epochs=5,
    batch_size=16,
    imgsz=640,
    device="",
    project=None,
    name=None,
    resume=False
):
    """
    Тренує модель YOLO використовуючи фреймворк Ultralytics та логує результати в Weights & Biases.
    
    Args:
        model_type (str): Шлях до файлу моделі, наприклад yolov8n.pt, yolov8s.pt, тощо.
        data (str): Шлях до YAML файлу даних, наприклад data.yaml
        epochs (int): Кількість епох тренування
        batch_size (int): Розмір батчу для тренування
        imgsz (int): Розмір зображення для тренування
        device (str): Пристрій для тренування, наприклад cuda device=0 або device=0,1,2,3 або device=cpu
        project (str): Назва проєкту для логування W&B
        name (str): Назва експерименту для логування W&B
        resume (bool): Відновити тренування з останньої контрольної точки
        
    Returns:
        YOLO model: Натренована модель
    """
    # Отримуємо параметри W&B
    wandb_params = get_wandb_params()
    
    if not wandb_params['api_key']:
        print("Warning: WANDB_API_KEY not found in environment variables or .env file")
    else:
        print(f"Using W&B API key from environment")
        os.environ["WANDB_API_KEY"] = wandb_params['api_key']
    
    # Отримуємо дату та час
    dt = get_training_datetime()
    date_str = dt['date']
    time_str = dt['time']
    
    # Отримуємо назви моделі та датасету
    names = get_model_dataset_names(model_type, data)
    model_name = names['model']
    dataset_name = names['dataset']
    
    # Якщо проєкт не вказано, намагаємося завантажити з конфігурації
    if project is None:
        config = load_config()
        if config and "wandb_project" in config:
            project = config["wandb_project"]
            print(f"Using project name from config: {project}")
        else:
            print("Error: project name not provided and not found in config")
            return None
    
    entity = wandb_params['entity']
    
    # Генеруємо назву експерименту з часовою міткою, якщо не надано
    if name is None:
        # Створюємо структуровану назву: датасет_модель_дата_час
        name = f"{dataset_name}_{model_name}_{date_str}_{time_str}"
    
    # Увімкнути інтеграцію W&B з Ultralytics
    os.system("yolo settings wandb=True")
    wandb_api_key = os.environ.get("WANDB_API_KEY", "")
    if wandb_api_key:
        os.system(f"yolo settings api_key={wandb_api_key}")
        print(f"Explicitly set Ultralytics API key from environment variable")
    
    print(f"Starting YOLO training:")
    print(f"  - Model: {model_type}")
    print(f"  - Data: {data}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Image size: {imgsz}")
    print(f"  - Device: {device}")
    print(f"  - Project: {project}")
    print(f"  - Run name: {name}")
    print(f"  - Resume: {resume}")
    
    try:
        # Завантажуємо модель
        if resume:
            # Відновлюємо з останньої контрольної точки
            model = YOLO(model_type)
            print(f"Resuming training from {model_type}")
        else:
            # Починаємо нове тренування
            if model_type.endswith('.yaml'):
                # Будуємо нову модель з YAML
                model = YOLO(model_type)
                print(f"Building new model from {model_type}")
            elif model_type.endswith('.pt'):
                # Завантажуємо попередньо натреновану модель
                model = YOLO(model_type)
                print(f"Loading pretrained model {model_type}")
            else:
                raise ValueError(f"Unsupported model format: {model_type}. Use .pt or .yaml files.")
        
        # Тренуємо модель
        results = model.train(
            data=data,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=device,
            project=project,
            name=name,
            exist_ok=True,
            pretrained=True,
            optimizer="Adam",  # Оптимізатор (Adam, SGD, тощо)
            lr0=1e-3,          # Початкова швидкість навчання
            lrf=1e-4,          # Кінцева швидкість навчання
            momentum=0.937,    # SGD momentum/Adam beta1
            weight_decay=5e-4, # Зменшення ваги оптимізатора
            warmup_epochs=1.0, # Епохи розігріву
            warmup_momentum=0.8, # Початковий momentum розігріву
            warmup_bias_lr=0.1,  # Початкова швидкість навчання bias розігріву
            save=True,         # Зберігати контрольні точки
            save_period=1,     # Зберігати контрольну точку кожні x епох
            workers=8,         # Кількість робочих потоків для завантаження даних
            verbose=True,      # Детальний вивід
            seed=42            # Випадкове зерно для відтворюваності
        )
        
        print(f"Training completed successfully")
        
        # Валідуємо модель
        print(f"Running validation...")
        val_results = model.val(
            data=data,
            project=project,
            name=name
        )
        
        print(f"Validation completed:")
        for k, v in val_results.results_dict.items():
            print(f"  - {k}: {v}")
        
        # Логуємо додаткову інформацію в W&B (за потреби)
        if wandb.run is not None:
            # Зберігаємо URL запуску W&B у файл для GitHub Actions
            run_url = wandb.run.get_url()
            print(f"W&B run URL: {run_url}")
            with open("wandb_run_url.txt", "w") as f:
                f.write(run_url)
                
            # Логуємо найкращу модель як артефакт в W&B
            best_model_path = os.path.join(project, name, "weights", "best.pt")
            if os.path.exists(best_model_path):
                # Створюємо назву артефакту у форматі: yolo_модель_датасет_дата
                artifact_name = f"yolo_{model_name}_{dataset_name}_{date_str}"
                
                # Явно вказуємо версію артефакту, щоб уникнути автоматичного перейменування
                run_id = wandb.run.id
                artifact_version = f"v{time_str}"
                
                # Створюємо локальну копію моделі з описовою назвою
                named_model_path = os.path.join(project, f"{artifact_name}.pt")
                try:
                    import shutil
                    shutil.copy2(best_model_path, named_model_path)
                    print(f"Model saved locally as: {named_model_path}")
                except Exception as e:
                    print(f"Warning: Could not save named model copy: {e}")
                
                artifact = wandb.Artifact(
                    name=artifact_name, 
                    type="model",
                    description=f"YOLOv8 model trained on {dataset_name} dataset"
                )
                
                # Додаємо файл моделі з явною назвою файлу
                artifact.add_file(best_model_path, name=f"{artifact_name}.pt")
                
                # Додаємо метадані до артефакту
                artifact.metadata = {
                    "model_type": model_type,
                    "dataset": data,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "image_size": imgsz,
                    "training_date": date_str,
                    "training_time": time_str,
                    "run_name": name,
                    "run_id": run_id,
                    "artifact_version": artifact_version
                }
                
                # Зберігаємо артефакт
                artifact_id = wandb.log_artifact(artifact, aliases=["latest", f"epoch_{epochs}", artifact_version])
                print(f"Best model saved to W&B as artifact: {artifact_name} ({artifact_version})")
                
                # Додатково зберігаємо інформацію про модель в wandb
                wandb.config.update({
                    "model_artifact": artifact_name,
                    "model_artifact_version": artifact_version,
                    "model_artifact_id": artifact_id.id if hasattr(artifact_id, 'id') else None
                })
        
        return model
    
    except Exception as e:
        print(f"Error during model training: {e}")
        return None
    
    finally:
        # Переконуємося, що завершуємо запуск W&B
        if wandb.run is not None:
            # Зберігаємо URL запуску W&B перед завершенням
            if not os.path.exists("wandb_run_url.txt"):
                try:
                    run_url = wandb.run.get_url()
                    with open("wandb_run_url.txt", "w") as f:
                        f.write(run_url)
                    print(f"Saved W&B run URL to wandb_run_url.txt: {run_url}")
                except Exception as e:
                    print(f"Error saving W&B run URL: {e}")
            
            wandb.finish()
            print("W&B logging completed")

if __name__ == "__main__":
    # Намагаємося завантажити конфігурацію для значень за замовчуванням
    config = load_config()
    default_project = None
    
    if config and "wandb_project" in config:
        default_project = config["wandb_project"]
    
    parser = argparse.ArgumentParser(description="Train YOLO model with W&B integration")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Model file (.pt) or configuration (.yaml)")
    parser.add_argument("--data", type=str, default="data.yaml", help="Dataset configuration file")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="", help="Device to use (e.g., cpu, 0, 0,1, mps)")
    parser.add_argument("--project", type=str, default=default_project, help="Project name for W&B (default from config)")
    parser.add_argument("--name", type=str, default=None, help="Run name for W&B")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    
    args = parser.parse_args()
    
    # Тренуємо модель
    model = train_yolo_model(
        model_type=args.model,
        data=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.img_size,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume
    )

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import ray
import json
import subprocess
import yaml
import logging
from datetime import datetime
from dotenv import load_dotenv

# Завантажуємо змінні середовища
load_dotenv()

# Вимикаємо непотрібний вивід від Ray
os.environ["RAY_DISABLE_JUPYTER_PROGRESS"] = "1"
logging.getLogger("ray").setLevel(logging.ERROR)

# Функція для завантаження конфігурації
def load_config(config_path="ray_training_config.yaml"):
    """Завантажує та повертає конфігурацію з файлу."""
    if not os.path.exists(config_path):
        print(f"Error: Configuration file {config_path} not found")
        return None
        
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None

# Ray автоматично ініціалізується при запуску завдання на кластері
# НЕ потрібно викликати ray.init() тут

# Визначаємо Ray актор з вимогою GPU
@ray.remote(num_gpus=1)
class GPUTrainer:
    def __init__(self):
        # Отримуємо контекст GitHub з середовища, якщо запускається як GitHub Action
        self.is_github_action = os.environ.get("IS_GITHUB_ACTION", "").lower() in ("1", "true", "yes")
        
        if self.is_github_action:
            self.github_run_id = os.getenv("GITHUB_RUN_ID", "")
            self.github_sha = os.getenv("GITHUB_SHA", "")
            self.github_repo = os.getenv("GITHUB_REPOSITORY", "")
            print(f"Running as GitHub Action: Run ID: {self.github_run_id}, SHA: {self.github_sha}")
        
        # Перевіряємо, чи потрібно встановлювати залежності
        self.config_file = self._find_config_file(["ray_training_config.yaml"])
        self.config = None
        
        if self.config_file:
            self.config = load_config(self.config_file)
            if self.config and self.config["auto_install_deps"]:
                self._install_dependencies()
            else:
                print(f"Skipping dependencies installation (disabled in config)")
        else:
            # Файл конфігурації не знайдено, встановлюємо залежності за замовчуванням
            self._install_dependencies()
        
        # Запускаємо перевірку GPU
        self.gpu_info = self._check_gpu()
        print(f"GPU status: {self.gpu_info.get('cuda_available', False)}")

    def _find_config_file(self, config_files):
        """Знаходить перший існуючий файл конфігурації зі списку"""
        for config_file in config_files:
            if os.path.exists(config_file):
                print(f"Using configuration file: {config_file}")
                return config_file
        print("Warning: No configuration file found")
        return None
    
    def _install_dependencies(self):
        """Встановлює необхідні залежності з файлу requirements.txt"""
        try:
            print("Installing dependencies from requirements.txt...")
            # Використовуємо користувацький шлях Python, якщо вказано
            python_exec = os.environ.get("RAY_PYTHON_PATH", sys.executable)
            
            # Перевіряємо, чи існує requirements.txt
            if not os.path.exists("requirements.txt"):
                print("Warning: requirements.txt not found, skipping dependencies installation")
                return
            
            # Встановлюємо залежності з requirements.txt
            try:
                print("Running pip install -r requirements.txt...")
                subprocess.check_call([
                    python_exec, "-m", "pip", "install", "-r", "requirements.txt"
                ])
                print("Dependencies installation completed successfully")
            except Exception as e:
                print(f"Error installing dependencies: {e}")
                
        except Exception as e:
            print(f"Error in dependencies installation: {e}")
    
    def _check_gpu(self):
        """Перевіряє доступність GPU"""
        try:
            # Використовуємо користувацький шлях Python, якщо вказано
            python_exec = os.environ.get("RAY_PYTHON_PATH", sys.executable)
            subprocess.run([python_exec, "check_gpu.py"], check=False)
            
            if os.path.exists("gpu_check_result.json"):
                with open("gpu_check_result.json", "r") as f:
                    return json.load(f)
            return {"cuda_available": False}
        except Exception as e:
            print(f"Error checking GPU: {e}")
            return {"cuda_available": False}
    
    def run_training(self, config_file=None):
        """Запускає тренування YOLO з параметрами конфігурації"""
        try:
            # Використовуємо наданий config_file або той, що знайдений під час ініціалізації
            if config_file:
                config = load_config(config_file)
            else:
                config = self.config
            
            # Перевіряємо, чи маємо дійсну конфігурацію
            if not config:
                return {"status": "error", "error": f"Config file not found or invalid"}
            
            # Налаштовуємо інтеграцію W&B, якщо існує API ключ
            wandb_api_key = os.environ.get("WANDB_API_KEY", "")
            if wandb_api_key:
                os.system("yolo settings wandb=True")
                
                # Встановлюємо проєкт W&B з конфігурації
                wandb_project = config["wandb_project"]
                os.environ["WANDB_PROJECT"] = wandb_project
                print(f"Using W&B project: {wandb_project}")
            
            # Вибираємо пристрій (GPU або CPU)
            device = "0" if self.gpu_info.get("cuda_available", False) else "cpu"
            
            # Отримуємо параметри тренування з конфігурації
            epochs = config["epochs"]
            batch_size = config["batch_size"] 
            img_size = config["img_size"]
            model = config["model_type"]
            data = config["dataset"]
            project = config["wandb_project"]
            
            # Визначаємо назву запуску
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            base_name = config["run_name"] if "run_name" in config else "yolo-training"
            
            if self.is_github_action:
                # Для GitHub Actions додаємо інформацію про коміт та запуск
                if self.github_run_id:
                    github_sha_short = self.github_sha[:7] if self.github_sha else ""
                    name = f"{base_name}-{timestamp}-gh{self.github_run_id}-{github_sha_short}"
                else:
                    name = f"{base_name}-{timestamp}-github"
            else:
                # Для звичайних запусків використовуємо той самий формат з датою та часом
                name = f"{base_name}-{timestamp}"
            
            # Використовуємо користувацький шлях Python, якщо вказано
            python_exec = os.environ.get("RAY_PYTHON_PATH", sys.executable)
            
            cmd = [
                python_exec, "train_yolo.py",
                "--model", model,
                "--data", data,
                "--epochs", str(epochs),
                "--batch-size", str(batch_size),
                "--img-size", str(img_size),
                "--device", device,
                "--project", project,
                "--name", name
            ]
            
            print(f"Starting training: {model} on {device}")
            print(f"Command: {' '.join(cmd)}")
            
            # Запускаємо процес тренування
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=dict(os.environ, PYTHONUNBUFFERED='1')
            )
            
            # Захоплюємо вивід процесу
            wandb_url = None
            
            # Обробляємо потік виводу
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    print(line.rstrip())
                    
                    # Захоплюємо URL W&B, якщо присутній
                    if "View run at" in line and "wandb.ai" in line:
                        try:
                            wandb_url = line.split("View run at")[1].strip()
                            with open('wandb_run_url.txt', 'w') as f:
                                f.write(wandb_url)
                        except:
                            pass
            
            # Отримуємо результат процесу
            returncode = process.poll()
            success = returncode == 0
            
            # Створюємо об'єкт результату
            result = {
                "status": "success" if success else "error",
                "returncode": returncode,
                "device_used": device,
                "wandb_url": wandb_url,
                "completed_at": datetime.now().isoformat()
            }
            
            # Додаємо інформацію GitHub, якщо доступна
            if self.is_github_action:
                result.update({
                    "github_run_id": self.github_run_id,
                    "github_sha": self.github_sha,
                    "github_repository": self.github_repo,
                    "run_name": name,
                    "gpu_available": self.gpu_info.get("cuda_available", False)
                })
            
            # Зберігаємо результат
            with open("training_result.json", "w") as f:
                json.dump(result, f, indent=2)
            
            # Виводимо результат JSON для парсингу GitHub Actions
            if self.is_github_action:
                print(json.dumps(result))
                
            return result
            
        except Exception as e:
            error_result = {
                "status": "error",
                "error_message": str(e),
                "device_used": "unknown",
                "completed_at": datetime.now().isoformat()
            }
            
            # Додаємо інформацію GitHub, якщо доступна
            if self.is_github_action:
                error_result.update({
                    "github_run_id": self.github_run_id if hasattr(self, 'github_run_id') else "",
                    "github_sha": self.github_sha if hasattr(self, 'github_sha') else "",
                    "github_repository": self.github_repo if hasattr(self, 'github_repo') else ""
                })
            
            # Зберігаємо результат помилки
            with open("training_result.json", "w") as f:
                json.dump(error_result, f, indent=2)
            
            # Виводимо результат помилки JSON для парсингу GitHub Actions
            if hasattr(self, 'is_github_action') and self.is_github_action:
                print(json.dumps(error_result))
                
            return error_result

def main():
    """
    Головна функція для запуску тренування.
    Визначає, чи запускається як частина завдання Ray або безпосередньо.
    """
    # Перевіряємо, чи скрипт запускається як частина завдання Ray
    is_ray_job = "RAY_JOB_ID" in os.environ
    
    if is_ray_job:
        # Запускається всередині завдання Ray, стартуємо безпосередньо без Actor
        print("Running as part of Ray job")
        trainer = GPUTrainer.__new__(GPUTrainer)
        trainer.__init__()
        result = trainer.run_training()
    else:
        # Запускається безпосередньо, використовуємо Ray API
        print("Running as standalone script, initializing Ray")
        ray.init(ignore_reinit_error=True)
        trainer = GPUTrainer.remote()
        result = ray.get(trainer.run_training.remote())
    
    print(f"Training completed: {result['status']}")
    return result

if __name__ == "__main__":
    main() 
# Grounding DINO Zero-Shot Detection Pipeline

Нотебук для **безклассовой (zero-shot)** детекции объектов с помощью [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO),  
сохраняющий **сырые текстовые предсказания модели** (как есть) в `.txt` файлы и визуализируя результаты на изображениях.

> ✅ Подходит для:  
> - сбора разметки без предопределённых классов,  
> - анализа фраз и неопределённости (`logit`),  
> - формирования пулов сложных/аномальных примеров,  
> - последующей фильтрации/кластеризации фраз (например, `"person"`, `"man"`, `"woman"` → `"person"`).

---

## 🔧 Установка

### 1. Клонируйте репозиторий Grounding DINO
```bash
# 1. Установка пакетов
python3 -m pip install --upgrade pip
pip install transformers==4.21.0
pip install opencv-python pillow matplotlib timm

git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .

# 2. Скачивание весов
mkdir -p weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O weights/groundingdino_swint_ogc.pth  

Поддерживаемые чекпоинты:
    groundingdino_swint_ogc.pth (Swim-T, ONNX/GPU-friendly)  
    groundingdino_swinb_cogcoor.pth (Swin-B, выше точность)  
```

## 🚀 Исходные данные
INPUT_DIR - папка с ихображениями  
CONFIG - конфигурация модели  
CHECKPOINT - чекпоинт модели  
BOX_THRESHOLD - threshold для боксов  
TEXT_THRESHOLD - threshold для текста  
PROMT - промпт для модели  

## 🏆  Результат работы
LABDELS_DIR - папка с метками (в формете yolo)  
ANNOTATION_DIR - папка с изображениями с боксами и текстом  

from ultralytics import YOLO

# YOLOv8n modelini eğit
model = YOLO('yolov8n.pt')

# Eğitim parametreleri ve gelişmiş augmentation
# Transfer learning/fine-tuning: İlk 10 katmanı dondur (sadece son katmanlar eğitilecek)
model.train(
    data='data.yaml',   # Veri kümesi konfigürasyon dosyası
    epochs=100,         # Eğitim epoch sayısı
    imgsz=640,          # Görüntü boyutu
    batch=16,           # Batch size
    hsv_h=0.015,        # Renk tonu değişimi
    hsv_s=0.7,          # Doygunluk değişimi
    hsv_v=0.4,          # Parlaklık değişimi
    degrees=0.2,        # Dönme
    translate=0.1,      # Çevirme
    scale=0.5,          # Ölçekleme
    shear=0.01,         # Kaydırma
    perspective=0.0,    # Perspektif
    flipud=0.0,         # Dikey çevirme
    fliplr=0.5,         # Yatay çevirme
    mosaic=1.0,         # Mosaic augmentation
    mixup=0.2,          # Mixup augmentation
    freeze=10           # İlk 10 katmanı dondur
)

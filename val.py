from ultralytics import YOLO

# Eğitilmiş modeli yükle (ör: 'runs/detect/train/weights/best.pt' veya başka bir model dosyası)
model = YOLO('yolov8n.pt')

# Validasyon (değerlendirme) yap
metrics = model.val(
    data='data.yaml',   # Veri kümesi konfigürasyon dosyası
    imgsz=640,          # Görüntü boyutu
    batch=16            # Batch size
)

# Sonuçları yazdır
print(metrics)
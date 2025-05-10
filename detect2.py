from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# 1. Modeli yükle (önceden eğitilmiş YOLOv8n modeli)
model = YOLO("yolov8n.pt")  # 'n' = nano, en hafif model

# 2. Görüntüyü oku
img_path = "test2.png"
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 3. Tahmin yap (yaygın parametrelerle)
results = model(
    img_rgb,
    conf=0.5,           # Güven eşiği
    iou=0.5,            # IoU eşiği (NMS için)
    classes=[15, 16],   # Sadece kedi (15) ve köpek (16) tespit et
    max_det=5,          # Maksimum 5 nesne tespit et
    agnostic_nms=True,  # Sınıf agnostik NMS uygula
    device='cpu'        # CPU'da çalıştır
)

# 3.1 Sonuçları incele
print("Boxes:", results[0].boxes)
print("Probs:", results[0].probs)
print("Classes:", results[0].boxes.cls)
print("Confidences:", results[0].boxes.conf)

# 4. Sonuçları çiz
results[0].plot()
plt.imshow(results[0].plot())  # Tahminleri göster
plt.axis('off')
plt.show()

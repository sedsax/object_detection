from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# 1. Modeli yükle (önceden eğitilmiş YOLOv8n modeli)
model = YOLO("yolov8n.pt")  # 'n' = nano, en hafif model

# 2. Görüntüyü oku
img_path = "test2.png"
img = cv2.imread(img_path)

# img = cv2.GaussianBlur(img, (3,3), 0)  # Gürültüyü azalt
# img = cv2.convertScaleAbs(img, alpha=1.1, beta=3)  # Kontrastı artır

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 3. Tahmin yap
results = model(img_rgb, conf=0.3, imgsz=1024)

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

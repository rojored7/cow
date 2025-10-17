import cv2
import os
from tqdm import tqdm

# Configuración
VIDEO_PATH = "video3.avi"  
FRAMES_DIR = "frames_video3"

# Crear carpeta de salida si no existe
os.makedirs(FRAMES_DIR, exist_ok=True)

# Abrir video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"[ERROR] No se pudo abrir el video: {VIDEO_PATH}")
    exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_num = 0

print(f"[INFO] Extrayendo {total_frames} frames...")

for _ in tqdm(range(total_frames)):
    ret, frame = cap.read()
    if not ret:
        break
    filename = os.path.join(FRAMES_DIR, f"frame_{frame_num:05d}.jpg")
    cv2.imwrite(filename, frame)
    frame_num += 1

cap.release()
print(f"[INFO] Extracción finalizada: {frame_num} frames guardados en '{FRAMES_DIR}'")

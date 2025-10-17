#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tracking de vacas con YOLO + Norfair 2.3.x (versión final con 4 acciones y máximo 2 sujetos)
---------------------------------------------------------------------------------------------
- Mantiene SIEMPRE dos IDs lógicos: Sujeto 1 y Sujeto 2. Nunca habrá más.
- Mapea cualquier ID interno de Norfair a {1,2} usando IoU con el estado previo.
- Soporta exactamente las 4 acciones del modelo: ['Bebiendo','Comiendo','Echada','Pie'].
- Dibuja recuadros y texto con la acción correcta. Exporta CSV con id=1/2, no el id de Norfair.

Requisitos:
  ultralytics, norfair>=2.3.0, opencv-python, numpy, pandas

Ejemplo:
  python track_vacas.py --model "train copy/weights/best.pt" --input video2.avi --output output_tracked7.avi --csv tracking_vacas.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
from ultralytics import YOLO
from norfair import Detection, Tracker, draw_points


# ---------------- CONFIG ----------------
MODEL_PATH = "train copy/weights/best.pt"
VIDEO_INPUT = "video2.avi"
VIDEO_OUTPUT = "output_tracked_final.avi"
CSV_OUTPUT = "tracking_vacas.csv"

CONF_THRESH = 0.60
MIN_AREA_RATIO = 0.010    # porcentaje del área del frame
MAX_VACAS = 2             # límite físico: 2
DIST_THRESHOLD = 150.0    # para Norfair (euclidean), alto para tolerar variaciones

ACCIONES_VALIDAS = ['Bebiendo', 'Comiendo', 'Echada', 'Pie']

# Colores por acción (en BGR)
COLOR_ACCION = {
    'Comiendo': (0, 200, 0),   # verde
    'Bebiendo': (0, 0, 255),   # rojo
    'Echada':   (0, 165, 255), # naranja
    'Pie':      (0, 255, 255), # amarillo
}


# ---------------- UTILIDADES ----------------
def safe_int(x): 
    return max(0, int(round(float(x))))

def iou(b1, b2):
    """ IoU entre dos cajas (x1,y1,x2,y2). """
    xA = max(b1[0], b2[0]); yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2]); yB = min(b1[3], b2[3])
    inter = max(0.0, xB-xA) * max(0.0, yB-yA)
    area1 = max(0.0, (b1[2]-b1[0])) * max(0.0, (b1[3]-b1[1]))
    area2 = max(0.0, (b2[2]-b2[0])) * max(0.0, (b2[3]-b2[1]))
    union = area1 + area2 - inter
    return inter/union if union > 0 else 0.0


# ---------------- FILTRO DE DETECCIONES ----------------
def filter_boxes(results, frame_shape):
    """ Devuelve a lo sumo 2 detecciones válidas, ordenadas por área desc:
        [(bbox, conf, label, area), ...] 
        bbox = (x1,y1,x2,y2); label es una de ACCIONES_VALIDAS. 
    """
    fh, fw = frame_shape[:2]
    min_area = MIN_AREA_RATIO * (fw * fh)
    valid = []

    if results is None or not hasattr(results, "boxes"):
        return []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        if conf < CONF_THRESH:
            continue
        area = (x2 - x1) * (y2 - y1)
        if area < min_area:
            continue
        cls = int(box.cls[0])
        label = results.names.get(cls, str(cls))
        if label not in ACCIONES_VALIDAS:
            continue
        valid.append(((x1, y1, x2, y2), conf, label, area))

    valid.sort(key=lambda v: v[3], reverse=True)
    return valid[:MAX_VACAS]


# ---------------- TRACKER NORFAIR ----------------
def make_tracker():
    # Norfair 2.3.x acepta funciones vectorizadas por nombre:
    # Usamos "euclidean" para evitar warnings y mantener compatibilidad.
    return Tracker(distance_function="euclidean", distance_threshold=DIST_THRESHOLD)


# ---------------- MAPEADOR DE IDs A SUJETO 1/2 ----------------
class SubjectMapper:
    """
    Mantiene dos sujetos lógicos {1,2} con su última bbox y acción.
    En cada frame, asigna las detecciones actuales (máximo 2) a sujeto1/sujeto2
    maximizando IoU con el estado previo. Nunca crea un tercero.
    """
    def __init__(self):
        # estado: {1: {"bbox": tuple or None, "accion": str or "Desconocido"}, 2: {...}}
        self.state = {
            1: {"bbox": None, "accion": "Desconocido"},
            2: {"bbox": None, "accion": "Desconocido"},
        }
        self.initialized = False

    def assign(self, dets):
        """
        dets: lista de dicts [{ "bbox":(x1,y1,x2,y2), "accion":label }, ...] len 0..2
        Devuelve lista de asignaciones [{ "subject_id":1/2, "bbox":..., "accion":...}, ...]
        """
        assignments = []

        if not dets:
            # no hay detecciones: no cambiamos nada (no emitimos filas nuevas)
            return assignments

        if not self.initialized:
            # primer frame con detecciones: ordenar por área y mapear al 1 y 2
            dets_sorted = sorted(dets, key=lambda d: (d["bbox"][2]-d["bbox"][0])*(d["bbox"][3]-d["bbox"][1]), reverse=True)
            if len(dets_sorted) >= 1:
                self.state[1]["bbox"] = dets_sorted[0]["bbox"]
                self.state[1]["accion"] = dets_sorted[0]["accion"]
                assignments.append({"subject_id": 1, **dets_sorted[0]})
            if len(dets_sorted) >= 2:
                self.state[2]["bbox"] = dets_sorted[1]["bbox"]
                self.state[2]["accion"] = dets_sorted[1]["accion"]
                assignments.append({"subject_id": 2, **dets_sorted[1]})
            self.initialized = True
            return assignments

        # Si ya está inicializado: asignar por máxima IoU a los dos sujetos.
        # Construimos matriz IoU entre cada det y cada sujeto {1,2}
        scores = []
        for d in dets:
            s1 = iou(d["bbox"], self.state[1]["bbox"]) if self.state[1]["bbox"] is not None else 0.0
            s2 = iou(d["bbox"], self.state[2]["bbox"]) if self.state[2]["bbox"] is not None else 0.0
            scores.append((s1, s2))

        # Greedy: asignar primero la mejor coincidencia global, luego la restante.
        taken_subjects = set()
        taken_dets = set()

        # flateamos todas las combinaciones (det_i -> subject_j) con su IoU
        combos = []
        for i, (s1, s2) in enumerate(scores):
            combos.append((i, 1, s1))
            combos.append((i, 2, s2))
        # ordenar por IoU desc
        combos.sort(key=lambda t: t[2], reverse=True)

        for det_idx, subj_id, _score in combos:
            if det_idx in taken_dets or subj_id in taken_subjects:
                continue
            # asignar
            d = dets[det_idx]
            self.state[subj_id]["bbox"] = d["bbox"]
            self.state[subj_id]["accion"] = d["accion"]
            assignments.append({"subject_id": subj_id, **d})
            taken_dets.add(det_idx)
            taken_subjects.add(subj_id)
            if len(assignments) == min(2, len(dets)):
                break

        # Garantía: nunca habrá más de 2 asignaciones
        return assignments


# ---------------- FUNCIÓN PRINCIPAL ----------------
def run():
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Modelo no encontrado: {MODEL_PATH}")
    if not os.path.exists(VIDEO_INPUT):
        sys.exit(f"Video no encontrado: {VIDEO_INPUT}")

    model = YOLO(MODEL_PATH)
    tracker = make_tracker()
    mapper = SubjectMapper()

    cap = cv2.VideoCapture(VIDEO_INPUT)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(VIDEO_OUTPUT, cv2.VideoWriter_fourcc(*'XVID'), fps, (W, H))

    csv_rows = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # YOLO
        yres = model(frame, conf=CONF_THRESH, verbose=False)[0]
        filtered = filter_boxes(yres, frame.shape)  # máx 2 detecciones

        # A Norfair solo le pasamos como mucho 2 detecciones de interés
        detections = []
        for (x1, y1, x2, y2), conf, label, _area in filtered:
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            detections.append(
                Detection(
                    points=np.array([[cx, cy]], dtype=np.float32),
                    scores=np.array([conf], dtype=np.float32),
                    data={"bbox": (x1, y1, x2, y2), "label": label}
                )
            )

        # Tracking Norfair
        tracked = tracker.update(detections=detections)

        # Construimos las detecciones “reales” del frame desde Norfair (al máximo 2)
        # Si Norfair no asocia, usamos las bbox de detections directamente.
        frame_dets = []
        for obj in tracked:
            det = getattr(obj, "last_detection", None)
            if det is None or det.data is None:
                continue
            x1, y1, x2, y2 = det.data["bbox"]
            label = det.data["label"]
            frame_dets.append({"bbox": (x1, y1, x2, y2), "accion": label})

        # Si Norfair devolvió menos de las filtradas, complementamos con las filtradas
        if len(frame_dets) < len(filtered):
            for (x1, y1, x2, y2), _conf, label, _ in filtered:
                cand = {"bbox": (x1, y1, x2, y2), "accion": label}
                # evitar duplicar si ya está esa bbox por IoU alto
                if all(iou(cand["bbox"], f["bbox"]) < 0.9 for f in frame_dets):
                    frame_dets.append(cand)

        # Limitar a dos
        if len(frame_dets) > MAX_VACAS:
            # ordena por área y corta
            frame_dets = sorted(frame_dets, key=lambda d: (d["bbox"][2]-d["bbox"][0])*(d["bbox"][3]-d["bbox"][1]), reverse=True)[:MAX_VACAS]

        # Asignar SIEMPRE a Sujeto 1 y Sujeto 2 (y solo esos)
        assigned = mapper.assign(frame_dets)  # lista de dicts con subject_id, bbox, accion

        # Dibujar y loguear SOLO sujetos asignados (máximo 2)
        for item in assigned:
            sid = item["subject_id"]
            x1, y1, x2, y2 = item["bbox"]
            accion = item["accion"]

            color = COLOR_ACCION.get(accion, (255, 255, 255))
            xi, yi, xj, yj = safe_int(x1), safe_int(y1), safe_int(x2), safe_int(y2)

            cv2.rectangle(frame, (xi, yi), (xj, yj), color, 2)
            cv2.putText(frame, f"Sujeto {sid} | {accion}", (xi, max(0, yi - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # CSV con id fijo 1/2, no el id de Norfair
            csv_rows.append([frame_idx, sid, x1, y1, x2, y2, accion])

        # Trayectoria opcional (dibujamos puntos, da igual el ID interno)
        if tracked:
            draw_points(frame, tracked, color=(255, 255, 255), radius=2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    pd.DataFrame(csv_rows, columns=["frame", "id", "x1", "y1", "x2", "y2", "accion"]).to_csv(CSV_OUTPUT, index=False)
    print(f"Tracking completado.\nVideo: {VIDEO_OUTPUT}\nCSV: {CSV_OUTPUT}")


# ---------------- ENTRY ----------------
if __name__ == "__main__":
    run()

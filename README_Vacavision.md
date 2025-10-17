# Vacavision – Detección y Tracking de Vacas 🐄

## Descripción
Pipeline completo de detección de acciones de vacas (`Comiendo`, `Bebiendo`, `Echada`, `Pie`) usando **YOLOv8 + Norfair**, más un panel interactivo de análisis en **Streamlit**.

Este proyecto permite detectar, seguir y analizar el comportamiento de hasta dos vacas en un corral, generando un archivo CSV con datos frame a frame y un video con IDs persistentes.

---

## Instrucciones rápidas

1. **Clonar el repositorio o descomprimir la carpeta:**
   ```bash
   git clone https://github.com/turepo/vacavision.git
   cd vacavision
   ```

2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Colocar el modelo YOLO entrenado** en:
   ```
   train copy/weights/best.pt
   ```

4. **Ejecutar el tracking:**
   ```bash
   python src/track_vacas.py
   ```

5. **Resultados:**
   - Video anotado → `outputs/output_tracked_final.avi`
   - Datos CSV → `outputs/tracking_vacas.csv`

6. **Ejecutar el dashboard interactivo:**
   ```bash
   streamlit run dashboard/dashboard_vacas.py
   ```

7. **Cargar el CSV generado** y explorar:
   - Gráficos por acción (barras, tortas, evolución temporal)
   - Porcentajes de tiempo por acción
   - Filtros por sujeto o rango de frames
   - Exportación a Excel

---

## Estructura del proyecto

```
vacavision/
│
├── README.md
├── requirements.txt
├── SHORT_REPORT.md
│
├── src/
│   ├── track_vacas.py
│   └── extraer_frames.py
│
├── outputs/
│   ├── tracking_vacas.csv
│   ├── output_tracked_final.avi
│   ├── demo.mp4
│   └── resumen_grafico.png
│
└── dashboard/
    └── dashboard_vacas.py
```

---

## Estructura de salida
| Archivo | Descripción |
|----------|--------------|
| `tracking_vacas.csv` | Datos frame a frame: `frame, id, x1, y1, x2, y2, acción, id_switched` |
| `output_tracked_final.avi` | Video con bounding boxes, IDs y acciones |
| `demo.mp4` | Recorte corto del video para presentación |
| `resumen_grafico.png` | Visualización resumen de acciones |

---

## Requisitos
- Python 3.10+
- GPU opcional (CUDA recomendado)
- YOLOv8 (Ultralytics)
- Norfair 2.3.x
- Streamlit 1.38+
- Plotly 5.22+

---

## Dependencias principales
```txt
ultralytics>=8.2.0
norfair>=2.3.0
opencv-python>=4.11.0
pandas>=2.2.0
numpy>=1.24.0
tqdm>=4.66.0
plotly>=5.22.0
streamlit>=1.38.0
openpyxl>=3.1.0
```

---

## Créditos
Desarrollado por **Rojored96** como parte del sistema de análisis de comportamiento animal basado en visión artificial.

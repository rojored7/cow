#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard de análisis de vacas - Streamlit + Plotly
---------------------------------------------------
Permite cargar el archivo tracking_vacas.csv y generar análisis visuales:
- Filtrado por sujeto, acción y rango de frames
- Gráficos de barras, tortas y líneas
- Cálculo de porcentajes y duración total por acción
- Exportación a Excel

Requisitos:
  pip install streamlit pandas plotly openpyxl
"""

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Análisis de Vacas", layout="wide")

st.title("🐄 Dashboard de Análisis de Actividad de Vacas")
st.markdown("Este panel permite analizar las acciones detectadas por el modelo YOLO.")

# ---------------- CARGA DE DATOS ----------------
uploaded_file = st.file_uploader("📁 Cargar archivo CSV de tracking", type=["csv", "xlsx"])

if uploaded_file:
    # Detectar tipo de archivo
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success(f"Archivo cargado: {uploaded_file.name}")

    # ---------------- LIMPIEZA ----------------
    df.columns = [c.strip().lower() for c in df.columns]

    # Normalizar nombres esperados
    expected_cols = {"frame", "id", "x1", "y1", "x2", "y2", "accion"}
    if not expected_cols.issubset(df.columns):
        st.error("El archivo no tiene las columnas esperadas. Revisa que provenga del tracking.")
        st.stop()

    # Normalizar texto
    df["accion"] = df["accion"].str.capitalize()
    df["id"] = df["id"].astype(int)

    # Crear columna de duración en segundos si FPS disponible
    fps = st.number_input("FPS del video (para calcular tiempo):", value=25, min_value=1)
    df["tiempo_seg"] = df["frame"] / fps

    # ---------------- FILTROS ----------------
    st.sidebar.header("Filtros")

    sujeto_sel = st.sidebar.multiselect(
        "Seleccionar sujeto:",
        sorted(df["id"].unique()),
        default=sorted(df["id"].unique())
    )

    accion_sel = st.sidebar.multiselect(
        "Seleccionar acción:",
        sorted(df["accion"].unique()),
        default=sorted(df["accion"].unique())
    )

    rango_frames = st.sidebar.slider(
        "Rango de frames:",
        int(df["frame"].min()),
        int(df["frame"].max()),
        (int(df["frame"].min()), int(df["frame"].max()))
    )

    df_filt = df[
        (df["id"].isin(sujeto_sel)) &
        (df["accion"].isin(accion_sel)) &
        (df["frame"].between(rango_frames[0], rango_frames[1]))
    ]

    st.subheader("📊 Datos filtrados")
    st.dataframe(df_filt, use_container_width=True, height=300)

    # ---------------- ESTADÍSTICAS ----------------
    st.subheader("📈 Estadísticas de resumen")

    resumen = (
        df_filt.groupby(["id", "accion"])
        .agg(
            frames=("frame", "count"),
            duracion_seg=("tiempo_seg", lambda x: x.max() - x.min())
        )
        .reset_index()
    )
    resumen["porcentaje"] = (resumen["frames"] / resumen["frames"].sum() * 100).round(2)

    st.write(resumen)

    # ---------------- GRAFICOS ----------------
    st.subheader("📊 Gráfico de distribución por acción")

    col1, col2 = st.columns(2)

    with col1:
        fig_bar = px.bar(
            resumen,
            x="accion",
            y="frames",
            color="id",
            barmode="group",
            text="frames",
            title="Cantidad de frames por acción y sujeto"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        fig_pie = px.pie(
            resumen,
            names="accion",
            values="frames",
            color="accion",
            title="Proporción de tiempo por acción"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ---------------- TIEMPO EN EL VIDEO ----------------
    st.subheader("⏱️ Evolución temporal")

    fig_line = px.line(
        df_filt,
        x="tiempo_seg",
        y="accion",
        color="id",
        markers=True,
        title="Evolución de acciones a lo largo del tiempo"
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # ---------------- EXPORTACIÓN ----------------
    st.subheader("📤 Exportar datos filtrados")

    def to_excel(dataframe):
        from io import BytesIO
        output = BytesIO()
        dataframe.to_excel(output, index=False)
        return output.getvalue()

    excel_data = to_excel(df_filt)
    st.download_button(
        label="Descargar Excel filtrado",
        data=excel_data,
        file_name="analisis_vacas_filtrado.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("Cargá el archivo CSV o Excel para comenzar el análisis.")

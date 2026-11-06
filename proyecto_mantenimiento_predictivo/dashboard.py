"""Dashboard interactivo: ejecutar con streamlit run dashboard.py"""
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
st.set_page_config(page_title="NASA FD001 | Mantenimiento predictivo", layout="wide")
st.title("Mantenimiento predictivo de motores turbofán")
st.caption("NASA C-MAPSS FD001 · Exploración de sensores y estimación de RUL")
f = DATA / "train_fd001_processed.csv"
if not f.exists():
    st.warning("Aún no hay datos procesados. Coloca los archivos NASA en data/raw/ y ejecuta: python src/pipeline.py")
    st.stop()
df = pd.read_csv(f)
engines = sorted(df.unit_number.unique())
selected = st.sidebar.selectbox("Motor", ["Todos"] + engines)
sensor_cols = [c for c in df if c.startswith("sensor_")]
sensor = st.sidebar.selectbox("Sensor", sensor_cols, index=0)
view = df if selected == "Todos" else df[df.unit_number == selected]
c1,c2,c3 = st.columns(3)
c1.metric("Motores", f"{df.unit_number.nunique():,}")
c2.metric("Registros", f"{len(df):,}")
c3.metric("Sensores disponibles", len(sensor_cols))
st.subheader("Tendencia del sensor frente a ciclos")
fig = px.line(view, x="time_in_cycles", y=sensor, color=None if selected != "Todos" else "unit_number",
              title=f"{sensor}: evolución temporal")
fig.update_layout(xaxis_title="Ciclo del motor (ciclos)", yaxis_title="Lectura del sensor (unidad del simulador)")
st.plotly_chart(fig, use_container_width=True)
st.subheader("Sensor frente a vida útil remanente")
fig2 = px.scatter(view.sample(min(len(view), 5000), random_state=42), x="RUL", y=sensor,
                  color="unit_number" if selected == "Todos" else None, opacity=.55)
fig2.update_layout(xaxis_title="RUL limitada (ciclos)", yaxis_title="Lectura del sensor (unidad del simulador)")
st.plotly_chart(fig2, use_container_width=True)
st.subheader("Distribución del sensor")
fig3 = px.histogram(view, x=sensor, nbins=40, title=f"Distribución de {sensor}")
st.plotly_chart(fig3, use_container_width=True)
st.subheader("Registros")
st.dataframe(view[["unit_number","time_in_cycles","RUL",sensor]].head(500), use_container_width=True)

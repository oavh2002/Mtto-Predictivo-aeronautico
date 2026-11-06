# Mantenimiento predictivo de motores — NASA C-MAPSS FD001

Proyecto reproducible de estimación de vida útil remanente (RUL) de motores turbofán.

## Estructura
- `data/raw/`: archivos originales NASA (`train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`).
- `data/processed/`: datos depurados generados por el pipeline.
- `notebooks/`: notebook de análisis y modelado.
- `src/`: funciones ETL, entrenamiento y evaluación.
- `models/`: modelos serializados.
- `reports/`: reporte técnico PDF y resultados.
- `dashboard.py`: tablero interactivo Streamlit.

## Instalación
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Datos
Descarga el conjunto NASA C-MAPSS y coloca los tres archivos FD001 en `data/raw/`.
No cambies sus nombres. Este repositorio no redistribuye los datos.

## Ejecución
Desde la raíz:
```bash
python src/pipeline.py
streamlit run dashboard.py
```
Abre `notebooks/proyecto_corregido.ipynb` y ejecuta todas las celdas para ver ETL, EDA, pruebas estadísticas, validación agrupada y evaluación oficial.

## Decisiones metodológicas
- Se elimina la columna de índice vacía que aparece al leer archivos de ancho fijo separado por espacios.
- Se revisan nulos, duplicados, sensores constantes y valores extremos.
- Los sensores constantes se eliminan porque no aportan información discriminante en FD001; se conserva un registro de la decisión.
- Los outliers se marcan mediante IQR y se reportan; no se borran automáticamente porque pueden representar estados reales de degradación.
- RUL de entrenamiento se limita a 125 ciclos como transformación objetivo habitual en C-MAPSS. Se conserva también RUL sin limitar para análisis.
- La validación usa `GroupKFold` agrupado por `unit_number`; nunca se divide aleatoriamente por fila.
- La evaluación final oficial agrega la RUL real del archivo `RUL_FD001.txt` a la última observación de cada motor de test.
- Los resultados numéricos se generan al ejecutar el proyecto con los archivos originales; no se incluyen métricas inventadas.

## Interpretación y límites
Los sensores C-MAPSS son variables simuladas de un motor turbofán. Sus números no deben asignarse a una estación o componente físico concreto sin consultar la documentación técnica del dataset. La asociación sensor-componente debe tratarse como hipótesis de ingeniería, no como identificación confirmada.

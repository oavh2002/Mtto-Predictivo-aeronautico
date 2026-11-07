"""Genera reports/reporte_tecnico_fd001.pdf con resultados reales del proyecto.

Uso (desde proyecto_mantenimiento_predictivo/):  python src/reporte.py
Lee: data/raw/*.txt y reports/metrics.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

AUTOR = "Completar nombre(s) del equipo"
ROOT = Path(__file__).resolve().parents[1]
RAW, REP = ROOT / "data" / "raw", ROOT / "reports"
FIG = REP / "figuras"
FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- datos
M = json.load(open(REP / "metrics.json", encoding="utf-8"))
A, V, T = M["audit"], M["validation"], M["official_test"]
feats = M["features"]
cols = (["unit_number", "time_in_cycles", "op_setting_1", "op_setting_2", "op_setting_3"]
        + [f"sensor_{i}" for i in range(1, 22)])
df = pd.read_csv(RAW / "train_FD001.txt", sep=r"\s+", header=None, names=cols)
df["RUL"] = df.groupby("unit_number")["time_in_cycles"].transform("max") - df["time_in_cycles"]
vida = df.groupby("unit_number")["time_in_cycles"].max()
df["etapa"] = pd.cut(df["RUL"], [-1, 50, 100, 1e9], labels=["critica", "intermedia", "sana"])

# ------------------------------------------------------- estadistica
rows = []
for f in feats:
    rho, pr_ = stats.spearmanr(df[f], df["RUL"])
    g = [df.loc[df["etapa"] == e, f].values for e in ["sana", "intermedia", "critica"]]
    F, pF = stats.f_oneway(*g)
    H, pH = stats.kruskal(*g)
    rows.append((f, rho, pr_, F, pF, H, pH))
est = pd.DataFrame(rows, columns=["sensor", "rho", "p_rho", "F", "p_F", "H", "p_H"])
est = est.reindex(est["rho"].abs().sort_values(ascending=False).index).reset_index(drop=True)
top4 = est["sensor"].head(4).tolist()


def save(fig, name):
    fig.tight_layout()
    path = FIG / f"{name}.png"
    fig.savefig(path, dpi=130)
    w, hh = fig.get_size_inches()
    plt.close(fig)
    return path, hh / w


# ------------------------------------------------------------ figuras
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.hist(vida, bins=20, color="#3b6ea5", edgecolor="white")
ax.axvline(vida.mean(), color="crimson", ls="--", label=f"media = {vida.mean():.0f} ciclos")
ax.set_xlabel("Vida del motor hasta la falla (ciclos de vuelo)")
ax.set_ylabel("Numero de motores")
ax.set_title("Distribucion de la vida util de los 100 motores de entrenamiento")
ax.legend()
F1 = save(fig, "fig1_vida")

fig, axs = plt.subplots(2, 2, figsize=(7, 5.2))
for ax, s in zip(axs.ravel(), top4):
    for u in [1, 2, 3, 4, 5]:
        d = df[df.unit_number == u]
        ax.plot(d["RUL"], d[s], lw=1, label=f"motor {u}")
    ax.invert_xaxis()
    ax.set_title(s, fontsize=9)
    ax.set_xlabel("RUL real (ciclos restantes)", fontsize=8)
    ax.set_ylabel("valor (escala del simulador)", fontsize=8)
axs[0, 0].legend(fontsize=6)
F2 = save(fig, "fig2_trayectorias")

fig, ax = plt.subplots(figsize=(7, 4))
e2 = est.sort_values("rho")
ax.barh(e2["sensor"], e2["rho"], color=["#b22222" if r < 0 else "#2e7d32" for r in e2["rho"]])
ax.set_xlabel("Correlacion de Spearman con la RUL")
ax.set_title("Asociacion monotona de cada sensor con la vida remanente")
F3 = save(fig, "fig3_spearman")

fig, ax = plt.subplots(figsize=(7, 3.4))
nombres = list(V.keys())
x = np.arange(len(nombres))
ax.bar(x - 0.2, [V[n]["RMSE"] for n in nombres], 0.4, label="RMSE", color="#3b6ea5")
ax.bar(x + 0.2, [V[n]["MAE"] for n in nombres], 0.4, label="MAE", color="#e0a030")
ax.set_xticks(x)
ax.set_xticklabels(nombres, fontsize=8)
ax.set_ylabel("Error (ciclos)")
ax.set_title("Comparacion de modelos en validacion (motores no vistos)")
ax.legend()
F4 = save(fig, "fig4_modelos")

F5 = None
try:
    pr = pd.read_csv(ROOT / "data" / "processed" / "predicciones_test_fd001.csv")
    print("Columnas de predicciones:", list(pr.columns))
    ctrue = next(c for c in pr.columns if any(k in c.lower() for k in ["real", "true", "verdad"]))
    cpred = next(c for c in pr.columns if "pred" in c.lower())
    err = pr[cpred] - pr[ctrue]
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.4))
    axs[0].scatter(pr[ctrue], pr[cpred], s=12, alpha=.7)
    lim = [0, max(pr[ctrue].max(), pr[cpred].max()) * 1.05]
    axs[0].plot(lim, lim, "r--", lw=1)
    axs[0].set_xlabel("RUL real (ciclos)")
    axs[0].set_ylabel("RUL predicha (ciclos)")
    axs[0].set_title("Test oficial: real vs predicho", fontsize=9)
    axs[1].hist(err, bins=20, color="#3b6ea5", edgecolor="white")
    axs[1].axvline(0, color="r", ls="--")
    axs[1].set_xlabel("Error (predicho - real), ciclos")
    axs[1].set_title("Residuos (positivo = predice vida de mas)", fontsize=9)
    F5 = save(fig, "fig5_test")
except Exception as e:  # noqa: BLE001
    print("Figura 5 omitida:", e)

# ---------------------------------------------------------------- PDF
ss = getSampleStyleSheet()
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=10, leading=14, alignment=4, spaceAfter=6)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=14, spaceBefore=10, spaceAfter=6)
CAP = ParagraphStyle("CAP", parent=B, fontSize=8, alignment=1, textColor=colors.grey)
TT = ParagraphStyle("TT", parent=ss["Title"], fontSize=20, leading=24)
S = []


def h(t): S.append(Paragraph(t, H1))
def p(t): S.append(Paragraph(t, B))
def img(fr, cap, w=15 * cm):
    path, r = fr
    S.append(Image(str(path), width=w, height=w * r))
    S.append(Paragraph(cap, CAP))
    S.append(Spacer(1, 6))
def tabla(data, widths=None):
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3b6ea5")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("FONTSIZE", (0, 0), (-1, -1), 8),
                           ("GRID", (0, 0), (-1, -1), .4, colors.grey),
                           ("ALIGN", (1, 0), (-1, -1), "CENTER")]))
    S.append(t)
    S.append(Spacer(1, 8))


n_out = sum(A["iqr_outlier_counts_train"].values())
ntr, nte = A["train"]["shape"][0], A["test"]["shape"][0]
removed = ", ".join(A["constant_sensors_removed"])

S.append(Spacer(1, 3 * cm))
S.append(Paragraph("Mantenimiento predictivo de motores turbofan:<br/>estimacion de la vida util remanente (RUL)<br/>con NASA C-MAPSS FD001", TT))
S.append(Spacer(1, 1 * cm))
p("<b>Reporte tecnico de analitica aeronautica</b><br/>Asignatura: Desarrollo de Algoritmos para la Simulacion de Sistemas<br/>"
  f"Carrera: Ingenieria en Aeronautica<br/>Autor(es): {AUTOR}<br/>Fecha: noviembre de 2026")
S.append(Spacer(1, 1 * cm))
h("Resumen")
p(f"Se desarrollo un pipeline reproducible que transforma la telemetria sintetica de 100 motores turbofan (NASA C-MAPSS, subconjunto FD001) "
  f"en una estimacion de la vida util remanente (RUL). Tras la limpieza y la eliminacion de {len(A['constant_sensors_removed'])} sensores constantes, "
  f"se compararon tres modelos con validacion agrupada por motor. El mejor, un Random Forest ajustado, obtuvo en la evaluacion oficial "
  f"un RMSE de {T['RMSE']:.2f} ciclos, un MAE de {T['MAE']:.2f} ciclos y un R2 de {T['R2']:.3f}. Los resultados se traducen en una propuesta "
  "de umbrales de alerta para programar inspecciones, y se entrega un tablero interactivo en Streamlit.")
S.append(PageBreak())

h("1. Problema y justificacion operativa")
p("El mantenimiento programado por intervalos fijos puede retirar componentes con vida util aun disponible o, en el extremo opuesto, "
  "no detectar a tiempo una degradacion acelerada. El mantenimiento predictivo (PdM) busca estimar cuantos ciclos de vuelo le quedan a un motor "
  "a partir de sus sensores, de modo que la inspeccion se programe segun su condicion real.")
p("El problema planteado es de regresion: dado el historial de sensores de un motor hasta el ciclo actual, estimar la RUL, es decir, los ciclos "
  "restantes antes de la falla. C-MAPSS es un conjunto sintetico de la NASA (Prognostics Center of Excellence) que simula la degradacion "
  "run-to-failure de motores turbofan, y es una referencia habitual para validar metodos de pronostico.")

h("2. Datos")
p(f"El subconjunto FD001 contiene una sola condicion operativa y un solo modo de falla. El conjunto de entrenamiento tiene {ntr} registros de 100 motores; "
  f"el de prueba, {nte} registros de otros 100 motores cuya RUL real se entrega en un archivo aparte (RUL_FD001.txt). "
  "Cada registro incluye el numero de motor, el ciclo, tres ajustes operacionales y 21 sensores.")
p(f"La vida de los motores de entrenamiento varia entre {vida.min()} y {vida.max()} ciclos (media {vida.mean():.0f}), lo que se aprecia en la Figura 1. "
  "Esta dispersion implica que no basta con un umbral fijo de ciclos para decidir el mantenimiento.")
img(F1, "Figura 1. Distribucion de la vida de los motores de entrenamiento.")

h("3. Metodologia de limpieza y validacion de consistencia")
p("<b>Nulos y duplicados.</b> La auditoria no encontro valores nulos ni filas duplicadas en entrenamiento ni en prueba. Aun asi, los modelos incluyen un imputador por mediana "
  "dentro del Pipeline, de modo que se ajusta solo con datos de entrenamiento y se evita fuga de informacion (data leakage).")
p(f"<b>Sensores constantes.</b> Se eliminaron {removed}, porque su valor no cambia a lo largo de la vida del motor y no aportan informacion discriminante en FD001. "
  f"Quedaron {len(feats)} sensores como variables de entrada.")
p(f"<b>Valores atipicos.</b> Se marcaron {n_out} observaciones fuera del rango intercuartilico (IQR), con mayor concentracion en sensor_9 "
  f"({A['iqr_outlier_counts_train']['sensor_9']}) y sensor_14 ({A['iqr_outlier_counts_train']['sensor_14']}). No se eliminaron: en un ensayo hasta la falla, "
  "los valores extremos suelen corresponder a los ultimos ciclos, es decir, a la degradacion real que se quiere detectar. Borrarlos eliminaria justamente la senal de interes.")
p("<b>Unidades.</b> C-MAPSS entrega ajustes y sensores en escalas propias del simulador, sin unidades SI verificadas, por lo que no se aplicaron conversiones para no inventarlas. "
  "La estandarizacion se aplica dentro de los pipelines solo cuando el modelo lo requiere.")
p("<b>Variable objetivo.</b> La RUL de entrenamiento se calcula como el ultimo ciclo del motor menos el ciclo actual, y se limita a 125 ciclos (transformacion habitual en C-MAPSS): "
  "mientras el motor esta sano, la degradacion es poco observable y distinguir 200 de 150 ciclos restantes no tiene valor operativo.")

h("4. Analisis exploratorio y estadistica inferencial")
p("La Figura 2 muestra la evolucion de los cuatro sensores con mayor asociacion a la RUL, en cinco motores. Se observan tendencias monotonas hacia la falla que se hacen "
  "mas pronunciadas en los ultimos ciclos, lo que es consistente con un proceso de degradacion progresiva.")
img(F2, f"Figura 2. Trayectorias de {', '.join(top4)} frente a la RUL real (cinco motores).", w=14 * cm)
p("Como las relaciones pueden ser no lineales, se uso la correlacion de Spearman (monotona) entre cada sensor y la RUL (Figura 3). "
  "Ademas se formularon las hipotesis: H0, la distribucion del sensor es la misma en las tres etapas de vida (sana: RUL mayor a 100 ciclos; intermedia: RUL entre 50 y 100; critica: RUL de 50 o menos); "
  "H1, al menos una etapa difiere. Se contrasto con ANOVA de un factor y con la prueba no parametrica de Kruskal-Wallis, que no exige normalidad.")
img(F3, "Figura 3. Correlacion de Spearman entre sensores y RUL.")
data = [["Sensor", "rho Spearman", "p (rho)", "F ANOVA", "p (ANOVA)", "H Kruskal", "p (Kruskal)"]]
for _, r in est.head(8).iterrows():
    data.append([r["sensor"], f"{r['rho']:.3f}", f"{r['p_rho']:.1e}", f"{r['F']:.0f}",
                 f"{r['p_F']:.1e}", f"{r['H']:.0f}", f"{r['p_H']:.1e}"])
tabla(data)
p(f"Para los 8 sensores con mayor |rho|, todos los p-valores son muy inferiores a 0.05, por lo que se rechaza H0: la distribucion de los sensores cambia significativamente "
  f"con la etapa de vida. Esto justifica usarlos como predictores. Con {ntr} observaciones, la significancia estadistica es facil de alcanzar; por eso se interpreta junto con "
  "el tamano del efecto (rho), no solo con el p-valor. Ademas, las observaciones de un mismo motor no son independientes entre si, de modo que los p-valores son orientativos.")
S.append(PageBreak())

h("5. Modelado y validacion")
p("Se compararon tres modelos: regresion lineal (referencia), HistGradientBoosting y Random Forest con busqueda de hiperparametros. "
  "La separacion entre entrenamiento y validacion es por motor (GroupShuffleSplit con 20% de motores y semilla fija 42), y la validacion cruzada usa GroupKFold por unit_number, "
  "de modo que ninguna fila de un motor de validacion aparece en el entrenamiento. Dividir al azar por filas habria filtrado informacion del mismo motor y dado metricas optimistas.")
p(f"El mejor modelo fue <b>{M['best_model']}</b> con los hiperparametros: max_depth = {M['best_params']['model__max_depth']}, "
  f"min_samples_leaf = {M['best_params']['model__min_samples_leaf']} y n_estimators = {M['best_params']['model__n_estimators']}. "
  f"Se entreno con {M['n_train_engines']} motores y se valido con {M['n_validation_engines']}.")
data = [["Modelo", "MAE", "RMSE", "R2", "NASA score"]]
for n in V:
    data.append([n, f"{V[n]['MAE']:.2f}", f"{V[n]['RMSE']:.2f}", f"{V[n]['R2']:.3f}", f"{V[n]['NASA_score']:.0f}"])
data.append(["Test oficial (RF)", f"{T['MAE']:.2f}", f"{T['RMSE']:.2f}", f"{T['R2']:.3f}", f"{T['NASA_score']:.0f}"])
tabla(data)
img(F4, "Figura 4. Error de cada modelo en validacion.")

h("6. Resultados")
p(f"En la evaluacion oficial, que usa la ultima observacion de cada uno de los 100 motores de prueba contra la RUL real de NASA, el modelo obtuvo MAE = {T['MAE']:.2f} ciclos, "
  f"RMSE = {T['RMSE']:.2f} ciclos y R2 = {T['R2']:.3f}. En validacion el RMSE fue {V[M['best_model']]['RMSE']:.2f} y el R2 {V[M['best_model']]['R2']:.3f}. "
  "Que ambos conjuntos den metricas similares indica que no hay sobreajuste evidente.")
p(f"El Random Forest mejora a la regresion lineal (RMSE {V['LinearRegression']['RMSE']:.2f}) y es practicamente equivalente a HistGradientBoosting; "
  "esta ultima diferencia es demasiado pequena para declarar un ganador claro.")
p("<b>Sobre el NASA score.</b> Esta metrica penaliza mas la sobreestimacion de la vida (predecir que queda mas vida de la real) que la subestimacion. "
  "No es comparable entre validacion y test, porque la validacion suma el error sobre todas las filas de 20 motores y el test solo sobre la ultima observacion de 100 motores. "
  "Ademas, la regresion lineal obtiene un NASA score menor que el Random Forest pese a su peor RMSE, lo que sugiere un sesgo distinto en el signo del error; esto debe verificarse con los residuos.")
if F5:
    img(F5, "Figura 5. Evaluacion oficial: RUL real frente a predicha y distribucion de residuos.")
S.append(PageBreak())

h("7. Recomendaciones operativas de ingenieria")
marg = 2 * T["RMSE"]
p("Las recomendaciones siguientes se derivan de los errores medidos. Son propuestas a validar con el area de mantenimiento, no procedimientos aprobados.")
p(f"<b>1. Alerta temprana con margen de error.</b> Como el error tipico (RMSE) es de {T['RMSE']:.0f} ciclos, conviene emitir la alerta cuando la RUL predicha caiga por debajo de la RUL "
  f"critica deseada mas un margen de aproximadamente dos RMSE (cerca de {marg:.0f} ciclos). Asi se reduce el riesgo de actuar tarde por una sobreestimacion del modelo.")
p("<b>2. Inspeccion segun condicion.</b> Programar una inspeccion boroscopica de las secciones calientes cuando la RUL predicha entre en la zona de alerta, en lugar de hacerlo a intervalo fijo, "
  "y usar el hallazgo para confirmar o descartar la prediccion.")
p("<b>3. Priorizar la flota por RUL.</b> Ordenar los motores por RUL predicha permite concentrar inspecciones y repuestos en los que estan mas cerca de la falla. "
  "El tablero entregado permite explorar esa informacion por motor.")
p(f"<b>4. Vigilar la tendencia, no el dato aislado.</b> Un valor puntual de sensor puede ser ruido. Los sensores con mayor asociacion ({', '.join(top4)}) deben seguirse como tendencia a lo largo de los ciclos.")
p("<b>5. Tratar con cautela las predicciones lejanas.</b> Al limitar la RUL a 125 ciclos, el modelo no distingue entre motores con mucha vida restante. Debe usarse para decidir cuando un motor "
  "se acerca a la falla, no para estimar su vida total.")

h("8. Limitaciones")
p("C-MAPSS es un conjunto sintetico con una sola condicion operativa y un solo modo de falla; el desempeno en motores reales, con varias condiciones y fallas, puede ser distinto. "
  "Los sensores se identifican por numero y no se asignan a un componente fisico concreto, porque esa asociacion debe tratarse como hipotesis de ingenieria y no como hecho confirmado. "
  "No se aplicaron conversiones de unidades por falta de unidades verificadas. Finalmente, se evaluo un solo conjunto de prueba, por lo que las metricas tienen incertidumbre que no se cuantifico.")

h("9. Reproducibilidad")
p("El repositorio contiene los datos originales (data/raw), el pipeline (src/pipeline.py), el notebook (notebooks/proyecto_corregido.ipynb), el modelo serializado (models), "
  "las metricas (reports/metrics.json), el tablero (dashboard.py) y el archivo requirements.txt. Se ejecuta con: python src/pipeline.py, streamlit run dashboard.py y python src/reporte.py. "
  "Las semillas aleatorias estan fijadas (random_state = 42).")


def pie(c, d):
    c.setFont("Helvetica", 8)
    c.drawCentredString(A4[0] / 2, 1.2 * cm, f"Pagina {d.page}")


SimpleDocTemplate(str(REP / "reporte_tecnico_fd001.pdf"), pagesize=A4, leftMargin=2.2 * cm,
                  rightMargin=2.2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                  title="Reporte tecnico C-MAPSS FD001").build(S, onFirstPage=pie, onLaterPages=pie)
try:
    from pypdf import PdfReader
    print("Reporte generado:", len(PdfReader(str(REP / "reporte_tecnico_fd001.pdf")).pages), "paginas")
except ImportError:
    print("Reporte generado")

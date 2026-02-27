# app.py
import pandas as pd
import numpy as np

from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 0) Cargar datos + features
# -----------------------------
df = pd.read_csv("df_limpio.csv")

# Normalizar nombres esperados (por si hay espacios raros)
for c in df.columns:
    if df[c].dtype == "object":
        df[c] = df[c].astype(str).str.strip()

# ---- Pregunta 1 feature (si no existe)
# En tu notebook P1 usas FAMI_EDU_PADRES_SUM ya hecho.
# Si no existe, no invento: se cae con error claro.
if "FAMI_EDU_PADRES_SUM" not in df.columns:
    raise ValueError("No existe FAMI_EDU_PADRES_SUM en df_limpio.csv (lo usa Pregunta 1).")

df["EDU_PADRES_CAT"] = pd.cut(
    df["FAMI_EDU_PADRES_SUM"],
    bins=[-1, 6, 12, 20],
    labels=["Baja", "Media", "Alta"]
)

# ---- Pregunta 2: asegurar binarios 0/1 si vienen como SI/NO
bin_cols = ["FAMI_TIENECOMPUTADOR", "FAMI_TIENEINTERNET", "FAMI_TIENEAUTOMOVIL", "FAMI_TIENELAVADORA"]
for c in bin_cols:
    if c in df.columns:
        s = df[c].astype(str).str.upper().str.strip().replace({"SÍ": "SI"})
        # Si ya es 0/1 numérico como string, esto lo deja bien.
        df[c] = np.where(s.isin(["SI", "1", "TRUE"]), 1,
                 np.where(s.isin(["NO", "0", "FALSE"]), 0, np.nan))

# ---- Pregunta 3 feature (si faltan)
# En tu notebook P3 construyes INDICE_ECON = auto + pc + internet + lavadora
need_for_indice = ["FAMI_TIENEAUTOMOVIL", "FAMI_TIENECOMPUTADOR", "FAMI_TIENEINTERNET", "FAMI_TIENELAVADORA"]
if all(c in df.columns for c in need_for_indice):
    df_ind = df.dropna(subset=need_for_indice).copy()
    df_ind["INDICE_ECON"] = (
        df_ind["FAMI_TIENEAUTOMOVIL"] +
        df_ind["FAMI_TIENECOMPUTADOR"] +
        df_ind["FAMI_TIENEINTERNET"] +
        df_ind["FAMI_TIENELAVADORA"]
    )
else:
    df_ind = df.copy()
    df_ind["INDICE_ECON"] = np.nan

# Para P3 necesitas EDU_MAX_HOGAR (ya venía en tu df según notebook)
if "EDU_MAX_HOGAR" not in df_ind.columns:
    # No lo invento. Prefiero error explícito.
    raise ValueError("No existe EDU_MAX_HOGAR en df_limpio.csv (lo usa Pregunta 3).")

# Score variables disponibles
score_vars = [c for c in ["PUNT_GLOBAL", "PUNT_MATEMATICAS", "PUNT_LECTURA_CRITICA", "PUNT_INGLES"] if c in df.columns]
if "PUNT_GLOBAL" not in score_vars:
    raise ValueError("No existe PUNT_GLOBAL en df_limpio.csv (es clave en P1, P2, P3).")

# Estratos
estrato_col = "FAMI_ESTRATOVIVIENDA" if "FAMI_ESTRATOVIVIENDA" in df.columns else None
if estrato_col is None:
    raise ValueError("No existe FAMI_ESTRATOVIVIENDA en df_limpio.csv (se usa como control).")

estratos = sorted(df[estrato_col].dropna().unique().tolist(), key=lambda x: str(x))

def mean_ci95(x: pd.Series):
    x = x.dropna().astype(float)
    if len(x) < 2:
        return (np.nan, np.nan, np.nan)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x))
    ci = 1.96 * se
    return (m, m - ci, m + ci)

# -----------------------------
# 1) App
# -----------------------------
app = Dash(__name__)
server = app.server

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "16px"},
    children=[
        html.H2("Tablero Saber 11 — Determinantes del desempeño (Usuario: DNP)"),
        html.Div(
            "Objetivo: visualizar evidencia descriptiva y comparativa sobre capital cultural, capital económico y brecha digital.",
            style={"marginBottom": "12px", "color": "#444"},
        ),

        dcc.Tabs(
            value="tab-p1",
            children=[
                dcc.Tab(label="Pregunta 1 — Educación padres vs desempeño", value="tab-p1"),
                dcc.Tab(label="Pregunta 2 — Brecha digital (PC/Internet) controlando estrato", value="tab-p2"),
                dcc.Tab(label="Pregunta 3 — ¿Cultural o económico explica más?", value="tab-p3"),
            ],
        ),

        html.Div(id="tab-content", style={"marginTop": "16px"})
    ]
)

# -----------------------------
# 2) Layout por tab
# -----------------------------
def layout_p1():
    return html.Div([
        html.H3("P1: Educación de los padres y desempeño, independiente del estrato"),
        html.Div([
            html.Div([
                html.Label("Variable de puntaje"),
                dcc.Dropdown(
                    id="p1-score",
                    options=[{"label": v, "value": v} for v in score_vars],
                    value="PUNT_GLOBAL",
                    clearable=False
                ),
            ], style={"flex": "1"}),

            html.Div([
                html.Label("Estratos (filtrar)"),
                dcc.Dropdown(
                    id="p1-estratos",
                    options=[{"label": str(e), "value": e} for e in estratos],
                    value=estratos, multi=True
                ),
            ], style={"flex": "2", "marginLeft": "12px"})
        ], style={"display": "flex", "gap": "12px"}),

        html.Div(id="p1-insight", style={"marginTop": "12px", "padding": "10px", "background": "#f6f6f6", "borderRadius": "10px"}),

        html.Div([
            dcc.Graph(id="p1-heatmap"),
            dcc.Graph(id="p1-lines"),
        ])
    ])

def layout_p2():
    return html.Div([
        html.H3("P2: ¿PC e Internet explican el puntaje (controlando estrato) — Cundinamarca?"),
        html.Div([
            html.Div([
                html.Label("Estrato (condicionar el análisis)"),
                dcc.Dropdown(
                    id="p2-estrato",
                    options=[{"label": str(e), "value": e} for e in estratos],
                    value=estratos[0] if len(estratos) else None,
                    clearable=False
                ),
            ], style={"flex": "1"}),

            html.Div([
                html.Label("Municipio (opcional)"),
                dcc.Dropdown(
                    id="p2-mpio",
                    options=(
                        [{"label": "Todos", "value": "__ALL__"}] +
                        ([{"label": m, "value": m} for m in sorted(df["COLE_MCPIO_UBICACION"].dropna().unique())]
                         if "COLE_MCPIO_UBICACION" in df.columns else [{"label": "No disponible", "value": "__ALL__"}])
                    ),
                    value="__ALL__",
                    clearable=False
                )
            ], style={"flex": "2", "marginLeft": "12px"})
        ], style={"display": "flex"}),

        html.Div(id="p2-insight", style={"marginTop": "12px", "padding": "10px", "background": "#f6f6f6", "borderRadius": "10px"}),

        html.Div([
            dcc.Graph(id="p2-heatmap"),
            dcc.Graph(id="p2-box"),
            dcc.Graph(id="p2-bars"),
        ])
    ])

def layout_p3():
    return html.Div([
        html.H3("P3: ¿Capital cultural o económico explica más el desempeño?"),
        html.Div([
            html.Div([
                html.Label("Puntaje objetivo"),
                dcc.Dropdown(
                    id="p3-score",
                    options=[{"label": v, "value": v} for v in score_vars],
                    value="PUNT_GLOBAL",
                    clearable=False
                )
            ], style={"flex": "1"}),

            html.Div([
                html.Label("Estrato (para heatmap cultural×estrato)"),
                dcc.Dropdown(
                    id="p3-estrato-hm",
                    options=[{"label": "Todos", "value": "__ALL__"}] + [{"label": str(e), "value": e} for e in estratos],
                    value="__ALL__",
                    clearable=False
                )
            ], style={"flex": "1", "marginLeft": "12px"})
        ], style={"display": "flex"}),

        html.Div(id="p3-insight", style={"marginTop": "12px", "padding": "10px", "background": "#f6f6f6", "borderRadius": "10px"}),

        html.Div([
            dcc.Graph(id="p3-r2"),
            dcc.Graph(id="p3-coef"),
            dcc.Graph(id="p3-heatmap"),
        ])
    ])



# Versión limpia: usa un Div wrapper
# (Si te molesta este bloque, al final te dejo versión minimal sin hacks.)

# -----------------------------
# 3) Callbacks por Tab (sin hacks)
# -----------------------------
# Re-def layout con wrapper limpio:
app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "16px"},
    children=[
        html.H2("Tablero Saber 11 — Determinantes del desempeño (Usuario: DNP)"),
        html.Div(
            "Capital cultural (educación), capital económico (estrato + bienes) y brecha digital.",
            style={"marginBottom": "12px", "color": "#444"},
        ),
        dcc.Tabs(id="tabs", value="tab-p1", children=[
            dcc.Tab(label="Pregunta 1", value="tab-p1"),
            dcc.Tab(label="Pregunta 2", value="tab-p2"),
            dcc.Tab(label="Pregunta 3", value="tab-p3"),
        ]),
        html.Div(id="tabs-content", style={"marginTop": "16px"})
    ]
)

@app.callback(Output("tabs-content", "children"), Input("tabs", "value"))
def render_tabs(tab):
    if tab == "tab-p1":
        return layout_p1()
    if tab == "tab-p2":
        return layout_p2()
    return layout_p3()

# -------- P1 callbacks --------
@app.callback(
    Output("p1-heatmap", "figure"),
    Output("p1-lines", "figure"),
    Output("p1-insight", "children"),
    Input("p1-score", "value"),
    Input("p1-estratos", "value"),
)
def update_p1(score_var, estratos_sel):
    d = df.dropna(subset=[score_var, "EDU_PADRES_CAT", estrato_col]).copy()
    d = d[d[estrato_col].isin(estratos_sel)]

    # Heatmap medias
    pivot_mean = d.pivot_table(values=score_var, index="EDU_PADRES_CAT", columns=estrato_col, aggfunc="mean")
    pivot_n = d.pivot_table(values=score_var, index="EDU_PADRES_CAT", columns=estrato_col, aggfunc="size").fillna(0).astype(int)

    hm = go.Figure(data=go.Heatmap(
        z=pivot_mean.values,
        x=[str(x) for x in pivot_mean.columns],
        y=[str(y) for y in pivot_mean.index],
        colorbar={"title": f"Media {score_var}"},
        hovertemplate="Estrato=%{x}<br>Edu=%{y}<br>Media=%{z:.1f}<extra></extra>"
    ))
    hm.update_layout(title=f"Media de {score_var} por educación padres × estrato")

    # Líneas de medias por estrato
    g = (d.groupby(["EDU_PADRES_CAT", estrato_col])[score_var]
         .apply(lambda s: pd.Series(mean_ci95(s), index=["mean","lo","hi"]))
         .reset_index())
    fig_lines = go.Figure()
    for e in sorted(g[estrato_col].unique(), key=lambda x: str(x)):
        ge = g[g[estrato_col] == e]
        fig_lines.add_trace(go.Scatter(
            x=ge["EDU_PADRES_CAT"].astype(str),
            y=ge["mean"],
            mode="lines+markers",
            name=str(e),
            error_y=dict(type="data", symmetric=False, array=(ge["hi"]-ge["mean"]), arrayminus=(ge["mean"]-ge["lo"]))
        ))
    fig_lines.update_layout(
        title=f"Media (±IC 95%) de {score_var} por educación de padres, separado por estrato",
        xaxis_title="Educación de los padres (categoría)",
        yaxis_title=f"{score_var} (media)"
    )

    # Insight: diferencia Alta vs Baja (promedio general)
    avg = d.groupby("EDU_PADRES_CAT")[score_var].mean()
    if all(k in avg.index for k in ["Alta","Baja"]):
        delta = float(avg["Alta"] - avg["Baja"])
        insight = f"En promedio, pasar de educación parental 'Baja' a 'Alta' se asocia con +{delta:.1f} puntos en {score_var} (en la muestra filtrada)."
    else:
        insight = "No hay suficientes datos para comparar 'Baja' vs 'Alta' en el filtro actual."
    return hm, fig_lines, insight

# -------- P2 callbacks --------
@app.callback(
    Output("p2-heatmap", "figure"),
    Output("p2-box", "figure"),
    Output("p2-bars", "figure"),
    Output("p2-insight", "children"),
    Input("p2-estrato", "value"),
    Input("p2-mpio", "value")
)
def update_p2(estrato_sel, mpio_sel):
    d = df.dropna(subset=["PUNT_GLOBAL", estrato_col, "FAMI_TIENECOMPUTADOR", "FAMI_TIENEINTERNET"]).copy()

    if mpio_sel != "__ALL__" and "COLE_MCPIO_UBICACION" in d.columns:
        d = d[d["COLE_MCPIO_UBICACION"] == mpio_sel]

    # Heatmap condicionado al estrato
    de = d[d[estrato_col] == estrato_sel].copy()
    hm_tab = de.groupby(["FAMI_TIENECOMPUTADOR", "FAMI_TIENEINTERNET"])["PUNT_GLOBAL"].mean().unstack()

    hm = go.Figure(data=go.Heatmap(
        z=hm_tab.values,
        x=[f"Internet={int(x)}" for x in hm_tab.columns],
        y=[f"PC={int(y)}" for y in hm_tab.index],
        colorbar={"title": "Media PUNT_GLOBAL"},
        hovertemplate="%{y}<br>%{x}<br>Media=%{z:.1f}<extra></extra>"
    ))
    hm.update_layout(title=f"PUNT_GLOBAL promedio (Estrato={estrato_sel}) — combinación PC×Internet")

    # Boxplot global por Internet
    box = px.box(d, x="FAMI_TIENEINTERNET", y="PUNT_GLOBAL",
                 points="outliers", title="Distribución PUNT_GLOBAL según Internet (0=No,1=Sí)")
    box.update_xaxes(title="Tiene Internet")

    # Barras: medias por estrato para PC/Internet (agregado)
    agg = (d.groupby([estrato_col])[["PUNT_GLOBAL"]].mean().reset_index()
           .rename(columns={"PUNT_GLOBAL":"Media"}))
    bars = px.line(agg, x=estrato_col, y="Media", markers=True,
                   title="Media de PUNT_GLOBAL por estrato (contexto del control)")
    bars.update_xaxes(title="Estrato")

    # Insight: brecha internet dentro del estrato
    if len(de) >= 10:
        m0 = de[de["FAMI_TIENEINTERNET"] == 0]["PUNT_GLOBAL"].mean()
        m1 = de[de["FAMI_TIENEINTERNET"] == 1]["PUNT_GLOBAL"].mean()
        if pd.notna(m0) and pd.notna(m1):
            insight = f"Dentro de Estrato={estrato_sel}, tener Internet se asocia con +{(m1-m0):.1f} puntos en PUNT_GLOBAL (media)."
        else:
            insight = "No hay suficiente información dentro del estrato seleccionado para estimar brecha por Internet."
    else:
        insight = "Muy pocos datos con el filtro actual para estimar brechas con confianza."

    return hm, box, bars, insight

# -------- P3 callbacks --------
@app.callback(
    Output("p3-r2", "figure"),
    Output("p3-coef", "figure"),
    Output("p3-heatmap", "figure"),
    Output("p3-insight", "children"),
    Input("p3-score", "value"),
    Input("p3-estrato-hm", "value")
)
def update_p3(score_var, estrato_hm):
    d = df_ind.dropna(subset=[score_var, "EDU_MAX_HOGAR", estrato_col, "INDICE_ECON"]).copy()

    # R2 test comparando modelos (replica tu notebook)
    X_cul = d[["EDU_MAX_HOGAR"]]
    X_eco = d[[estrato_col, "INDICE_ECON"]]
    X_all = d[["EDU_MAX_HOGAR", estrato_col, "INDICE_ECON"]]
    y = d[score_var]

    Xc_tr, Xc_te, y_tr, y_te = train_test_split(X_cul, y, test_size=0.2, random_state=42)
    Xe_tr, Xe_te, _, _ = train_test_split(X_eco, y, test_size=0.2, random_state=42)
    Xa_tr, Xa_te, _, _ = train_test_split(X_all, y, test_size=0.2, random_state=42)

    m_cul = LinearRegression().fit(Xc_tr, y_tr)
    m_eco = LinearRegression().fit(Xe_tr, y_tr)
    m_all = LinearRegression().fit(Xa_tr, y_tr)

    r2_cul = r2_score(y_te, m_cul.predict(Xc_te))
    r2_eco = r2_score(y_te, m_eco.predict(Xe_te))
    r2_all = r2_score(y_te, m_all.predict(Xa_te))

    r2_df = pd.DataFrame({
        "Modelo": ["Cultural (EDU_MAX_HOGAR)", "Económico (estrato + bienes)", "Conjunto"],
        "R2_test": [r2_cul, r2_eco, r2_all]
    })
    fig_r2 = px.bar(r2_df, x="Modelo", y="R2_test", title="Poder explicativo: R² en test por dimensión")
    fig_r2.update_xaxes(tickangle=15)

    # Coefs estandarizados (como tu notebook)
    X = d[["EDU_MAX_HOGAR", estrato_col, "INDICE_ECON"]].copy()
    yv = d[score_var].values.reshape(-1, 1)

    scX = StandardScaler()
    scY = StandardScaler()
    Xz = scX.fit_transform(X)
    yz = scY.fit_transform(yv).ravel()

    X_tr, X_te, y_tr, y_te = train_test_split(Xz, yz, test_size=0.2, random_state=42)
    m = LinearRegression().fit(X_tr, y_tr)

    coef_df = pd.DataFrame({
        "Variable": ["EDU_MAX_HOGAR", "Estrato", "INDICE_ECON"],
        "Coef_estandarizado": m.coef_
    }).sort_values("Coef_estandarizado", ascending=False)

    fig_coef = px.bar(coef_df, x="Variable", y="Coef_estandarizado",
                      title="Importancia relativa (coeficientes estandarizados) — modelo conjunto")

    # Heatmap cultural × económico (simple y muy explicativo)
    dh = d.copy()
    if estrato_hm != "__ALL__":
        dh = dh[dh[estrato_col] == estrato_hm]

    pivot = dh.pivot_table(values=score_var, index="EDU_MAX_HOGAR", columns="INDICE_ECON", aggfunc="mean")
    fig_hm = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[str(x) for x in pivot.columns],
        y=[str(y) for y in pivot.index],
        colorbar={"title": f"Media {score_var}"},
        hovertemplate="IndiceEcon=%{x}<br>EduMax=%{y}<br>Media=%{z:.1f}<extra></extra>"
    ))
    title_hm = f"{score_var} promedio: EDU_MAX_HOGAR × INDICE_ECON"
    if estrato_hm != "__ALL__":
        title_hm += f" (Estrato={estrato_hm})"
    fig_hm.update_layout(title=title_hm, xaxis_title="INDICE_ECON (0–4)", yaxis_title="EDU_MAX_HOGAR")

    # Insight: quién gana
    winner = "Cultural" if r2_cul > r2_eco else "Económico"
    insight = (
        f"Comparando R² test: Cultural={r2_cul:.3f}, Económico={r2_eco:.3f}, Conjunto={r2_all:.3f}. "
        f"En esta muestra, la dimensión con mayor poder explicativo individual es: {winner}."
    )

    return fig_r2, fig_coef, fig_hm, insight

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
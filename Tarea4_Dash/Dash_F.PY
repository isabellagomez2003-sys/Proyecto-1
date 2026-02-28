import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
from dash import dash_table
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import statsmodels.formula.api as smf  


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

#cargar datos

df = pd.read_csv("df_limpio.csv")
# ======================================================
# PREP PREGUNTA 1 (Educación padres vs puntaje, controlando estrato)
# ======================================================

cols_p1 = ["FAMI_EDU_PADRES_SUM", "PUNT_GLOBAL", "FAMI_ESTRATOVIVIENDA"]
df_p1 = df[cols_p1].dropna().copy()

# Si tu estrato quedó como int, lo convertimos a etiqueta legible
# (si ya lo manejas como string tipo "Estrato 1", esto igual funciona)
df_p1["FAMI_ESTRATOVIVIENDA"] = df_p1["FAMI_ESTRATOVIVIENDA"].astype(int)
df_p1["ESTRATO_LABEL"] = df_p1["FAMI_ESTRATOVIVIENDA"].apply(lambda x: f"Estrato {x}")

# Categorías como en tu notebook: Baja / Media / Alta
df_p1["EDU_PADRES_CAT"] = pd.cut(
    df_p1["FAMI_EDU_PADRES_SUM"],
    bins=[-1, 6, 12, 20],
    labels=["Baja", "Media", "Alta"]
)

# Orden para que salga bien en el eje
df_p1["EDU_PADRES_CAT"] = pd.Categorical(
    df_p1["EDU_PADRES_CAT"],
    categories=["Baja", "Media", "Alta"],
    ordered=True
)

estratos_disponibles_p1 = sorted(df_p1["ESTRATO_LABEL"].unique(), key=lambda s: int(s.split()[-1]))
#modelo de regresión pregunta 2


modelo = smf.ols(
    'PUNT_GLOBAL ~ FAMI_TIENECOMPUTADOR + FAMI_TIENEINTERNET + C(FAMI_ESTRATOVIVIENDA)',
    data=df
).fit()

resumen_modelo = modelo.summary().as_text()

# Quitar nulos importantes
df = df.dropna(subset=["PUNT_GLOBAL", "FAMI_ESTRATOVIVIENDA", "COLE_MCPIO_UBICACION"])

df["FAMI_ESTRATOVIVIENDA"] = df["FAMI_ESTRATOVIVIENDA"].astype(int)

#para la tabla de estadisticas
stats_df = df["PUNT_GLOBAL"].describe().reset_index()
stats_df.columns = ["Estadística", "Valor"]
stats_df["Valor"] = stats_df["Valor"].round(2)

# Opcional: redondear
stats_df["Valor"] = stats_df["Valor"].round(2)


# ==============================
# PROMEDIO POR MUNICIPIO
# ==============================

df_mcpio = (
    df.groupby("COLE_MCPIO_UBICACION")["PUNT_GLOBAL"]
    .mean()
    .reset_index()
)

df_mcpio["ubicacion"] = df_mcpio["COLE_MCPIO_UBICACION"] + ", Colombia"

# ======================================================
# PREGUNTA 3 — Helpers y validación de columnas (antes del layout)
# ======================================================

# Variables candidatas (solo se mostrarán las que existan)
P3_CULTURE_CANDIDATES = [
    ("Educación máxima del hogar", "EDU_MAX_HOGAR"),
    ("Educación madre", "FAMI_EDUCACIONMADRE"),
    ("Educación padre", "FAMI_EDUCACIONPADRE"),
]

P3_ECON_CANDIDATES = [
    ("Estrato", "FAMI_ESTRATOVIVIENDA"),
    ("Índice de bienes del hogar", "INDICE_ECON"),
]

def _p3_existing_options(candidates):
    """Devuelve options Dash solo con columnas existentes."""
    opts = []
    for label, col in candidates:
        if col in df.columns:
            opts.append({"label": label, "value": col})
    return opts

P3_CULTURE_OPTIONS = _p3_existing_options(P3_CULTURE_CANDIDATES)
P3_ECON_OPTIONS = _p3_existing_options(P3_ECON_CANDIDATES)

# Defaults razonables (si existen)
P3_DEFAULT_CULTURE = "EDU_MAX_HOGAR" if "EDU_MAX_HOGAR" in df.columns else P3_CULTURE_OPTIONS[0]["value"]
P3_DEFAULT_ECON = "FAMI_ESTRATOVIVIENDA" if "FAMI_ESTRATOVIVIENDA" in df.columns else P3_ECON_OPTIONS[0]["value"]

def _p3_make_mean_line(df_in, xvar, title, xlabel):
    g = df_in.groupby(xvar)["PUNT_GLOBAL"].mean().reset_index().sort_values(xvar)
    fig = px.line(g, x=xvar, y="PUNT_GLOBAL", markers=True, title=title)
    fig.update_layout(template="simple_white", xaxis_title=xlabel, yaxis_title="Puntaje global promedio")
    return fig

def _p3_group_edu(series):
    # Bins del notebook: Bajo (0-2), Medio (3-6), Alto (7-10)
    s = series.astype(float)
    return pd.cut(s, bins=[-1, 2, 6, 10], labels=["Bajo (0-2)", "Medio (3-6)", "Alto (7-10)"])

def _p3_group_econ(series, econ_var):
    s = series.astype(float)
    if econ_var == "FAMI_ESTRATOVIVIENDA":
        return pd.cut(s, bins=[0.5, 2.5, 4.5, 6.5], labels=["Bajo (1-2)", "Medio (3-4)", "Alto (5-6)"])
    # INDICE_ECON típico 0-4: Bajo (0-1), Medio (2-3), Alto (4+)
    return pd.cut(s, bins=[-1, 1.5, 3.5, 99], labels=["Bajo (0-1)", "Medio (2-3)", "Alto (4+)"])

def _p3_make_heatmaps(df_in, cultura_var, econ_var):
    df_aux = df_in.copy()
    df_aux["EDU_G"] = _p3_group_edu(df_aux[cultura_var])
    df_aux["ECO_G"] = _p3_group_econ(df_aux[econ_var], econ_var)

    hm_mean = df_aux.pivot_table(index="ECO_G", columns="EDU_G", values="PUNT_GLOBAL", aggfunc="mean")
    hm_n = df_aux.pivot_table(index="ECO_G", columns="EDU_G", values="PUNT_GLOBAL", aggfunc="count")

    fig_mean = px.imshow(hm_mean, text_auto=".1f", aspect="auto",
                         title="Mapa de calor (promedio): Puntaje por Economía × Cultura")
    fig_mean.update_layout(template="simple_white", xaxis_title="Cultura (grupo)", yaxis_title="Economía (grupo)")

    fig_n = px.imshow(hm_n, text_auto=True, aspect="auto",
                      title="Mapa de calor (N observaciones): tamaño muestral por celda")
    fig_n.update_layout(template="simple_white", xaxis_title="Cultura (grupo)", yaxis_title="Economía (grupo)")

    return fig_mean, fig_n

def _p3_compute_models(df_in, test_size=0.2, random_state=42):
    # Columnas “core” según tu notebook (cultura + economía + puntaje)
    required = ["PUNT_GLOBAL", "EDU_MAX_HOGAR", "FAMI_ESTRATOVIVIENDA", "INDICE_ECON"]
    dfm = df_in.dropna(subset=[c for c in required if c in df_in.columns]).copy()

    X_cul = dfm[["EDU_MAX_HOGAR"]].astype(float)
    X_eco = dfm[["FAMI_ESTRATOVIVIENDA", "INDICE_ECON"]].astype(float)
    X_all = dfm[["EDU_MAX_HOGAR", "FAMI_ESTRATOVIVIENDA", "INDICE_ECON"]].astype(float)
    y = dfm["PUNT_GLOBAL"].astype(float)

    # Split consistente (mismas filas para todos los modelos)
    idx = np.arange(len(dfm))
    idx_tr, idx_te = train_test_split(idx, test_size=test_size, random_state=random_state)

    m_cul = LinearRegression().fit(X_cul.iloc[idx_tr], y.iloc[idx_tr])
    m_eco = LinearRegression().fit(X_eco.iloc[idx_tr], y.iloc[idx_tr])
    m_all = LinearRegression().fit(X_all.iloc[idx_tr], y.iloc[idx_tr])

    r2_cul = r2_score(y.iloc[idx_te], m_cul.predict(X_cul.iloc[idx_te]))
    r2_eco = r2_score(y.iloc[idx_te], m_eco.predict(X_eco.iloc[idx_te]))
    r2_all = r2_score(y.iloc[idx_te], m_all.predict(X_all.iloc[idx_te]))

    tabla_r2 = pd.DataFrame({
        "Modelo": ["Cultural", "Económico (estrato + bienes)", "Conjunto"],
        "R2_test": [r2_cul, r2_eco, r2_all]
    }).round(4)

    delta_cul = r2_all - r2_eco
    delta_eco = r2_all - r2_cul

    r2_parcial_c = (r2_all - r2_eco) / (1 - r2_eco) if (1 - r2_eco) != 0 else np.nan
    r2_parcial_e = (r2_all - r2_cul) / (1 - r2_cul) if (1 - r2_cul) != 0 else np.nan

    tabla_marginal = pd.DataFrame({
        "Métrica": [
            "ΔR2 (Cultura agregado a Económico)",
            "ΔR2 (Economía agregado a Cultural)",
            "R2 parcial Cultura",
            "R2 parcial Economía",
        ],
        "Valor": [delta_cul, delta_eco, r2_parcial_c, r2_parcial_e]
    }).round(4)

    # Betas estandarizados (modelo conjunto)
    scX = StandardScaler()
    scY = StandardScaler()
    Xz = scX.fit_transform(X_all.values)
    yz = scY.fit_transform(y.values.reshape(-1, 1)).ravel()

    Xz_tr, Xz_te, yz_tr, yz_te = train_test_split(Xz, yz, test_size=test_size, random_state=random_state)
    mz = LinearRegression().fit(Xz_tr, yz_tr)

    tabla_beta = pd.DataFrame({
        "Variable": ["EDU_MAX_HOGAR", "FAMI_ESTRATOVIVIENDA", "INDICE_ECON"],
        "Coef_estandarizado": mz.coef_
    }).sort_values("Coef_estandarizado", ascending=False).round(4)

    # VIF (colinealidad)
    X_vif = sm.add_constant(X_all)
    tabla_vif = pd.DataFrame({
        "Variable": X_vif.columns,
        "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    }).round(3)

    # Solapamiento: cuánto EDU_MAX explica estrato (como en tu notebook)
    reg_aux = LinearRegression().fit(dfm[["EDU_MAX_HOGAR"]].astype(float), dfm["FAMI_ESTRATOVIVIENDA"].astype(float))
    r2_overlap = reg_aux.score(dfm[["EDU_MAX_HOGAR"]].astype(float), dfm["FAMI_ESTRATOVIVIENDA"].astype(float))

    return tabla_r2, tabla_marginal, tabla_beta, tabla_vif, float(r2_overlap)
# ==============================
# LAYOUT
# ==============================

app.layout = html.Div([

    html.H2(
        "Analítica de Resultados Saber 11 - Cundinamarca",
        style={'textAlign': 'center'}
    ),

    dcc.Tabs(value="tab-proyecto", children=[

        # ======================================================
        # =================== NUESTRO PROYECTO ==================
        # ======================================================

        dcc.Tab(label="Nuestro Proyecto", value="tab-proyecto", children=[

            html.Br(),
            html.H3("Nuestro Proyecto"),

            html.H4("Introducción"),
            html.P(
                "Este proyecto desarrolla un tablero de analítica de datos en Dash, desplegado en la nube, "
                "que permite al DNP analizar los determinantes del desempeño académico en las pruebas Saber 11 "
                "en el departamento de Cundinamarca."
            ),

            html.Br(),

            html.H4("Usuario final"),
            html.P([html.Strong("DNP"), " (Departamento Nacional de Planeación)"]),

            html.Br(),

            html.H4("Integrantes del grupo"),
            html.Ul([
                html.Li("Daniela Solarte"),
                html.Li("Diego Galvan"),
                html.Li("Isabella Gomez"),
            ]),

            html.Br(),

            html.H4("Objetivo"),
            html.P("El objetivo es proporcionar evidencia cuantitativa que apoye:"),
            html.Ul([
                html.Li("La formulación de políticas públicas educativas"),
                html.Li("La focalización de inversión social"),
                html.Li("El diseño de estrategias para reducir brechas educativas territoriales y socioeconómicas"),
                html.Li("La priorización de intervenciones en capital humano y transformación digital"),
            ]),

            html.Br(),

            html.H4("Dimensiones analizadas"),
            html.P("El análisis se enfoca en tres dimensiones clave:"),
            html.Ul([
                html.Li("Eduación de Padres/Capital Cultural"),
                html.Li("Capital económico del hogar, y estrato socioeconómico"),
                html.Li("Acceso a tecnología (brecha digital)"),
            ]),

            html.Br(),

            html.H4("Preguntas de negocio"),
            html.Ul([
                html.Li([
                    "¿Cómo influye el nivel educativo de los padres en el desarrollo de competencias críticas "
                    "independientemente del estrato socioeconómico? ",
                    html.Em("(Isabella)")
                ]),
                html.Li([
                    "¿En qué medida el acceso a computador e internet en el hogar explica las diferencias en el puntaje global, "
                    "una vez considerado el estrato socioeconómico? ",
                    html.Em("(Daniela)")
                ]),
                html.Li([
                    "¿Qué dimensión del capital familiar tiene mayor poder explicativo sobre el desempeño en Saber 11: cultural o económico? "
                    "Entendiendo, Cultural como la educación, y ecónomico, como diversos factores materiales, y estrato ",
                    html.Em("(Diego)")
                ]),
            ]),

            html.Br(),

            html.H4("Relevancia para el DNP"),
            html.P("Este análisis permite identificar:"),
            html.Ul([
                html.Li("Qué tipo de capital tiene mayor peso estructural"),
                html.Li("Si distintas áreas responden a diferentes determinantes"),
                html.Li("Dónde deben focalizarse las intervenciones según tipo de competencia"),
            ]),

            html.Br(),
            html.H4("Mapa de Puntaje Promedio por Municipio"),

            dcc.Graph(
                figure=px.scatter_geo(
                    df_mcpio,
                    locations="ubicacion",
                    locationmode="country names",
                    size="PUNT_GLOBAL",
                    color="PUNT_GLOBAL",
                    color_continuous_scale="Blues",
                    projection="natural earth",
                    title="Promedio Puntaje Global por Municipio"
                ).update_layout(
                    geo=dict(
                        scope="south america",
                        showland=True,
                        landcolor="rgb(243, 243, 243)",
                    ),
                    margin={"r":0,"t":40,"l":0,"b":0}
                )
            )

        ]),

        # ======================================================
        # =================== PREGUNTA 1 ========================
        # ======================================================

            
dcc.Tab(label="Pregunta 1", value="tab-1", children=[

    html.Br(),
    html.H3("Análisis Pregunta 1"),

    html.Strong(
        "¿Cómo influye el nivel educativo de los padres en el desempeño (Puntaje Global), "
        "independientemente del estrato socioeconómico?"
    ),

    html.Br(), html.Br(),

    html.Div([
        html.Label("Seleccione estratos a incluir:"),
        dcc.Dropdown(
            id="p1-estratos",
            options=[{"label": e, "value": e} for e in estratos_disponibles_p1],
            value=estratos_disponibles_p1,   # por defecto: todos
            multi=True,
            clearable=False
        )
    ], style={"width": "60%"}),

    html.Br(),

    html.Div([
        dcc.Graph(id="p1-violin")
    ]),

    html.Br(),

    html.Div([
        dcc.Graph(id="p1-heatmap")
    ]),

    html.Br(),
    html.H4("Modelo de regresión (controlando por estrato)"),

    html.P(
        "Se estima un modelo lineal para aislar el efecto de la educación parental "
        "sobre el puntaje global, controlando por estrato."
    ),

    html.Div([
        # IZQUIERDA: resumen modelo
        html.Div([
            html.H4("Resultados del modelo", style={"marginBottom": "15px"}),

            html.Pre(
                id="p1-reg-summary",
                style={
                    "whiteSpace": "pre-wrap",
                    "fontSize": "14px",
                    "lineHeight": "1.6",
                    "backgroundColor": "#f2f2f2",
                    "padding": "25px",
                    "borderRadius": "12px",
                    "boxShadow": "0 4px 12px rgba(0,0,0,0.08)",
                    "overflowX": "auto",
                    "border": "1px solid #e0e0e0"
                }
            )
        ], style={"flex": "2"}),

        # DERECHA: fórmula + espacio para explicación
        html.Div([
            html.Div(
                "Puntaje Global = β₀ + β₁(Educación padres) + β₂(Estrato) + ε",
                style={
                    "fontSize": "20px",
                    "fontWeight": "bold",
                    "textAlign": "center",
                    "padding": "20px",
                    "backgroundColor": "white",
                    "borderRadius": "10px",
                    "boxShadow": "0 4px 12px rgba(0,0,0,0.05)"
                }
            ),

            html.Div([
                html.H5("Interpretación (por escribir)", style={"marginTop": "15px"}),
                html.Div(
                    "• [Espacio para 2–4 bullets clave orientados al DNP: hallazgo principal, "
                    "matiz importante, implicación de política, y limitación.]",
                    style={"lineHeight": "1.7"}
                ),
            ], style={
                "backgroundColor": "white",
                "padding": "20px",
                "borderRadius": "10px",
                "boxShadow": "0 4px 12px rgba(0,0,0,0.05)",
                "marginTop": "20px"
            })

        ], style={"flex": "1", "paddingLeft": "40px"})

    ], style={
        "display": "flex",
        "gap": "40px",
        "alignItems": "flex-start",
        "marginTop": "30px"
    }),
]),

        # ======================================================
        # =================== PREGUNTA 2 ========================
        # ======================================================

        dcc.Tab(label="Pregunta 2", value="tab-2", children=[

            html.Br(),
            html.H3("Análisis Pregunta 2"),

            html.Strong(
                "¿En qué medida el acceso a computador e internet en el hogar explica " \
                "las diferencias en el puntaje global de las pruebas Saber 11, una vez considerado " \
                "el estrato socioeconómico, en Cundinamarca?" 
                ""
            ),

            html.Br(),
            html.P(
                "Con esta pregunta, en particular, se busca establecer si los recursos tecnológicos "
                "tienen un efecto propio sobre el desempeño académico, más allá de las desigualdades "
                "estructurales asociadas al nivel socioeconómico."
            ),

            html.Br(),
            html.H4("Análisis General del Puntaje Global"),

 html.Div([

    # -------- COLUMNA IZQUIERDA (GRÁFICO) --------
    html.Div([
        dcc.Graph(id="histograma-general")
    ], style={
        'width': '65%',
        'display': 'inline-block',
        'verticalAlign': 'top'
    }),

    # -------- COLUMNA DERECHA (TEXTO + TABLA) --------
    html.Div([

        html.P(
            "Del histograma notamos que la mayor concentración de estudiantes está entre 220 y 300 puntos, "
            "con un pico alrededor de 250–270, lo que indica que ese es el rango típico de desempeño. "
            "No se observan colas muy largas, lo que indica que no hay muchos estudiantes con puntajes "
            "extremadamente bajos o extremadamente altos.",
            style={"lineHeight": "1.8"}
        ),

        html.H5("Estadísticas Descriptivas",
                style={
                    "marginTop": "25px",
                    "fontWeight": "bold"
                }),

        dash_table.DataTable(
            data=stats_df.to_dict("records"),
            columns=[{"name": i, "id": i} for i in stats_df.columns],
            style_table={
                "marginTop": "10px",
                "width": "90%"   
            },
            style_cell={
                "textAlign": "center",
                "padding": "6px",     
                "fontSize": "12px"    
            },
            style_header={
                "backgroundColor": "#2E8B57",
                "color": "white",
                "fontWeight": "bold",
                "fontSize": "12px"
            },
        )

    ], style={
        'width': '30%',
        'display': 'inline-block',
        'verticalAlign': 'top',
        'paddingLeft': '40px'
    })

]),

            html.P(
                "Continuamos a analizar el comportamiento del puntaje global comparado con estrato"
            ),
            html.Br(),
            html.H4("Puntaje Global vs Estrato"),


            html.Div([

                # TEXTO A LA IZQUIERDA
                html.Div([

                    html.P("Encontramos diferentes hallazgos: "), 
                    html.Ul([
                        html.Li("A mayor estrato socioeconómico, mayor mediana del puntaje global."),
                        html.Li("Aunque existe una tendencia creciente, algunos estudiantes de estratos " \
                        "bajos superan a estudiantes de estratos altos."),
                        html.Li("El estrato parece ser un determinante estructural fuerte, dado el patrón sistemático " \
                        "y consistente en todas las categorías."),
                        html.Li("Mayor concentración en puntajes bajos en estratos 1 y 2," 
                        "con densidad importante entre 200–250 puntos.")
                    ]),

                ], style={
                    'width': '30%',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'paddingRight': '40px'
                }),

                # GRÁFICO A LA DERECHA
                html.Div([
                    dcc.Graph(id="violin-estrato")
                ], style={
                    'width': '65%',
                    'display': 'inline-block'
                })

            ]), 

            html.Br(),

            html.H4(id="titulo-heatmap"),

            dcc.Slider(
                id="slider-estrato",
                min=int(df["FAMI_ESTRATOVIVIENDA"].min()),
                max=int(df["FAMI_ESTRATOVIVIENDA"].max()),
                step=1,
                value=int(df["FAMI_ESTRATOVIVIENDA"].min()),
                marks={
                    int(e): str(int(e))
                    for e in sorted(df["FAMI_ESTRATOVIVIENDA"].unique())
                }
            ),

            html.Br(),

            html.Div([

                # HEATMAP
                html.Div([
                    dcc.Graph(id="heatmap-estrato")
                ], style={
                    'width': '65%',
                    'display': 'inline-block'
                }),

                # TEXTO A LA DERECHA
                html.Div([
                    html.P("Los promedios por combinación de acceso tecnológico refuerzan este hallazgo. En el estrato 1, pasar de no tener acceso a contar con computador e internet implica una mejora cercana a 16 puntos en el puntaje global (de 237,5 a 253,4). En el estrato 6, la diferencia entre no tener acceso y tener ambos recursos supera los 100 puntos (de 216,7 a 323,4). En todos los estratos, la combinación de computador e internet se asocia con los puntajes más altos, lo que evidencia un efecto complementario entre ambos recursos. Estos resultados muestran que el acceso tecnológico incrementa el desempeño dentro de cada nivel socioeconómico, aunque no elimina la brecha estructural entre estratos. ")
                ], style={
                    'width': '30%',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'paddingLeft': '40px'
                })

            ]),

            html.Br(),

            html.H4("Modelo de regresión"),

            html.P(
                "Para aislar el efecto del acceso tecnológico del efecto del estrato, "
                "se estimó un modelo de regresión lineal en el que el puntaje global "
                "se explicó en función de la tenencia de computador, el acceso a internet "
                "y el estrato socioeconómico."
            ),

            html.Br(),

            html.Div([

    # -------- IZQUIERDA: RESULTADOS DEL MODELO --------
    html.Div([

        html.H4("Resultados del modelo de regresión",
                style={"marginBottom": "15px"}),

        html.Pre(
    resumen_modelo,
    style={
        "whiteSpace": "pre-wrap",
        "fontSize": "14px",          
        "lineHeight": "1.6",         
        "backgroundColor": "#f2f2f2",
        "padding": "25px",
        "borderRadius": "12px",
        "boxShadow": "0 4px 12px rgba(0,0,0,0.08)",
        "overflowX": "auto",
        "border": "1px solid #e0e0e0"
    }
)

    ], style={
        "flex": "2"
    }),

    # -------- DERECHA: FORMULA + INTERPRETACION --------
    html.Div([

        # Fórmula destacada
        html.Div(
            "Puntaje Global = β₀ + β₁(Computador) + β₂(Internet) + β₃(Estrato) + ε",
            style={
                "fontSize": "20px",
                "fontWeight": "bold",
                "textAlign": "center",
                "padding": "20px",
                "backgroundColor": "white",
                "borderRadius": "10px",
                "boxShadow": "0 4px 12px rgba(0,0,0,0.05)"
            }
        ),

        # Interpretación debajo
        html.Div([

            html.P(
                "Los resultados indican que tanto el computador (+14,23 puntos) "
                "como el internet (+8,86 puntos) tienen un efecto positivo y "
                "estadísticamente significativo sobre el puntaje global (p < 0,001), "
                "incluso después de controlar por estrato.",
                style={"marginTop": "25px", "lineHeight": "1.7"}
            ),

            html.P(
                "Además, el efecto del computador es mayor que el del internet, "
                "lo que sugiere que la disponibilidad del dispositivo puede ser "
                "más determinante que la conectividad por sí sola.",
                style={"lineHeight": "1.7"}
            ),

            html.P(
                "Por su parte, el estrato presenta un efecto fuerte y creciente: "
                "en comparación con el estrato 1, los estudiantes del estrato 6 "
                "obtienen en promedio 57,40 puntos adicionales, lo que confirma "
                "la persistencia de desigualdades estructurales.",
                style={"lineHeight": "1.7"}
            )

        ], style={
            "backgroundColor": "white",
            "padding": "20px",
            "borderRadius": "10px",
            "boxShadow": "0 4px 12px rgba(0,0,0,0.05)",
            "marginTop": "20px"
        })

    ], style={
        "flex": "1",
        "paddingLeft": "40px"
    })

], style={
    "display": "flex",
    "gap": "40px",
    "alignItems": "flex-start",
    "marginTop": "30px"
})
                
                    

            

        ]),

        # ======================================================
        # =================== PREGUNTA 3 ========================
        # ======================================================

# ======================================================
# =================== PREGUNTA 3 ========================
# ======================================================

dcc.Tab(label="Pregunta 3", children=[

    html.Br(),
    html.H3("Análisis Pregunta 3"),
    html.Strong(
        "¿Qué dimensión del capital familiar (cultural vs económico) tiene mayor poder explicativo "
        "sobre el desempeño en Saber 11?"
    ),

    html.Br(),
    html.P(
        "La narrativa sigue tres pasos: (1) explorar cada dimensión por separado, "
        "(2) observar la interacción entre ambas con mapas de calor, "
        "y (3) comparar modelos predictivos (R² test, ΔR² y R² parcial)."
    ),

    # -----------------------
    # 1) Exploración: Cultura
    # -----------------------
    html.H4("1) Exploración por dimensión"),
    html.Div([

        html.Div([
            html.H5("Capital cultural"),
            html.P("Selecciona la variable cultural para ver el gradiente promedio del puntaje."),
            dcc.Dropdown(
                id="p3-cultura-var",
                options=P3_CULTURE_OPTIONS,
                value=P3_DEFAULT_CULTURE,
                clearable=False
            ),
        ], style={"width": "30%", "display": "inline-block", "verticalAlign": "top", "paddingRight": "40px"}),

        html.Div([
            dcc.Graph(id="p3-graf-cultura")
        ], style={"width": "65%", "display": "inline-block"}),

    ]),

    html.Br(),

    # -----------------------
    # 2) Exploración: Economía
    # -----------------------
    html.Div([

        html.Div([
            html.H5("Capital económico"),
            html.P("Selecciona la variable económica para ver su relación con el puntaje promedio."),
            dcc.Dropdown(
                id="p3-econ-var",
                options=P3_ECON_OPTIONS,
                value=P3_DEFAULT_ECON,
                clearable=False
            ),
        ], style={"width": "30%", "display": "inline-block", "verticalAlign": "top", "paddingRight": "40px"}),

        html.Div([
            dcc.Graph(id="p3-graf-economia")
        ], style={"width": "65%", "display": "inline-block"}),

    ]),

    html.Br(),

    # -----------------------
    # 3) Heatmaps (2)
    # -----------------------
    html.H4("2) Interacción entre dimensiones (mapas de calor)"),
    html.P(
        "El primer mapa muestra el puntaje promedio por celda (Economía × Cultura). "
        "El segundo muestra el tamaño muestral (N) por celda para evitar sobre-interpretar celdas pequeñas."
    ),

    html.Div([
        html.Div([dcc.Graph(id="p3-heatmap-mean")], style={"width": "49%", "display": "inline-block"}),
        html.Div([dcc.Graph(id="p3-heatmap-count")], style={"width": "49%", "display": "inline-block", "float": "right"}),
    ]),

    html.Br(),

    # -----------------------
    # 4) Modelos
    # -----------------------
    html.H4("3) Modelos comparativos (test)"),
    html.P("Usamos un split train/test fijo para reportar R² en test y aportes incrementales (ΔR²)."),

    html.Div([
        html.Div([
            html.Label("Test size (por defecto 0.20):"),
            dcc.Slider(
                id="p3-test-size",
                min=0.1, max=0.4, step=0.05, value=0.2,
                marks={0.1: "0.10", 0.2: "0.20", 0.3: "0.30", 0.4: "0.40"}
            ),
            html.Br(),
            html.Label("Random seed (por defecto 42):"),
            dcc.Input(id="p3-seed", type="number", value=42, step=1),
            html.Br(), html.Br(),
            html.Button("Recalcular modelos", id="p3-calc-models", n_clicks=0),
        ], style={"width": "30%", "display": "inline-block", "verticalAlign": "top", "paddingRight": "40px"}),

        html.Div([

            html.H5("R² en test"),
            dash_table.DataTable(
                id="p3-r2-table",
                columns=[{"name": "Modelo", "id": "Modelo"}, {"name": "R2_test", "id": "R2_test"}],
                data=[],
                style_table={"overflowX": "auto"},
                style_cell={"padding": "8px", "textAlign": "left"},
            ),

            html.Br(),
            html.H5("Aporte incremental y R² parcial"),
            dash_table.DataTable(
                id="p3-delta-table",
                columns=[{"name": "Métrica", "id": "Métrica"}, {"name": "Valor", "id": "Valor"}],
                data=[],
                style_table={"overflowX": "auto"},
                style_cell={"padding": "8px", "textAlign": "left"},
            ),

            html.Br(),
            html.H5("Importancia relativa (coeficientes estandarizados)"),
            dash_table.DataTable(
                id="p3-beta-table",
                columns=[{"name": "Variable", "id": "Variable"}, {"name": "Coef_estandarizado", "id": "Coef_estandarizado"}],
                data=[],
                style_table={"overflowX": "auto"},
                style_cell={"padding": "8px", "textAlign": "left"},
            ),

            html.Br(),
            html.H5("Solapamiento (VIF)"),
            dash_table.DataTable(
                id="p3-vif-table",
                columns=[{"name": "Variable", "id": "Variable"}, {"name": "VIF", "id": "VIF"}],
                data=[],
                style_table={"overflowX": "auto"},
                style_cell={"padding": "8px", "textAlign": "left"},
            ),

            html.Br(),
            html.P(id="p3-overlap-text", style={"lineHeight": "1.7"}),

        ], style={"width": "65%", "display": "inline-block"}),

    ]),

])
])

    ])


# ======================================================
# CALLBACK PREGUNTA 1
# ======================================================

# ======================================================
# CALLBACK PREGUNTA 1 (Educación padres)
# ======================================================

@app.callback(
    Output("p1-violin", "figure"),
    Output("p1-heatmap", "figure"),
    Output("p1-reg-summary", "children"),
    Input("p1-estratos", "value")
)
def actualizar_pregunta1(estratos_sel):

    if not estratos_sel:
        estratos_sel = estratos_disponibles_p1

    dff = df_p1[df_p1["ESTRATO_LABEL"].isin(estratos_sel)].copy()

    # ---------- 1) Violín (EDU_PADRES_CAT vs puntaje, coloreado por estrato) ----------
    fig_violin = px.violin(
        dff,
        x="EDU_PADRES_CAT",
        y="PUNT_GLOBAL",
        color="ESTRATO_LABEL",
        box=True,
        points=False,
        category_orders={
            "EDU_PADRES_CAT": ["Baja", "Media", "Alta"],
            "ESTRATO_LABEL": estratos_disponibles_p1
        },
        title="Distribución del Puntaje Global por educación parental (comparando estratos)"
    )
    fig_violin.update_layout(
        template="simple_white",
        xaxis_title="Educación de los padres (categoría)",
        yaxis_title="Puntaje Global",
        legend_title="Estrato"
    )

    # ---------- 2) Heatmap (promedio + n) ----------
    tabla_media = dff.pivot_table(
        index="EDU_PADRES_CAT",
        columns="ESTRATO_LABEL",
        values="PUNT_GLOBAL",
        aggfunc="mean"
    )
    tabla_n = dff.pivot_table(
        index="EDU_PADRES_CAT",
        columns="ESTRATO_LABEL",
        values="PUNT_GLOBAL",
        aggfunc="size"
    ).fillna(0).astype(int)

    # Orden consistente
    tabla_media = tabla_media.reindex(index=["Baja", "Media", "Alta"], columns=estratos_disponibles_p1)
    tabla_n = tabla_n.reindex(index=["Baja", "Media", "Alta"], columns=estratos_disponibles_p1)

    labels = tabla_media.round(1).astype(str) + "<br>(n=" + tabla_n.astype(str) + ")"

    fig_heat = px.imshow(
        tabla_media,
        text_auto=False,
        aspect="auto",
        title="Promedio de Puntaje Global por educación parental y estrato"
    )
    fig_heat.update_traces(text=labels, texttemplate="%{text}")
    fig_heat.update_layout(
        template="simple_white",
        xaxis_title="Estrato socioeconómico",
        yaxis_title="Educación de los padres (categoría)",
        coloraxis_colorbar=dict(title="Promedio Puntaje")
    )

    # ---------- 3) Regresión ----------
    # PUNT_GLOBAL ~ FAMI_EDU_PADRES_SUM + C(estrato)
    modelo1 = smf.ols(
        "PUNT_GLOBAL ~ FAMI_EDU_PADRES_SUM + C(ESTRATO_LABEL)",
        data=dff
    ).fit()

    return fig_violin, fig_heat, modelo1.summary().as_text()

# ==============================
# HISTOGRAMA MEJORADO
# ==============================

@app.callback(
    Output("histograma-general", "figure"),
    Input("histograma-general", "id")
)
def actualizar_histograma(_):

    fig = px.histogram(
        df,
        x="PUNT_GLOBAL",
        nbins=40,
        color_discrete_sequence=["lightgreen"]
    )

    fig.update_layout(
        template="simple_white",
        title="Distribución del Puntaje Global",
        xaxis_title="Puntaje Global",
        yaxis_title="Frecuencia",
        bargap=0.05
    )

    return fig


# ==============================
# VIOLIN PLOT
# ==============================

@app.callback(
    Output("violin-estrato", "figure"),
    Input("violin-estrato", "id")
)
def actualizar_violin(_):

    fig = px.violin(
        df,
        x="FAMI_ESTRATOVIVIENDA",
        y="PUNT_GLOBAL",
        box=True,
        points=False,
        color_discrete_sequence=["darkgreen"]
    )

    fig.update_layout(
        template="simple_white",
        title="Diagrama de Violín: Puntaje Global por Estrato",
        xaxis_title="Estrato",
        yaxis_title="Puntaje Global"
    )

    return fig

#-------------HEAT MAP----

@app.callback(
    Output("titulo-heatmap", "children"),
    Output("heatmap-estrato", "figure"),
    Input("slider-estrato", "value")
)
def actualizar_heatmap(estrato):

    df_e = df[df["FAMI_ESTRATOVIVIENDA"] == estrato]

    tabla_heatmap = (
        df_e.groupby(
            ["FAMI_TIENECOMPUTADOR", "FAMI_TIENEINTERNET"]
        )["PUNT_GLOBAL"]
        .mean()
        .unstack()
    )

    fig = px.imshow(
        tabla_heatmap,
        text_auto=".1f",
        color_continuous_scale="YlGnBu",
        aspect="auto"
    )

    fig.update_layout(
        title=None,
        xaxis_title="Tiene Internet",
        yaxis_title="Tiene Computador",
        coloraxis_colorbar=dict(
            title="Puntaje Global Promedio"
        ),
        template="simple_white"
    )

    return f"Mapa de Calor: Puntaje Global - Estrato {estrato}", fig

# ======================================================
# CALLBACKS PREGUNTA 3
# ======================================================

@app.callback(
    Output("p3-graf-cultura", "figure"),
    Output("p3-graf-economia", "figure"),
    Output("p3-heatmap-mean", "figure"),
    Output("p3-heatmap-count", "figure"),
    Input("p3-cultura-var", "value"),
    Input("p3-econ-var", "value")
)
def p3_actualizar_graficos(cultura_var, econ_var):
    # df limpio asumido; igual filtramos por seguridad mínima
    df_p3 = df.dropna(subset=["PUNT_GLOBAL", cultura_var, econ_var]).copy()

    fig_cul = _p3_make_mean_line(
        df_p3, cultura_var,
        title=f"Capital cultural: Puntaje promedio vs {cultura_var}",
        xlabel="Variable cultural"
    )

    fig_eco = _p3_make_mean_line(
        df_p3, econ_var,
        title=f"Capital económico: Puntaje promedio vs {econ_var}",
        xlabel="Variable económica"
    )

    fig_hm_mean, fig_hm_n = _p3_make_heatmaps(df_p3, cultura_var, econ_var)

    return fig_cul, fig_eco, fig_hm_mean, fig_hm_n


@app.callback(
    Output("p3-r2-table", "data"),
    Output("p3-delta-table", "data"),
    Output("p3-beta-table", "data"),
    Output("p3-vif-table", "data"),
    Output("p3-overlap-text", "children"),
    Input("p3-calc-models", "n_clicks"),
    State("p3-test-size", "value"),
    State("p3-seed", "value")
)
def p3_actualizar_modelos(n_clicks, test_size, seed):
    # Calcula incluso con n_clicks=0 (Dash llama el callback al cargar)
    try:
        tabla_r2, tabla_marginal, tabla_beta, tabla_vif, r2_overlap = _p3_compute_models(
            df, test_size=float(test_size), random_state=int(seed)
        )

        txt_overlap = (
            f"Solapamiento cultura→economía: un modelo auxiliar sugiere que EDU_MAX_HOGAR explica "
            f"aprox. {r2_overlap:.2%} de la variación del estrato. Esto indica que parte de la señal "
            f"“económica” y “cultural” está estructuralmente solapada (interpretar ΔR² con cuidado)."
        )

        return (
            tabla_r2.to_dict("records"),
            tabla_marginal.to_dict("records"),
            tabla_beta.to_dict("records"),
            tabla_vif.to_dict("records"),
            txt_overlap
        )

    except Exception as e:
        # Si algo falla, no revientes el Dash completo:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, f"Error al recalcular modelos: {e}"



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8055, debug=True)
    
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
from dash import dash_table


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

#cargar datos

df = pd.read_csv("df_limpio.csv")

#modelo de regresión pregunta 2
import statsmodels.formula.api as smf

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
            html.P("aqui va la introducción"),

            html.Br(),

            html.H4("Usuario final"),
            html.P("aqui va la explicación del usuario final"),

            html.Br(),

            html.H4("Integrantes"),
            html.Ul([
                html.Li("Daniela"),
                html.Li("Isa"),
                html.Li("Diego")
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

                html.Label("Seleccione Estrato:"),

                dcc.Dropdown(
                    id="estrato-dropdown",
                    options=[
                        {"label": f"Estrato {e}", "value": e}
                        for e in sorted(df["FAMI_ESTRATOVIVIENDA"].unique())
                    ],
                    value=sorted(df["FAMI_ESTRATOVIVIENDA"].unique())[0],
                    clearable=False
                ),

                dcc.Graph(id="boxplot-puntaje"),
                dcc.Graph(id="bar-promedios")

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

        dcc.Tab(label="Pregunta 3", children=[

            html.Br(),
            html.H4("Análisis para la pregunta 3"),

            dcc.Graph(id="grafico-p3-1"),
            dcc.Graph(id="grafico-p3-2")

        ])

    ])

])

# ======================================================
# CALLBACK PREGUNTA 1
# ======================================================

@app.callback(
    Output("boxplot-puntaje", "figure"),
    Output("bar-promedios", "figure"),
    Input("estrato-dropdown", "value")
)
def actualizar_graficos(estrato):

    df_filtrado = df[df["FAMI_ESTRATOVIVIENDA"] == estrato]

    # Boxplot
    fig_box = px.box(
        df_filtrado,
        x="FAMI_TIENEINTERNET",
        y="PUNT_GLOBAL",
        color="FAMI_TIENECOMPUTADOR",
        title=f"Distribución del Puntaje Global - Estrato {estrato}",
        labels={
            "FAMI_TIENEINTERNET": "Tiene Internet (0=No, 1=Sí)",
            "FAMI_TIENECOMPUTADOR": "Tiene Computador (0=No, 1=Sí)",
            "PUNT_GLOBAL": "Puntaje Global"
        }
    )

    # Promedios
    df_prom = df_filtrado.groupby(
        ["FAMI_TIENECOMPUTADOR", "FAMI_TIENEINTERNET"]
    )["PUNT_GLOBAL"].mean().reset_index()

    fig_bar = px.bar(
        df_prom,
        x="FAMI_TIENECOMPUTADOR",
        y="PUNT_GLOBAL",
        color="FAMI_TIENEINTERNET",
        barmode="group",
        title="Promedio del Puntaje Global",
        labels={
            "FAMI_TIENECOMPUTADOR": "Tiene Computador",
            "FAMI_TIENEINTERNET": "Tiene Internet",
            "PUNT_GLOBAL": "Promedio Puntaje"
        }
    )

    return fig_box, fig_bar

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
# CALLBACKS PREGUNTA 3 (Ejemplo Base)
# ======================================================

@app.callback(
    Output("grafico-p3-1", "figure"),
    Output("grafico-p3-2", "figure"),
    Input("grafico-p3-1", "id")
)
def actualizar_pregunta3(_):

    fig1 = px.box(
        df,
        x="FAMI_ESTRATOVIVIENDA",
        y="PUNT_GLOBAL",
        title="Boxplot por Estrato"
    )

    fig2 = px.bar(
        df.groupby("FAMI_ESTRATOVIVIENDA")["PUNT_GLOBAL"].mean().reset_index(),
        x="FAMI_ESTRATOVIVIENDA",
        y="PUNT_GLOBAL",
        title="Promedio por Estrato"
    )

    return fig1, fig2


if __name__ == "__main__":
    app.run(debug=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
    
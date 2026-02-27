import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# cargar datos
df = pd.read_csv("df_limpio.csv")

# asegurar tipos correctos
df["FAMI_ESTRATOVIVIENDA"] = df["FAMI_ESTRATOVIVIENDA"].astype(int)

# ==============================
# LAYOUT PRINCIPAL CON PESTAÑAS
# ==============================

app.layout = html.Div([

    html.H2("Analítica de Resultados Saber 11 - Cundinamarca"),

    dcc.Tabs([

        # ======================================================
        # =================== PREGUNTA 1 ========================
        # ======================================================

        dcc.Tab(label="Pregunta1", children=[

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

        dcc.Tab(label="Pregunta 2", children=[

            html.Br(),
            html.H4("Análisis para la pregunta 2"),

            dcc.Graph(id="grafico-p2-1"),
            dcc.Graph(id="grafico-p2-2")

        ]),

        # ======================================================
        # =================== PREGUNTA 3 ========================
        # ======================================================

        dcc.Tab(label="pregunta 3", children=[

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


# ======================================================
# CALLBACKS PREGUNTA 2 (Ejemplo Base)
# ======================================================

@app.callback(
    Output("grafico-p2-1", "figure"),
    Output("grafico-p2-2", "figure"),
    Input("estrato-dropdown", "value")  # puedes cambiar inputs
)
def actualizar_pregunta2(estrato):

    fig1 = px.histogram(df, x="PUNT_GLOBAL", title="Distribución General")
    fig2 = px.scatter(df, x="FAMI_ESTRATOVIVIENDA", y="PUNT_GLOBAL",
                      title="Estrato vs Puntaje")

    return fig1, fig2


# ======================================================
# CALLBACKS PREGUNTA 3 (Ejemplo Base)
# ======================================================

@app.callback(
    Output("grafico-p3-1", "figure"),
    Output("grafico-p3-2", "figure"),
    Input("estrato-dropdown", "value")  # puedes cambiar inputs
)
def actualizar_pregunta3(estrato):

    fig1 = px.box(df, x="FAMI_ESTRATOVIVIENDA", y="PUNT_GLOBAL",
                  title="Boxplot por Estrato")
    fig2 = px.bar(df.groupby("FAMI_ESTRATOVIVIENDA")["PUNT_GLOBAL"].mean().reset_index(),
                  x="FAMI_ESTRATOVIVIENDA",
                  y="PUNT_GLOBAL",
                  title="Promedio por Estrato")

    return fig1, fig2


if __name__ == "__main__":
    app.run(debug=True)
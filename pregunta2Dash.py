import dash
from dash import dcc  # dash core components
from dash import html # dash html components
import plotly.express as px
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

#cargar datos de cundinamarca
df = pd.read_csv("Datos_Cundinamarca.csv")

app = dash.Dash(__name__)
server = app.server

#----titulo----
app.layout = html.Div([

    html.H1("Pregunta 2: ¿En qué medida el acceso a computador e internet " \
    "en el hogar explica las diferencias en el puntaje global de las pruebas Saber 11, " \
    "una vez considerado el estrato socioeconómico, en Cundinamarca? "),

#seleccionar el estrato

    html.Label("Seleccione Estrato Socioeconómico:"),
    dcc.Dropdown(
        id="estrato-dropdown",
        options=[
            {"label": str(e), "value": e}
            for e in sorted(df["FAMI_ESTRATOVIVIENDA"].dropna().unique())
        ],
        value=df["FAMI_ESTRATOVIVIENDA"].dropna().unique()[0],
        clearable=False
    ),

    dcc.Graph(id="boxplot-puntaje"),

    dcc.Graph(id="bar-promedios")

])

#callbacks con graficas 
@app.callback(
    Output("boxplot-puntaje", "figure"),
    Output("bar-promedios", "figure"),
    Input("estrato-dropdown", "value")
)
def actualizar_graficos(estrato):

    df_filtrado = df[df["FAMI_ESTRATOVIVIENDA"] == estrato]

    # Boxplot distribución
    fig_box = px.box(
        df_filtrado,
        x="FAMI_TIENEINTERNET",
        y="PUNTAJE_GLOBAL",
        color="FAMI_TIENECOMPUTADOR",
        title="Distribución del Puntaje Global según acceso TIC"
    )

    # Promedios
    df_prom = df_filtrado.groupby(
        ["FAMI_TIENECOMPUTADOR", "FAMI_TIENEINTERNET"]
    )["PUNTAJE_GLOBAL"].mean().reset_index()

    fig_bar = px.bar(
        df_prom,
        x="FAMI_TIENECOMPUTADOR",
        y="PUNTAJE_GLOBAL",
        color="FAMI_TIENEINTERNET",
        barmode="group",
        title="Promedio del Puntaje Global"
    )

    return fig_box, fig_bar

#correr el dash
if __name__ == '__main__':
    app.run(debug=True)

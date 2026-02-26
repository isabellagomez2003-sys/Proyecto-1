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

app.layout = html.Div([

    html.H2("Acceso a TIC y Puntaje Global Saber 11 - Cundinamarca"),

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

])

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


if __name__ == "__main__":
    app.run(debug=True)
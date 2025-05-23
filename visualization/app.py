import dash
from dash import html, dcc, Output, Input, page_registry
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.SPACELAB])

# Navbar dengan logo sejajar dan background putih
navbar = dbc.Navbar(
    dbc.Container(fluid=True, children=[
        dbc.Row([
            dbc.Col(html.Img(
                src="https://petra.ac.id/img/logo-text.2e8a4502.png",
                height="30px",
                style={"backgroundColor": "white", "padding": "1px"},
            ), width="auto", align="center"),

            dbc.Col(html.Div(
                "Dashboard of Electricity Demand Modelling and Prediction Data Visualization",
                className="fw-bold text-white",
                style={"fontSize": "15px"}
            ), width="auto", align="center"),
        ], className="g-2", align="center", style={"flexWrap": "nowrap"}),

        dbc.NavbarToggler(id="navbar-toggler"),
        dbc.Collapse(
            dbc.Nav([
                dbc.NavItem(dbc.NavLink(page["name"], href=page["path"], className="text-white", style={"fontSize": "15px"}))
                for page in dash.page_registry.values()
            ], className="ms-auto", navbar=True),
            id="navbar-collapse",
            navbar=True
        )
    ]),
    color="dark",
    dark=True,
    className="mb-2"
)

# Layout utama
app.layout = dbc.Container(
    fluid=True,
    children=[
        dcc.Location(id="url", refresh=False),  # URL handler
        html.Div(id="redirect-output"),  # Tempat redirect
        dbc.Row([dbc.Col(navbar, width=12)]),  # Navbar
        dbc.Row([dbc.Col(dash.page_container, width=12)])  # Konten utama
    ]
)

# Callback untuk redirect hanya pertama kali
@app.callback(
    Output("redirect-output", "children"),
    Input("url", "pathname")
)
def redirect_to_credits(pathname):
    if pathname == "/" or pathname is None:  # Jika pertama kali buka
        return dcc.Location(href="/credits", id="redirect")
    return None

app.config.suppress_callback_exceptions = True  # Hindari error callback

if __name__ == "__main__":
    app.run(debug=False)



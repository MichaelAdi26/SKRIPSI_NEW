from dash import Dash, dcc, html, register_page, callback
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import datetime
import plotly.express as px
import dash_leaflet as dl
from math import floor, ceil
from lib import attribute as att

register_page(__name__, path='/credits')

def sec_header():
    return dbc.Nav([]
    )

def about():
    return dbc.Col(
        [
            html.H1("About"),
            html.P(
                "This website aims to provide electricity demand forecasts to the wider community. Here you can find electricity demand forecasts covering all provinces in Java-Bali islands with some selected provinces."
            ),
            html.P(
                "This website implements Machine Learning and Deep Learning models developed by members of the research group at Petra Christian University. This website is intended to help predict electricity demand in Java and Bali in the future."
            ),
            html.P([
                    "It is an open-source project, with code",
                    html.A(" available here", href="https://instagram.com/michaeladi26", target="_blank"),".",
                ]),
        ], lg=6, class_name="text-center"
    )

def masalah():
    return dbc.Col([
            html.Div(
                dbc.Container(
                    dbc.Row(
                        dbc.Col([
                                html.H2("Background & Problems", style={"text-align": "center"}),
                                html.Hr(),
                                html.P(
                                    html.Ul([
                                        html.Li( "The increasing use of electrical energy sources in recent years. National electricity sales in 2023 increased by 14.41 TWh or 5.32% compared to 2022 (Ministry of Energy and Mineral Resources, 2024 February 20).", className="lead"),
                                        html.Br(),
                                        html.Li( "Based on statistical data from PT. PLN, where Java and Bali cover around 69% of the total electricity needs in Indonesia (State Electricity Company, 2024).",className="lead"),
                                        html.Br(),
                                        html.Li( "According to Diyono et al. (2023), the demand for electricity in Java-Bali is influenced by several key parameters, including population, number of households, Gross Regional Domestic Product (PDRB), and number of electricity customers.", className="lead"),
                                        ],
                                    ),
                                ),
                            ]),
                    ), className="px-4",
                ), className="bg-light rounded-3 py-5 mb-4",
            ),
        ],lg=6,)

def tujuan():
    return dbc.Col([
            html.Div(
                dbc.Container(
                    dbc.Row(
                        dbc.Col([
                                html.H2("Goals", style={"text-align": "center"}),
                                html.Hr(),
                                html.P(
                                    html.Ul([
                                        html.Li( "Creating an electricity demand prediction system that can provide a clearer picture of the situation and potential of electricity use in Indonesia. In particular, electricity demand that covers all provinces in Java-Bali.", className="lead"),
                                        html.Br(),
                                        html.Li( "Through the electricity demand prediction system, it can help related parties in knowing the need for electricity supply and help make decisions on regional electricity demand in Java and Bali in the future.",className="lead"),
                                        ],
                                    ),
                                ),
                            ]),
                    ), className="px-4",
                ), className="bg-light rounded-3 py-5 mb-4", style={"height": "455px"}
            ),
        ],lg=6)

def builtby():
    return dbc.Col([
            html.Div(
                dbc.Container(
                    dbc.Row(
                        dbc.Col([
                                html.H1([
                                        "Developed by the ",
                                        html.A(
                                            "Data Science and Analytics Study Program",
                                            href="https://dsa.petra.ac.id/",
                                            target="_blank"
                                        ),
                                        " at the Petra Christian University",],
                                    className="display-6",),
                                html.Div([
                                    html.A(
                                        html.Img(src="https://petra.ac.id/img/logo-pcu.4d2cad68.png", height="100px"),
                                        href="https://www.petra.ac.id/",
                                        target="_blank"
                                    ),
                                    html.A(
                                        html.Img(src="https://petra.ac.id/img/logo-text.2e8a4502.png", height="100px"),
                                        href="https://www.petra.ac.id/",
                                        target="_blank"
                                    )
                                ], style={'display': 'flex', 'justifyContent': 'center', 'gap': '20px', 'marginTop': '50px'})
                            ],
                            className="text-center",
                        ),
                    ),
                    className="px-4",
                ),
                className="rounded-3 py-5 mb-4",
            ),
        ],
        lg=12,
    )

def parse_people():
    pic = {
        "satu": "https://informatics.petra.ac.id/wp-content/uploads/2023/07/cropped-Dr.-Gregorius-Satiabudhi-S.T.-M.T-scaled-1.jpg",
        "dua":  "https://electrical.petra.ac.id/wp-content/uploads/2023/07/cropped-S12341545214212.jpg",
        "tiga": "/assets/michael_crop.jpg"
    }
    name = {
        "satu": "Gregorius Satia Budhi, S. T., M. T.",
        "dua" : "Yusak Tanoto, S. T., M. Eng.",
        "tiga": "Michael Adi Pratama"
    }
    desc = {        
        "satu": "Dr. Gregory Satia Budhi, S.T., M.T. is a Lecturer in Informatics Departments, Petra Christian University. His research interests focus on Artificial Intelligence, Machine Learning, Data & Text Mining.",
        "dua": "Yusak Tanoto, S.T., M.Eng., Ph.D. is a Lecturer in Electrical Engineering Departments, Petra Christian University. His research interests focus on Energy Management System & Data Analytics in Renewable Energy.",
        "tiga": "Michael Adi Pratama is Final Year Student in major Data Science and Analytics, Petra Christian University. He concentrates in Big Data and Analytics."
    }
    # Link untuk Instagram atau media lainnya
    link = {
        "satu": None,
        "dua": None,
        "tiga": "https://www.instagram.com/michaeladi26"  # Ganti dengan link Instagram
    }
    lst = ['satu', 'dua', 'tiga']
    return [
        dbc.Col(
            [
                html.A([
                    html.Img(
                        src=pic[i],
                        height="148px",
                        className="rounded-circle shadow",
                    ),
                    html.H5(
                        className="mt-4 font-weight-medium mb-0",
                    )
                ], href=link[i], target="_blank", className="text-decoration-none text-dark") 
                if link[i] else html.Img(
                    src=pic[i],
                    height="148px",
                    className="rounded-circle shadow",
                ),
                html.H5(
                    name[i],
                    className="mt-4 font-weight-medium mb-0",
                ),
                html.H6(
                    "Petra Christian University",
                    className="subtitle mb-3 text-muted",
                ),
                html.P(desc[i], className="desc-text"),
            ],
            lg=4,
            sm=6,
            className="text-center mb-5",
        )
        for i in lst
    ]

def parse_poweredby():
    url = {
        "satu": "http://clipart-library.com/images_k/python-logo-transparent/python-logo-transparent-5.png",
        "dua":  "https://mediaresource.sfo2.digitaloceanspaces.com/wp-content/uploads/2024/04/22112939/scikit-learn-logo-8766D07E2E-seeklogo.com.png",
        "tiga": "https://store-images.s-microsoft.com/image/apps.36868.bfb0e2ee-be9e-4c73-807f-e0a7b805b1be.712aff5d-5800-47e0-97be-58d17ada3fb8.a46845e6-ce94-44cf-892b-54637c6fcf06" ,
        "empat": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ-NEICv1aGTvDRncdvM_fXoah5SNWx4pXAvg&s",
        "lima": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQgSs6bavwSXZUMjdLkJZd9tBPentUjvxuS_ltaMmas5TSzTcEmhIed-1tDAnuC4TDZymM&usqp=CAU",
        "enam": "https://logo-download.com/wp-content/data/images/png/Bootstrap-logo.png",
        "tujuh": "https://numpy.org/images/logo.svg",
        "delapan": "https://xgboost.ai/images/logo/xgboost-logo.png"}    
    name = {
        "satu": "Python",
        "dua" : "Scikit-Learn",
        "tiga": "Plotly Dash",
        "empat": "Pandas",
        "lima": "TensorFlow",
        "enam": "Bootstrap",
        "tujuh": "Numpy",
        "delapan": "XGBoost"}
    link = {        
        "satu": "https://www.python.org/",
        "dua": "https://scikit-learn.org/stable/",
        "tiga": "https://plotly.com/dash/",
        "empat": "https://pandas.pydata.org/",
        "lima": "https://www.tensorflow.org/resources/libraries-extensions",
        "enam": "https://getbootstrap.com/",
        "tujuh": "https://numpy.org/",
        "delapan": "https://xgboost.readthedocs.io/en/release_3.0.0/"}
    lst=['satu','dua','tiga','empat','lima','enam', 'tujuh','delapan']

    return [ dbc.Col(
            [
                html.A(
                    [
                        html.Img(
                            src=url[i],
                            height="96px",
                            style={"margin-bottom": "8px", "margin-right": "16px"}
                        ),
                        html.H5(name[i])],
                    href=link[i],
                    target='_blank'
                )
            ],
            lg=3,
            md=6,
            xs=8,
            className="text-center d-flex flex-column align-items-center justify-content-center",
            style={"margin-bottom": "16px"},
        )
        for i in lst
    ]
    
def footer(): 
    return html.Div(
            dbc.Nav([
            ])  
    )

    
def body_layout():
    return dbc.Container(
        [
            sec_header(),
            dbc.Row([
                about()
            ], class_name="mt-4", justify="center"),
            dbc.Row([
                masalah(),
                tujuan(),
            ], class_name="mt-4"),
            dbc.Row([builtby()]),
            dbc.Row([dbc.Col( html.H2('Contributors', style={"margin-top": "40px", "margin-bottom": "40px"}),lg=12)],className="text-center"),
            dbc.Row(parse_people()), 
            dbc.Row([dbc.Col( html.H2('Library or Tools', style={"margin-top": "45px", "margin-bottom": "55px"}),lg=12)],className="text-center"),
            dbc.Row(parse_poweredby()),
            dbc.Row(footer())
            

        ],
        style={"margin-bottom": "64px"},
        className="mb-5",
    )
layout = html.Div([dcc.Location(id="url", refresh=False), body_layout()])
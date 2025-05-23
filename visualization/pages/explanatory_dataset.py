import dash
from dash import dcc, html, register_page, dash_table, callback
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from lib import attribute as att
import plotly.graph_objects as go
import dash_ag_grid as dag
import plotly.express as px

# Mengambil dataset dari attribute.py
df = att.data_all 
# Dropdown unik (mengambil langsung dari attribute.py)
unique_provinces = att.provinces
unique_years = att.years
unique_parameters = df.columns[2:]  # Ambil kolom selain Province & Year
# Ambil semua kolom kecuali 'Date' dan 'Time'
parameter_options = [col for col in df.columns if col not in ["Province","Year","Latitude","Longitude"]]

register_page(__name__, path='/explanatory_dataset')

# Layout tanpa Navbar (karena sudah ada di app.py)
layout = dbc.Container([
    dbc.Row([dbc.Col(html.H4("Explanatory Dataset", className="text-center fw-bold"))]),
    # Dropdown Filter
    dbc.Row([
        # Warning untuk Fill in All dropdown:
        html.Div(
                id='fill-in-warning',
                children="⚠️ Please fill in all the dropdowns below to view the complete explanatory dataset dashboard!",
                style={'color': 'red', 'fontWeight': 'bold', 'display': 'none'}
            ),
        dbc.Col(dcc.Dropdown(
            id="param-dropdown",
            options=[{"label": param, "value": param} for param in parameter_options],
            placeholder="Choose Parameter",
            multi=True,  # Aktifkan multi-select agar dapat memilih lebih dari 1 parameter atau attribute
            style={"textAlign": "center"}
        ), width=6),
        dbc.Col(dcc.Dropdown(
            id="province-dropdown",
            options=[{"label": prov, "value": prov} for prov in unique_provinces],
            placeholder="Choose Province",
            style={"textAlign": "center"}
        ), width=3),
        dbc.Col(dcc.Dropdown(
            id="year-dropdown",
            options=[{"label": year, "value": year} for year in unique_years],
            placeholder="Choose Year",
            style={"textAlign": "center"}
        ), width=3)
    ],justify="center"),
    html.Br(),
    dbc.Row([dbc.Col(dbc.Card(dcc.Graph(id="map-graph", figure=att.get_indonesia_map(), style={"width": "100%", "height": "400px"}), body=True, style={"opacity": "0.5"}))]),
    html.Br(),
    # Tabel Dataset
    dbc.Row([
    dbc.Col(
        dbc.Card([
            dbc.CardHeader(html.H5("Table Dataset of Electricity Demand", className="text-center fw-bold")),
            dbc.CardBody([
                dag.AgGrid(
                    id="table-dataset",
                    rowData=[],  # Data diisi dari callback
                    columnDefs=[],  # Kolom diisi dari callback
                    defaultColDef={"resizable": True, "sortable": True},  
                    style={"height": "350px", "width": "100%"},
                    # **Gunakan Tema Dark**
                    className="ag-theme-alpine",              
                )], style={"backgroundColor": "#fff", "padding": "2px"})
                    ]), width=20
                )
            ]),
    html.Br(),
    dbc.Row([
        dbc.Col([
            html.H5("Average Daily Electricity Demand Line Chart in Selected Province in Selected Year", className="text-center fw-bold"),
            dcc.Graph(id="linechart-avg-demand-energy", style={"width": "100%", "height": "450px"})  
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.H5("Tree Map Chart of the Top 5 Regencies Based on Total Electricity Demand", className="text-center fw-bold"),
            dcc.Graph(id="heatmap-top-5-regency", style={"width": "100%", "height": "500px"})  
        ])
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([
        html.H5("Bar Chart - Top 3 Regencies Based on Highest Daily Temperature", className="text-center fw-bold"),
        dcc.Graph(id="bar-top-3-regency", style={"width": "100%", "height": "300px"})], width=6),  # Kolom 2: Bar Chart (Top 3)
        dbc.Col([
            html.H5("Bar Chart - Bottom 3 Regencies Based on Lowest Daily Temperature", className="text-center fw-bold"),
            dcc.Graph(id="bar-bottom-3-regency", style={"width": "100%", "height": "300px"})], width=6),  # Kolom 3: Bar Chart (Bottom 3)
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([
            html.H5("Tree Map Chart of List of Regencies in Selected Province in Selected Year", className="text-center fw-bold"),
            dcc.Graph(id="heatmap-list-regency", style={"width": "100%", "height": "500px"})
        ])
    ]),
    html.Br(),
])

@callback(
    [
        Output("map-graph", "figure"),
        Output("table-dataset", "columnDefs"),
        Output("table-dataset", "rowData"),
        Output("heatmap-top-5-regency", "figure"),
        Output("bar-top-3-regency", "figure"),
        Output("bar-bottom-3-regency", "figure"),
        Output("linechart-avg-demand-energy", "figure"),
        Output("heatmap-list-regency", "figure"),
    ],
    [
        Input("param-dropdown", "value"),
        Input("province-dropdown", "value"),
        Input("year-dropdown", "value")
    ]
)
def update_graph(selected_params, selected_province, selected_year):
    print(f"Params: {selected_params}, Province: {selected_province}, Year: {selected_year}")
    # Logika Map
    if not selected_params or not selected_province or not selected_year:
        print("✅ All dropdowns are empty, showing a map of Indonesia")
        toggle_fill_in_warning()
        map_fig = att.get_indonesia_map()
    else:
        map_fig = att.get_province_point(selected_province,selected_year,top_n=20)

    # Filter dataset
    filtered_df = df[
        (df["Province"] == selected_province) & 
        (df["Year"] == selected_year)
    ]
    print(f"🔹Amount of Data After Filtering: {len(filtered_df)}")
    fig = go.Figure()
    if filtered_df.empty:
        print("⛔ There is no matching data, because the filter_df is empty")
        return fig, [], []

    # Pilih kolom yang akan ditampilkan
    selected_columns = ["Province", "Year"]  
    if selected_params:
        selected_columns += [param for param in selected_params if param in filtered_df.columns]
    print(f"🔹 Table Column: {selected_columns}")
    filtered_df = filtered_df[selected_columns]
    filtered_df.columns = filtered_df.columns.str.strip()  

    # Daftar semua kolom numerik yang bisa diformat
    all_numeric_columns = ["Jumlah_Penduduk", "Jumlah_Penduduk_Miskin", "PDRB", "Jumlah_Pelanggan_Listrik", "Listrik_Terjual", "Daya_Terpasang", "Produksi_Listrik"]

    # Filter hanya kolom numerik yang dipilih oleh user
    numeric_columns = [col for col in selected_columns if col in all_numeric_columns]
    # Definisi kolom dengan format yang berbeda sesuai kebutuhan
    columnDefs = [
        {
            "headerName": col,
            "field": col,
            "headerClass": "center-header", 
            "cellStyle": {"textAlign": "center"} 
        } if col not in numeric_columns else {
            "headerName": col,
            "field": col,
            "type": "numericColumn",
            "valueFormatter": {"function": "d3.format(',')(params.value)"},  
            "headerClass": "center-header",  
            "cellStyle": {"textAlign": "center"}  
        }
        for col in selected_columns
    ]
    # Konversi data hanya untuk kolom numerik yang ada
    rowData = filtered_df.astype({col: "float" for col in numeric_columns}).to_dict("records")
   # 🔹 Logika Heatmap Top 5 Regency by Demand Energy
    top_5_df = att.get_top_5_regency_by_demand(selected_province, selected_year,selected_params)

    if not top_5_df.empty:
        print("🔹Top 5 Regencies based on electricity demand")
        print(top_5_df)
        # Buat Heatmap Tree dengan Plotly Treemap utk visualisasi Top5 regency
        top5_heatmap_fig = px.treemap(
            top_5_df,
            path=["Regency"],  values="Demand",  color="Demand",  
            color_continuous_scale="reds", custom_data=["Regency", "Demand"]
            # title=f"Top 5 Regencies Based on Total Electricity Demand in {selected_year} ({selected_province})"
        )
        # Perbaiki hover text agar hanya menampilkan yang diperlukan
        top5_heatmap_fig.update_traces(
            hovertemplate="<b>Regency:</b> %{customdata[0]}</b><br>" +  f"Province: {selected_province}<br>" +  "Electricity Demand: %{customdata[1]:,.2f} KWh",
            insidetextfont=dict(size=18),  # Perbesar teks dalam kotak
            outsidetextfont=dict(size=18)  # Jika ada teks di luar (jarang di Treemap)  
        )
        top5_heatmap_fig.update_layout(
            margin=dict(l=0, r=0, t=20, b=20),  
            coloraxis_colorbar=dict(title="Electricity Demand (KWh)", orientation='h',  x=0.5, xanchor="center", y=-0.2, yanchor="bottom", len=0.7, 
                                    thickness=20, tickfont=dict(size=12), titlefont=dict(size=14), title_side="top"), 
            title_x=0.5)
    else:
        print("⛔ No data for Top 5 Regency, because regency is null")
        top5_heatmap_fig = go.Figure()

    # Logika Bar Chart Top 3 regency by Highest average Temperature:
    top_3_df = att.get_top_3_regency_by_temperature(selected_province, selected_year, selected_params)

    if not top_3_df.empty:
        print("🔹Top 3 Regency based on Highest Temperature:")
        print(top_3_df)
        # Buat Bar Chart dengan Plotly Treemap utk visualisasi Top3 regency
        top3_barchart_fig = px.bar(
            top_3_df,
            x="Temperature", y="Regency",  orientation="h", 
            color="Temperature", color_continuous_scale="reds", text="Temperature",  
            # title=f"Top 3 Regencies Based on Highest Average Daily Temperature in {selected_year} ({selected_province})"
        )
        # Atur tampilan hover text agar lebih informatif
        top3_barchart_fig.update_traces(
            texttemplate="%{text:.2f}°C",  
            textposition="inside", 
            hovertemplate="<b>Regency:</b> %{y}<br>" +  f"Province: {selected_province}<br>" + "Temperature: %{x:.2f}°C"
        )
        # Perbaiki layout agar lebih rapi
        top3_barchart_fig.update_layout(
            xaxis_title="Temperature (°C)", yaxis_title="Regency",
            margin=dict(l=50, r=20, t=40, b=40), title_x=0.5  
        )
    else:
        print("⛔ No data for Top 3 Regency, because regency and temperature is null")
        top3_barchart_fig = go.Figure()

    # Logika Bar Chart Bottom 3 regency by Lowest average Temperature:
    bottom_3_df = att.get_bottom_3_regency_by_temperature(selected_province, selected_year, selected_params)

    if not bottom_3_df.empty:
        print("🔹 Top 3 Regency based on Lowest Temperature:")
        print(bottom_3_df)
        # Buat Bar Chart dengan Plotly Treemap utk visualisasi Top3 regency
        bottom3_barchart_fig = px.bar(
            bottom_3_df,
            x="Temperature", y="Regency",  orientation="h", 
            color="Temperature", color_continuous_scale="reds", text="Temperature",  
            # title=f"Bottom 3 Regencies Based on Lowest Average Daily Temperature in {selected_year} ({selected_province})"
        )
        # Atur tampilan hover text agar lebih informatif
        bottom3_barchart_fig.update_traces(
            texttemplate="%{text:.2f}°C",  textposition="inside", 
            hovertemplate="<b>Regency:</b> %{y}<br>" +  f"Province: {selected_province}<br>" +  "Temperature: %{x:.2f}°C"
        )
        # Perbaiki layout agar lebih rapi
        bottom3_barchart_fig.update_layout(
            xaxis_title="Temperature (°C)", yaxis_title="Regency",
            margin=dict(l=50, r=20, t=40, b=40), title_x=0.5  
        )
    else:
        print("⛔ No data for Bottom 3 Regency, because regency and temperature is null")
        bottom3_barchart_fig = go.Figure()

    # Logika Line Chart Average Demand Energy:
    if selected_params and "Demand" in selected_params and selected_province and selected_year:
        # Filter berdasarkan provinsi dan tahun yang dipilih
        filtered_df = df[
            (df["Province"] == selected_province) & (df["Year"] == selected_year) 
        ]
        if not filtered_df.empty and "Demand" in filtered_df.columns:
            filtered_df["Date"] = pd.to_datetime(filtered_df["Date"]) 
            # Buat tiga versi data (Daily, Weekly, Monthly)
            df_daily = filtered_df.groupby("Date", as_index=False)["Demand"].mean()
            df_weekly = filtered_df.set_index("Date").resample("W")["Demand"].mean().reset_index()
            df_monthly = filtered_df.set_index("Date").resample("M")["Demand"].mean().reset_index()

            # Buat Figure dengan 3 trace untuk Daily, Weekly, Monthly
            avg_demand_fig = go.Figure()
            # Daily (Default Visible)
            avg_demand_fig.add_trace(go.Scatter(x=df_daily["Date"], y=df_daily["Demand"], mode="lines", name="Daily", visible=True))
            # Weekly (Hidden)
            avg_demand_fig.add_trace(go.Scatter(x=df_weekly["Date"], y=df_weekly["Demand"], mode="lines", name="Weekly", visible=False))
            # Monthly (Hidden)
            avg_demand_fig.add_trace(go.Scatter(x=df_monthly["Date"], y=df_monthly["Demand"], mode="lines", name="Monthly", visible=False))

            # Menambahkan Dropdown Menu
            # Buat list tanggal awal setiap bulan di tahun yang dipilih
            tickvals = pd.date_range(start=f"{selected_year}-01-01", end=f"{selected_year}-12-01", freq='MS')
            ticktext = [date.strftime('%b %y') for date in tickvals]  # Format contoh: Jan 21, Feb 21, dst.

            avg_demand_fig.update_layout(
                updatemenus=[
                    {
                        "buttons": [
                            {"label": "Daily", "method": "update", "args": [{"visible": [True, False, False]}]},
                            {"label": "Weekly", "method": "update", "args": [{"visible": [False, True, False]}]},
                            {"label": "Monthly", "method": "update", "args": [{"visible": [False, False, True]}]},
                        ],
                        "direction": "down", "font": {"color": "black"}, "bgcolor": "#f0f0f0",  "bordercolor": "black", 
                        "active": 0, "x": 1.15, "xanchor": "right", "y": 1.2, "yanchor": "top"
                    }
                ],
                # title={
                #     # "text": f"Average Daily Electricity Demand in {selected_province} ({selected_year})",
                #     "x": 0.5, 
                #     "xanchor": "center",
                #     "yanchor": "top"
                # },
                xaxis=dict(
                    title="Date", tickvals=tickvals,  # Set label hanya di awal bulan
                    ticktext=ticktext, tickangle=-45  # Supaya tidak bertabrakan
                ),
                yaxis=dict(
                    title="Average Daily Electricity Demand (KWh)", automargin=True, range=[0, 800]
                ),
                title_x=0.5, margin=dict(l=20, t=60, b=40), # margin untuk line chart visualisasi data
                # Background plot
                template="plotly_white", plot_bgcolor="white", paper_bgcolor="white", font=dict(color="black") 
            )
            avg_demand_fig.update_xaxes(tickangle=-45) 
        else:
            avg_demand_fig = go.Figure()
            print("⛔ There is no data for calculating the average electricity demand")
    else:
        avg_demand_fig = go.Figure()
        print("⛔ The calculation of the average electrical energy demand was not performed because the 'Demand' parameter was not selected or the dropdown was incomplete")

    # Logika List Regency Treemap
    if selected_province and ("Demand" in selected_params):
        # Ambil data Regency menggunakan fungsi dari attribute.py
        list_regency_treemap_data = att.get_all_regency(selected_province, selected_year, df)

        if not list_regency_treemap_data.empty:
            print(f"🔹 Treemap List Regency in {selected_province} ({selected_year}):")
            print(list_regency_treemap_data)

            # Treemap Chart menggunakan px.treemap
            list_regency_treemap_fig = px.treemap(
                list_regency_treemap_data,
                path=["Regency"], values="Demand",   
                color="Demand", color_continuous_scale="reds", custom_data=["Province", "Regency", "Demand"]
                # title=f"Treemap of Demand Energy in {selected_province} ({selected_year})",
            )
            # Perbaiki hover text agar menampilkan Province, Regency, dan Demand
            list_regency_treemap_fig.update_traces(
                hovertemplate="<b>Province:</b> %{customdata[0]}<br>" + "<b>Regency:</b> %{customdata[1]}<br>" + "<b>Electricity Demand:</b> %{customdata[2]:,.2f} KWh"
            )
            list_regency_treemap_fig.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                template="plotly_dark", title_x=0.5  
            )
        else:
            list_regency_treemap_fig = go.Figure()
            print("⛔ There is no data for Treemap List Regency")
    else:
        list_regency_treemap_fig = go.Figure()
        print("⛔ Treemap is not displayed because the parameters are not selected correctly")

    return map_fig, columnDefs, rowData, top5_heatmap_fig, top3_barchart_fig, bottom3_barchart_fig, avg_demand_fig, list_regency_treemap_fig

# Warning untuk Fill in Dropdown:
@callback(
    Output('fill-in-warning', 'style'),
    Input('param-dropdown', 'value'),
    Input('province-dropdown', 'value'),
    Input('year-dropdown', 'value'),
    prevent_initial_call=True
)
def toggle_fill_in_warning(selected_param, selected_province, selected_year):
    if selected_param is None or selected_province is None or selected_year is None:
        return {'color': 'red', 'fontWeight': 'bold', 'display': 'block'}
    return {'display': 'none'}
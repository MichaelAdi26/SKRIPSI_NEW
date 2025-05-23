import dash, os, re
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash import dcc, html, register_page, dash_table, callback, no_update
import plotly.express as px
import pandas as pd
from lib import attribute as att  # Import dataset dari file attribute.py
from dash.dash_table.Format import Format, Group, Scheme

# Ambil dataset dari attribute.py
df = att.data_all
df2 = att.data_all
# Dropdown unik
unique_provinces = att.provinces
unique_years = att.years
unique_parameters = df.columns[2:]  # Kolom selain Province & Year
# Ambil semua kolom kecuali 'Date' dan 'Time'
parameter_options = [col for col in df.columns if col not in ["Date", "Time","Province", "Regency", "Year","Latitude","Longitude", "Demand", "Temperature", "Persentase_Penduduk_Miskin", "Geo"]]
register_page(__name__, path='/raw_dataset')

# Layout Dashboard
layout = html.Div([
    html.Br(),
    dbc.Row([dbc.Col(html.H4("Raw Dataset of Electricity Demand", className="text-center border border-dark p-2 fw-bold"))]),
    html.Br(),
    html.Div([
                # Dropdown Filter & Submit Button
                dbc.Row([
                    # Warning untuk Waktu menampilkan butuh beberapa menit:
                    html.Div(
                            id='open-warning',
                            children="⚠️ It takes 3 to 5 minutes to display the raw dataset.",
                            style={'color': 'red', 'fontWeight': 'bold', 'display': 'none'}
                        ),
                    dbc.Col(dcc.Dropdown(id="param-dropdown",
                                        options=[{"label": p, "value": p} for p in parameter_options],
                                        placeholder="Choose Parameter"), width=4),
                    dbc.Col(dcc.Dropdown(id="province-dropdown",
                                        options=[{"label": p, "value": p} for p in unique_provinces],
                                        placeholder="Choose Province"), width=4),
                    dbc.Col(dcc.Dropdown(id="year-dropdown",
                                        options=[{"label": y, "value": y} for y in unique_years],
                                        placeholder="Choose Year"), width=3),
                    dbc.Col(dbc.Button("Submit", id="submit-btn", color="success"), width=1)
                ], justify="center"),
    ]),
    html.Br(),

    # Tabel Raw Dataset (Default: Hidden)
    dbc.Row(id="table-raw-data", children=[
        dbc.Col(html.H5("Table Raw Dataset", className="fw-bold", style={"textAlign": "center"})),
        dbc.Col(dash_table.DataTable(
            id="data-raw",
            columns=[{"name": col, "id": col} for col in df.columns],
            page_size=11,
            style_table={'overflowX': 'auto'},
            style_cell={  
                'textAlign': 'center',  # Memusatkan isi sel
                'padding': '5px',  # Menambahkan padding agar lebih rapi
            },
            style_header={
                'textAlign': 'center',  # Memusatkan header kolom
                'fontWeight': 'bold',  # Membuat header lebih tegas
                'backgroundColor': '#f8f9fa',  # Memberikan warna background ke header
            },
                ), width=12)
        ], style={"textAlign": "center"}),
    html.Br(),

    # Tabel Dataset after preprocessing:
    dbc.Row(id="table-preprocess-data", children=[
        dbc.Col(html.H5("Table Dataset after Data Cleaning & Geo Matching", className="fw-bold", style={"textAlign": "center"})),
        dbc.Col(dash_table.DataTable(
            id="data-preprocess",
            columns=[{"name": col, "id": col} for col in df2.columns],
            page_size=11,
            style_table={'overflowX': 'auto'},
            style_cell={  
                'textAlign': 'center',  # Memusatkan isi sel
                'padding': '5px',  # Menambahkan padding agar lebih rapi
            },
            style_header={
                'textAlign': 'center',  # Memusatkan header kolom
                'fontWeight': 'bold',  # Membuat header lebih tegas
                'backgroundColor': '#f8f9fa',  # Memberikan warna background ke header
            },
                ), width=12)
        ], style={"textAlign": "center"}),
    html.Br(),

    # Tabel Dataset selected parameter before Interpolasi Data:
    dbc.Row(id="table-before-interpolasi-data", children=[
        dbc.Col(html.H5("Table Dataset Before Interpolation Data BY Selected Parameter", className="fw-bold", style={"textAlign": "center"})),
        dbc.Col(dash_table.DataTable(
            id="data-before-interpolation",
            columns=[],
            data=[],
            page_size=12,
            style_table={'overflowX': 'auto'},
            style_cell={  
                'textAlign': 'center',  # Memusatkan isi sel
                'padding': '5px',  # Menambahkan padding agar lebih rapi
            },
            style_header={
                'textAlign': 'center',  # Memusatkan header kolom
                'fontWeight': 'bold',  # Membuat header lebih tegas
                'backgroundColor': '#f8f9fa',  # Memberikan warna background ke header
            },
            ), width=12)
        ], style={"textAlign": "center", "display": "none"}),
    html.Br(),

    # Tabel Dataset selected parameter after Interpolasi Data:
    dbc.Row(id="table-after-interpolasi-data", children=[
        dbc.Col(html.H5("Table Dataset After Interpolation Data BY Selected Parameter and Selected Year", className="table-striped fw-bold", style={"textAlign": "center"})),
        dbc.Col(dash_table.DataTable(
            id="data-after-interpolation",
            columns=[],
            data=[],
            page_size=12,
            style_table={'overflowX': 'auto'},
            style_cell={  
                'textAlign': 'center',  # Memusatkan isi sel
                'padding': '5px',  # Menambahkan padding agar lebih rapi
            },
            style_header={
                'textAlign': 'center',  # Memusatkan header kolom
                'fontWeight': 'bold',  # Membuat header lebih tegas
                'backgroundColor': '#f8f9fa',  # Memberikan warna background ke header
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#e0e0e0',  # Warna untuk baris ganjil
                },
                {
                    'if': {'state': 'active'},
                    'backgroundColor': '#d3d3d3',  # Warna saat baris di-hover
                    'border': '1px solid #A9A9A9',
                }
            ]), width=12)
        ], style={"textAlign": "center", "display": "none"}),
    html.Br(),

    # TreeMap Chart (Default: Hidden)
    dbc.Row(id="treemap-container", children=[
        dbc.Col(html.H5("TreeMap Chart: Sum of Data in Selected Province", className="fw-bold", style={"textAlign": "center"})),
        dbc.Col(dcc.Graph(id="treemap-chart"), width=12)
    ], style={"textAlign": "center"})
])

# Callback untuk update semua grafik & tabel sekaligus
@callback(
    [Output("treemap-chart", "figure"),
     Output("data-raw", "data"),
     Output("data-raw", "columns"),
     Output("data-preprocess", "data"),
     Output("data-preprocess", "columns"),
     Output("table-raw-data", "style"),
     Output("table-preprocess-data", "style"),
     Output("data-before-interpolation", "data"),
     Output("data-before-interpolation", "columns"),
     Output("table-before-interpolasi-data", "style"),
     Output("data-after-interpolation", "data"),
     Output("data-after-interpolation", "columns"),
     Output("table-after-interpolasi-data", "style"),
     Output("treemap-container", "style"),],
    [Input("submit-btn", "n_clicks")],
    [State("province-dropdown", "value"),
     State("year-dropdown", "value"),
     State("param-dropdown", "value")]
)
def update_graph(n_clicks, selected_province, selected_year, selected_param):

    dataset_paths = {
                        2021: "C:/Users/Michael Adi/Documents/DATA SKRIPSI/SKRIPSI/dataset_mentah/2021_merged_before_geo_matching.parquet",
                        2022: "C:/Users/Michael Adi/Documents/DATA SKRIPSI/SKRIPSI/dataset_mentah/2022_merged_before_geo_matching.parquet",
                        2023: "C:/Users/Michael Adi/Documents/DATA SKRIPSI/SKRIPSI/dataset_mentah/2023_merged_before_geo_matching.parquet"
                    }
    dataset_after_paths = {
                        2021: "C:/Users/Michael Adi/Documents/DATA SKRIPSI/SKRIPSI/dataset_mentah/2021_merged_after_geo_matching.parquet",
                        2022: "C:/Users/Michael Adi/Documents/DATA SKRIPSI/SKRIPSI/dataset_mentah/2022_merged_after_geo_matching.parquet",
                        2023: "C:/Users/Michael Adi/Documents/DATA SKRIPSI/SKRIPSI/dataset_mentah/2023_merged_after_geo_matching.parquet"
                          }
    
    # Jika tombol belum ditekan, sembunyikan tabel dan grafik
    if n_clicks is None or n_clicks == 0:
        empty_fig = px.treemap(title="TreeMap is Empty")  # Buat grafik kosong
        return empty_fig, [], [], [], [], {"display": "none"}, {"display": "none"}, [], [], {"display": "none"}, [], [], {"display": "none"}, {"display": "none"}  # data-before-interpolation

    # Jika salah satu dropdown tidak diisi, tetap sembunyikan tabel dan grafik
    if not (selected_province and selected_year and selected_param):
        empty_fig = px.treemap(title="TreeMap is Empty because one of the dropdowns is not filled")  
        return empty_fig, [], [], [], [], {"display": "none"}, {"display": "none"}, [], [], {"display": "none"}, [], [], {"display": "none"}, {"display": "none"}  # data-before-interpolation

    filtered_tree_df = df[(df["Province"] == selected_province) & (df["Year"] == selected_year)]

    # Hitung jumlah baris per Regency
    regency_counts = filtered_tree_df.groupby(["Province", "Regency"]).size().reset_index(name="Count")

    # Treemap List Regency by Count of Data per Province:
    if not regency_counts.empty:
        fig_treemap = px.treemap(
                                regency_counts,
                                path=["Province", "Regency"],  # Struktur hierarki (Province → Regency)
                                values="Count",
                                title=f"Data Distribution per Regency in {selected_province} Province, Year {selected_year}",
                                hover_data={"Province": True, "Regency": True, "Count": True}, 
                                )
        fig_treemap.update_layout(title_x=0.5) 
        fig_treemap.update_traces(
                    hovertemplate="<b>Province:</b> %{customdata[0]}<br>" "<b>Regency:</b> %{label}<br>" "<b>Sum of Data:</b> %{value}<extra></extra>",
                    customdata=regency_counts[["Province"]].values
        )
    else:
        fig_treemap = px.treemap(title="Data not Available")  
        fig_treemap.update_layout(title_x=0.5)  

    # Cek apakah tahun memiliki path yang tersedia --> Untuk Tabel Raw Dataset dan Tabel Dataset after Preprocessing
    dataset_path = dataset_paths.get(selected_year)
    dataset_after_paths = dataset_after_paths.get(selected_year)

    if (not dataset_path or not os.path.exists(dataset_path)) or (not dataset_after_paths or not os.path.exists(dataset_after_paths)):
        empty_fig = px.treemap(title=f"Data for Year {selected_year} is not available, because the selected year is invalid")
        return empty_fig, [], [], [], [], {"display": "none"}, {"display": "none"}, [], [], {"display": "none"}, [], [], {"display": "none"}, {"display": "none"}  # data-before-interpolation

    # Baca dataset sesuai tahun yang dipilih
    df_year = pd.read_parquet(dataset_path)
    df2_year = pd.read_parquet(dataset_after_paths)

    # Menambahkan prefix atau keterangan di awal kolom, agar tidak terjadi ommit saat di read dan pengecekan data di code function:
    df_year = df_year.add_prefix("raw_")
    df2_year = df2_year.add_prefix("cleaned_")
    
    # dataset before cleaning & geo matching
    new_headers = df_year.iloc[0].tolist()  # Ambil baris pertama sebagai list header
    df_year.columns = new_headers  # Ganti nama kolom dengan header baru
    # Tampilkan hanya 50 baris pertama dan 50 kolom pertama tanpa mengubah tipe data
    df_limited = df_year.iloc[:2600, :2600]  # Ambil 50 baris pertama dan 50 kolom pertama
    # Tampilkan semua data dari tahun yang dipilih tanpa filter tambahan
    data_table = df_limited.to_dict("records")
    columns = [{"name": col, "id": col} for col in df_limited.columns]

    # dataset after cleaning & geo matching
    new_headers2 = df2_year.iloc[2].tolist()  # Ambil baris pertama sebagai list header
    df2_year.columns = new_headers2  # Ganti nama kolom dengan header baru
    # Tampilkan hanya 50 baris pertama dan 50 kolom pertama tanpa mengubah tipe data
    df_limited2 = pd.concat([df2_year.iloc[:2], df2_year.iloc[27:2600]], ignore_index=True)
    # Tampilkan semua data dari tahun yang dipilih tanpa filter tambahan
    data_table2 = df_limited2.to_dict("records")
    columns2 = [{"name": col2, "id": col2} for col2 in df_limited2.columns]

    # Dataset by selected parameter before interpolasi dan after interpolasi:
    # File path & sheet mapping
    selected_param_path = "C:/Users/Michael Adi/Documents/DATA SKRIPSI/SKRIPSI/dataset_mentah/Dataset Jumlah Penduduk & PDRB Provinsi (2018 - 2023).xlsx"
    sheet_mapping = {
        "Jumlah_Penduduk": "Data Jumlah Penduduk",
        "Jumlah_Penduduk_Miskin": "Data Jumlah Penduduk Miskin",
        "PDRB": "Data PDRB per Kapita Atas Dasar",
        "Jumlah_Pelanggan_Listrik": "Data Jumlah Pelanggan Listrik",
        "Listrik_Terjual": "Terpasang, Produksi,Listrik KWh",
        "Daya_Terpasang": "Terpasang, Produksi,Listrik KWh",
        "Produksi_Listrik": "Terpasang, Produksi,Listrik KWh"
    }
    data_table3, columns3 = [], []
    data_table4, columns4 = [], []
    # Folder mapping khusus untuk dataset per provinsi dan per tahun
    param_folder_mapping = {
        "Jumlah_Penduduk": "Data Demografi dan PDRB",
        "PDRB": "Data Demografi dan PDRB",
        "Jumlah_Penduduk_Miskin": "Data Demografi dan PDRB",
        "Jumlah_Pelanggan_Listrik": "Data Pelanggan Listrik, Daya Terpasang, Listrik Terjual, dan Produksi Listrik",
        "Daya_Terpasang": "Data Pelanggan Listrik, Daya Terpasang, Listrik Terjual, dan Produksi Listrik",
        "Produksi_Listrik": "Data Pelanggan Listrik, Daya Terpasang, Listrik Terjual, dan Produksi Listrik",
        "Listrik_Terjual": "Data Pelanggan Listrik, Daya Terpasang, Listrik Terjual, dan Produksi Listrik"
    }
    base_path = r"C:/Users/Michael Adi/Documents/DATA SKRIPSI/SKRIPSI"

    if os.path.exists(selected_param_path) and selected_param:
        try:
            print("Selected Param:", selected_param)
            sheet_name = sheet_mapping.get(selected_param.strip())
            print("Sheet Name:", sheet_name)

            if not sheet_name:
                print(f"Parameter '{selected_param}' not found in mapping.")
            else:
                if selected_param in ["Jumlah_Penduduk", "PDRB", "Jumlah_Pelanggan_Listrik"] and selected_province and selected_year:
                    # 📌 Dataset standar lainnya, baca dari baris ke-3 (skip header dan 1 baris)
                    df_param = pd.read_excel(
                        selected_param_path,
                        sheet_name=sheet_name,
                        skiprows=1
                    )
                    # Hapus kolom 'No' jika ada
                    df_param = df_param.drop(columns=["No"], errors="ignore")
                    # Identifikasi kolom tahun (yang bentuknya 4 digit angka)
                    tahun_columns = [col for col in df_param.columns if re.match(r"^\d{4}$", str(col))]

                    # Konversi angka: hapus koma dan ubah ke float
                    for col in tahun_columns:
                        df_param[col] = (
                            df_param[col]
                            .astype(str)
                            .str.replace(",", "", regex=False)
                            .replace("nan", None)
                            .astype(float)
                        )
                    # Ubah ke long format
                    df_long = df_param.melt(id_vars=["Province"], var_name="Year", value_name="Value")
                    # Filter tahun numerik dan cast ke int
                    df_long = df_long[df_long["Year"].astype(str).str.match(r"^\d{4}$")]
                    df_long["Year"] = df_long["Year"].astype(int)
                    # Filter berdasarkan tahun
                    df_filtered = df_long[df_long["Year"] == selected_year]

                    # Membuka Folder File Data after Interpolation data:
                    folder_name = param_folder_mapping.get(selected_param)
                    full_path = os.path.join(base_path, folder_name, selected_province)

                    file_to_open = None
                    for file in os.listdir(full_path):
                        if file.endswith(".xlsx") and str(selected_year) in file:
                            file_to_open = os.path.join(full_path, file)
                            break
                    if file_to_open:
                        print("Opening file:", file_to_open)
                        df_param_after = pd.read_excel(file_to_open)
                        # df_filtered_after = df_param_after.copy()
                        # df_filtered_after = df_filtered_after.head(13)
                        if "Timestamp" in df_param_after.columns and selected_param in df_param_after.columns:
                            df_filtered_after = df_param_after[["Timestamp", selected_param]].copy()
                            df_filtered_after = df_filtered_after.head(13)
                            df_filtered_after["Timestamp"] = pd.to_datetime(df_filtered_after["Timestamp"]).dt.strftime("%Y-%m-%d")
                        data_table4 = df_filtered_after.to_dict("records")
                        columns4 = [{"name": col, "id": col} for col in df_filtered_after.columns]
                    else:
                        print(f"No files found for parameter {selected_param} year {selected_year} in {selected_province} Province")

                elif selected_param == "Jumlah_Penduduk_Miskin":
                    # Baca dengan multi-header
                    df_param = pd.read_excel(
                        selected_param_path,
                        sheet_name=sheet_name,
                        header=[0, 1, 2]
                    )
                    # Gabungkan multi-level columns menjadi satu string
                    df_param.columns = [
                        f"{str(col[1]).strip()} {str(col[2]).strip()}" if "Province" not in str(col[1]) else "Province"
                        for col in df_param.columns
                    ]
                    print("Combined results column:", df_param.columns.tolist())
                    # Cari kolom yang sesuai tahun dan bulan September
                    target_col = None
                    for col in df_param.columns:
                        if str(selected_year) in col and "September" in col:
                            target_col = col
                            break
                    if not target_col:
                        raise ValueError(f"Column for '{selected_year} September' Not found in dataset! \n"
                                        f"Column available: {df_param.columns.tolist()}")

                    # Ambil kolom yang diperlukan
                    df_filtered = df_param[["Province", target_col]].copy()
                    df_filtered.rename(columns={target_col: "Value"}, inplace=True)
                    df_filtered["Year"] = selected_year

                    # Open Folder File Data After Interpolasi:
                    folder_name = param_folder_mapping.get(selected_param)
                    full_path = os.path.join(base_path, folder_name, selected_province)

                    file_to_open = None
                    for file in os.listdir(full_path):
                        if file.endswith(".xlsx") and str(selected_year) in file:
                            file_to_open = os.path.join(full_path, file)
                            break
                    if file_to_open:
                        print("Opening file:", file_to_open)
                        df_param_after = pd.read_excel(file_to_open)
                        # df_filtered_after = df_param_after.copy()
                        # df_filtered_after = df_filtered_after.head(13)
                        if "Timestamp" in df_param_after.columns and selected_param in df_param_after.columns:
                            df_filtered_after = df_param_after[["Timestamp", selected_param]].copy()
                            df_filtered_after = df_filtered_after.head(13)
                            df_filtered_after["Timestamp"] = pd.to_datetime(df_filtered_after["Timestamp"]).dt.strftime("%Y-%m-%d")
                        data_table4 = df_filtered_after.to_dict("records")
                        columns4 = [{"name": col, "id": col} for col in df_filtered_after.columns]
                    else:
                        print(f"No files found for parameter {selected_param} year {selected_year} in {selected_province} Province")

                # Tangani dataset dengan struktur khusus untuk 2 parameter energi listrik (daya terpasang dan produksi listrik):
                elif selected_param in ["Daya_Terpasang", "Produksi_Listrik", "Listrik_Terjual"]:
                    df_param = pd.read_excel(
                        selected_param_path,
                        sheet_name=sheet_name,
                        header=[0, 1, 2]
                    )
                    # Gabungkan multi-level columns
                    df_param.columns = [
                        "Province" if "Province" in str(col[0]) else f"{str(col[1]).strip()} {str(col[2]).strip()}"
                        for col in df_param.columns
                    ]
                    df_param = df_param.drop(columns=[col for col in df_param.columns if "No" in col], errors="ignore")
                    # Mapping nama readable
                    readable_param = {
                        "Daya_Terpasang": "Daya Terpasang",
                        "Produksi_Listrik": "Produksi Listrik",
                        "Listrik_Terjual" : "Listrik Terjual"
                    }
                    param_label = readable_param[selected_param]
                    target_col = None
                    for col in df_param.columns:
                        if str(selected_year) in col and param_label in col:
                            target_col = col
                            break

                    if not target_col:
                        raise ValueError(f"No column found for parameter '{selected_param}' in {selected_year}")

                    df_filtered = df_param[["Province", target_col]].copy()
                    df_filtered["Year"] = selected_year
                    df_filtered.rename(columns={target_col: "Value"}, inplace=True)
                    # Membuka Folder File Data after Interpolation:
                    folder_name = param_folder_mapping.get(selected_param)
                    full_path = os.path.join(base_path, folder_name, selected_province)

                    file_to_open = None
                    for file in os.listdir(full_path):
                        if file.endswith(".xlsx") and str(selected_year) in file:
                            file_to_open = os.path.join(full_path, file)
                            break
                    if file_to_open:
                        print("Opening file:", file_to_open)
                        df_param_after = pd.read_excel(file_to_open)
                        # df_filtered_after = df_param_after.copy()
                        # df_filtered_after = df_filtered_after.head(13)
                        if "Timestamp" in df_param_after.columns and selected_param in df_param_after.columns:
                            df_filtered_after = df_param_after[["Timestamp", selected_param]].copy()
                            df_filtered_after = df_filtered_after.head(13)
                            df_filtered_after["Timestamp"] = pd.to_datetime(df_filtered_after["Timestamp"]).dt.strftime("%Y-%m-%d")
                        data_table4 = df_filtered_after.to_dict("records")
                        columns4 = [{"name": col, "id": col} for col in df_filtered_after.columns]
                    else:
                        print(f"No files found for parameter {selected_param} year {selected_year} in {selected_province} Province")

                else:
                    # 📌 Dataset standar lainnya, baca dari baris ke-3 (skip header dan 1 baris)
                    df_param = pd.read_excel(
                        selected_param_path,
                        sheet_name=sheet_name,
                        skiprows=1
                    )
                    # Hapus kolom 'No' jika ada
                    df_param = df_param.drop(columns=["No"], errors="ignore")
                    # Identifikasi kolom tahun (yang bentuknya 4 digit angka)
                    tahun_columns = [col for col in df_param.columns if re.match(r"^\d{4}$", str(col))]

                    # Konversi angka: hapus koma dan ubah ke float
                    for col in tahun_columns:
                        df_param[col] = (
                            df_param[col]
                            .astype(str)
                            .str.replace(",", "", regex=False)
                            .replace("nan", None)
                            .astype(float)
                        )
                    # Ubah ke long format
                    df_long = df_param.melt(id_vars=["Province"], var_name="Year", value_name="Value")
                    # Filter tahun numerik dan cast ke int
                    df_long = df_long[df_long["Year"].astype(str).str.match(r"^\d{4}$")]
                    df_long["Year"] = df_long["Year"].astype(int)
                    # Filter berdasarkan tahun
                    df_filtered = df_long[df_long["Year"] == selected_year]

                    # Code untuk membuka Folder file Data after Interpolation Data:
                    df_param_after = df_param.dropna(subset=["Nama Provinsi"])
                    df_long_after = pd.melt(df_param_after, id_vars=["Nama Provinsi"], var_name="Year", value_name="Value")
                    df_long_after["Year"] = df_long_after["Year"].astype(str).str.extract(r'(\d{4})').astype(int)
                    df_long_after["Value"] = pd.to_numeric(df_long_after["Value"], errors="coerce")

                    # Simpan hasil data dan kolom dari df_long (setelah interpolasi) ke data_table4 dan columns4
                    df_filtered2 = df_long_after[(df_long_after["Nama Provinsi"] == selected_province) & (df_long_after["Year"] == selected_year)]
                    # df_filtered2 = df_filtered2.head(13)
                    # ✅ Format Timestamp jika ada
                    if "Timestamp" in df_long_after.columns and selected_param in df_param_after.columns:
                        df_filtered2 = df_filtered2[["Timestamp", selected_param]].copy()
                        df_filtered2 = df_filtered2.head(13)
                        df_filtered2["Timestamp"] = pd.to_datetime(df_filtered2["Timestamp"]).dt.strftime("%Y-%m-%d")
                    data_table4 = df_filtered2.to_dict("records")
                    columns4 = [{"name": col, "id": col} for col in df_filtered2.columns]

                # Filter provinsi jika dipilih
                if selected_province:
                    df_filtered = df_filtered[df_filtered["Province"] == selected_province]
                    # df_filtered2 = df_filtered2[df_filtered2["Province"] == selected_province]
                # Batas 10 baris
                df_filtered = df_filtered.head(13)
                # df_filtered2 = df_filtered2.head(10)
                # Konversi ke format tabel
                data_table3 = df_filtered.to_dict("records")
                columns3 = [{"name": col, "id": col} for col in df_filtered.columns]
                data_table4 = df_filtered2.to_dict("records")
                columns4 = [{"name": col, "id": col} for col in df_filtered2.columns]

                print("Columns3:", columns3)
                print("DataTable3 Sample:", data_table3[:2])
                print("Columns4:", columns4)
                print("DataTable4 Sample:", data_table4[:2])
        except Exception as e:
            print(f"Failed to process sheet {selected_param}: {e}")

    return fig_treemap, data_table, columns, data_table2, columns2, {"display": "block"}, {"display": "block"}, data_table3, columns3, {"display": "block", "textAlign": "center"}, data_table4, columns4, {"display": "block", "textAlign": "center"}, {"display": "block"}

# Warning untuk Open Raw Dataset:
@callback(
    Output('open-warning', 'style'),
    Input('param-dropdown', 'value'),
    Input('submit-btn', 'n_clicks'),
    prevent_initial_call=True
)
def toggle_open_warning(selected_param, n_clicks):
    print("Selected param:", selected_param)
    print("Parameter options:", parameter_options)
    if selected_param in parameter_options and (n_clicks is None or n_clicks == 0):
        return {'color': 'red', 'fontWeight': 'bold', 'display': 'block'}
    elif selected_param is None and (n_clicks is None or n_clicks == 0):
        return {'color': 'red', 'fontWeight': 'bold', 'display': 'block'}
    elif selected_param is None and (n_clicks is None or n_clicks != 0):
        return {'color': 'red', 'fontWeight': 'bold', 'display': 'block'}
    return {'display': 'none'}
import dash
from dash import html, dcc, Input, Output, State, register_page, dash_table, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from joblib import load  # untuk model RF & XGBoost
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Add, Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from lib import attribute as att
import tensorflow as tf  # untuk model LSTM
import plotly.express as px
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib

register_page(__name__, path='/prediction')
# Dictionary untuk display model name di dropdown:
model_display_names = {
    "Linear Regression (Machine Learning)": "Linear Regression", #1
    "Random Forest (Machine Learning)": "Random Forest", #2
    "XGBoost (Machine Learning)": "XGBoost", #3
    "SVR (Machine Learning)": "SVR", #4
    "KNN (Machine Learning)": "KNN", #5
    "LightGBM (Machine Learning)": "LightGBM", #6
    "CatBoost (Machine Learning)": "CatBoost", #7
    "Extra Trees (Machine Learning)": "Extra Trees", #8
    "LSTM (Deep Learning)": "LSTM", #9
    "Transformer (Deep Learning)": "Transformer", #10
    "GRU (Deep Learning)": "GRU", #11
    "CNN-1D (Deep Learning)": "CNN-1D", #12
    "MLP (Deep Learning)": "MLP" #13
}
# Load dataset parquet
df = att.data_all

# Layout utama sesuai template
layout = dbc.Container([
    html.Br(),
    html.H4("Prediction and Visualization", className="text-center border border-dark p-2 fw-bold"),
    html.Br(),
    html.Div([
            dbc.Row([
                html.Div(
                    id='year-slider-warning',
                    children="⚠️ To display the matrix evaluation model (RMSE, MAE, and R²), select the year slider starting from 2021!",
                    style={'color': 'red', 'fontWeight': 'bold', 'display': 'none'}
                ),

                dbc.Col(dcc.RangeSlider(2021, 2026, 1, value=[2021, 2026],
                                        marks={i: str(i) for i in range(2021, 2027)},
                                        id='year-slider'), width=14)
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col(dcc.Dropdown(id='province-dropdown',
                                    options=[{'label': i, 'value': i} for i in df['Province'].unique()],
                                    placeholder='Choose Province',
                                    style={"textAlign": "center"}
                                    ), width=3),
                                    
                dbc.Col(dcc.Dropdown(id='frequency-dropdown',
                                    options=[
                                        {'label': 'Daily', 'value': 'daily'},
                                        {'label': 'Weekly', 'value': 'weekly'},
                                        {'label': 'Monthly', 'value': 'monthly'}
                                    ],
                                    placeholder='Choose Frequency',
                                    style={"textAlign": "center"}
                                    ), width=3),
                dbc.Col(dcc.Dropdown(id='model-dropdown',
                                    options=[{'label': m, 'value': m} for m in model_display_names.keys()],
                                    placeholder='Choose Model',
                                    style={"textAlign": "center"}
                                    ), width=5),
                dbc.Col(html.Button("Predict", id='predict-btn', className="btn btn-success", style={"textAlign": "center"}), width=1),
            ], justify='center'),
    ]),

    html.Br(),
    # Map by Selected Province:
    dbc.Row([dbc.Col(dbc.Card(dcc.Graph(id="map-graph-predict", figure=att.get_indonesia_map(), style={"width": "100%", "height": "350px"}), body=True, style={"opacity": "0.5"}), width=12)]),
    html.Br(),
    # Line Chart Prediction Demand Energy with All Parameter/Attribute:
    dbc.Row([
        dbc.Col(dcc.Graph(id='prediction-line-chart'), width=12)
    ]),
    html.Br(),
    # Matrix Evaluation Table with All Parameter/Attribute
    dbc.Table(
        children=[
            dbc.Row(
                dbc.Col(
                    html.H6("Matrix Evaluation Table using All Parameter", className="fw-bold"), width=12), className="text-center mt-4"
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Table(
                        id="eval-metrics", bordered=True, striped=True, hover=True,
                    ), width=12
                )
            )
        ]
    ),
    html.Br(),
    # Line Chart Prediction Demand Energy with Top 5 Highest Importance Parameter/Attribute:
    dbc.Row([
        dbc.Col(dcc.Graph(id='prediction-line-chart-top5'), width=12)
    ]),
    html.Br(),
    # Matrix Evaluation Table with Top 5 Highest Importance Parameter/Attribute:
    dbc.Table(
        children=[
            dbc.Row(
                dbc.Col(
                    html.H6("Matrix Evaluation Table using Top 5 Highest Important Parameter", className="fw-bold"), width=12), className="text-center mt-4"
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Table(
                        id="eval-metrics-top5", bordered=True, striped=True, hover=True,
                    ), width=12
                )
            )
        ]
    ),
    html.Br(),
    # Card Demografi dan PDRB:
    dbc.Row([
        dbc.Col(
        dbc.Card([
            dbc.CardHeader("Prediction of Total Population"),
            dbc.CardBody([
                html.H3(id='card-pred-jumlah-penduduk', className='p-3'),
            ]),
            ], className='text-center',color='light'),
            width=4,
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Prediction of Total Poor People"),
                dbc.CardBody([
                    html.H3(id='card-pred-jumlah-penduduk-miskin', className='p-3'),
                ]),
            ], className='text-center', color='light'),
            width=4,
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Prediction of Gross Regional Domestic Product (PDRB)"),
                dbc.CardBody([
                    html.H3(id='card-pred-pdrb', className='p-3'),
                ]),
            ], className='text-center', color='light'),width=4,
        ),
    ]),
    html.Br(),
])
@callback(
    [
        Output('map-graph-predict', 'figure'),
        Output('prediction-line-chart', 'figure'),
        Output('eval-metrics', 'children'),
        Output('prediction-line-chart-top5', 'figure'),
        Output('eval-metrics-top5', 'children'),
        Output('card-pred-jumlah-penduduk', 'children'),
        Output('card-pred-jumlah-penduduk-miskin', 'children'),
        Output('card-pred-pdrb', 'children'),
    ],
    Input("predict-btn", "n_clicks"),
    [
        State("province-dropdown", "value"),
        State("frequency-dropdown", "value"),
        State("model-dropdown", "value"),
        State("year-slider", "value")
    ]
)
def update_prediction(n_clicks,selected_province, selected_freq, selected_model, year_range):
    # Pastikan tombol prediksi sudah diklik dan semua input sudah dipilih
    if not n_clicks:
        return att.get_indonesia_map(), go.Figure(), "", go.Figure(), "", "", "", ""
    if not selected_province or not selected_model or not year_range:
        return att.get_indonesia_map(), go.Figure(), "Please fill in and complete all input dropdowns and year sliders!", go.Figure(), "", "", "", ""
    
    df_filtered = df[(df["Province"] == selected_province)]
    if df_filtered.empty:
        return att.get_indonesia_map(), go.Figure(), "Data is not available for that filter, please try again!", go.Figure(), "", "", "", ""

    selected_display_name = selected_model  # misal dari Dash callback
    actual_model_name = model_display_names.get(selected_display_name)
    # buat list model untuk predict dan train evaluate model:
    model = att.load_model_dynamic(selected_province, actual_model_name)
    # buat list model untuk Top 5 Highest Parameter/Features:
    model_top5 = att.load_model_dynamic_top5(selected_province, actual_model_name)

    model_map_predict = att.load_model_dynamic_predict(selected_province, actual_model_name)
    # Lakukan prediksi jika tombol klik valid
    df_filtered = df_filtered.sort_values(by='Year')
    train_df = df_filtered[df_filtered['Year'] <= 2023].copy()
    test_df = df_filtered[df_filtered['Year'] >= year_range[0]]

    train_df2 = df_filtered[df_filtered['Year'] <= 2023].copy()
    test_df2 = df_filtered[df_filtered['Year'] >= year_range[0]]

    # khusus utk menampilkan map_predict_figure saja:
    train_df3 = df_filtered[df_filtered['Year'] <= 2023].copy()
    test_df3 = df_filtered[df_filtered['Year'] >= year_range[0]]

    # Batas data historis untuk Line Prediction Chart All Parameter:
    feature_cols = ['Temperature', 'Jumlah_Penduduk', 'Jumlah_Penduduk_Miskin', 'PDRB', 'Jumlah_Pelanggan_Listrik', 'Listrik_Terjual', 
        'Daya_Terpasang', 'Produksi_Listrik', 'Persentase_Penduduk_Miskin']
    scaler = StandardScaler()
    X_train = train_df[feature_cols]
    y_train = train_df['Demand']
    scaler.fit(X_train)

    # Batas data historis untuk Line Prediction Chart Top 5 Parameter:
    feature_cols2 = ['Temperature', 'Jumlah_Penduduk', 'Jumlah_Penduduk_Miskin', 'Jumlah_Pelanggan_Listrik', 'Persentase_Penduduk_Miskin']
    X_train2 = train_df2[feature_cols2]
    y_train2 = train_df2['Demand']
    scaler2 = StandardScaler()
    scaler2.fit(X_train2)

    # khusus utk menampilkan map figure pada year range 2021-2026 atau dari tahun 2021 sampai tahun nya lebih dari > 2023:
    feature_cols3 = ['Temperature', 'Jumlah_Penduduk', 'Jumlah_Penduduk_Miskin', 'PDRB', 'Jumlah_Pelanggan_Listrik', 'Listrik_Terjual', 
        'Daya_Terpasang', 'Produksi_Listrik', 'Persentase_Penduduk_Miskin','Longitude', 'Latitude']
    scaler3 = StandardScaler()
    X_train3 = train_df3[feature_cols3]
    y_train3 = train_df3['Demand']
    scaler3.fit(X_train3)

    # Jika tahun prediksi > 2023 tapi slider year dari tahun 2021-2023 → Actual data untuk train (2021–2023) and tahun > 2023 untuk predict (2024-2030)
    if year_range[0] <= 2023 and year_range[1] > 2023:
        location_coords = df_filtered[["Province", "Regency", "Latitude", "Longitude"]].drop_duplicates()
        selected_year = year_range[-1]
        # Pastikan kolom Date datetime
        if train_df["Date"].dtype == "O":
            train_df["Date"] = pd.to_datetime(train_df["Date"], errors='coerce')

        # --- Data historis (2021–2023) --- untuk All Parameter :
        historical_df = train_df[(train_df["Date"].dt.year >= year_range[0]) & (train_df["Date"].dt.year <= 2023)].copy()
        # --- Data historis (2021–2023) --- untuk Top 5 Parameter :
        historical_df2 = train_df2[(train_df2["Date"].dt.year >= year_range[0]) & (train_df2["Date"].dt.year <= 2023)].copy()
        # --- Data historis (2021–2023) --- untuk menampilkan map predict figure:
        historical_df3 = train_df3[(train_df3["Date"].dt.year >= year_range[0]) & (train_df3["Date"].dt.year <= 2023)].copy()

        X_hist = historical_df[feature_cols]
        y_hist = historical_df["Demand"]
        scaler.fit(X_hist)
        X_hist_scaled = scaler.transform(X_hist)

        # Train model
        # Prediction for year 2021–2022 only → for matrix evaluation table (using actual y_test_eval)
        if actual_model_name in ["Linear Regression", "Random Forest", "XGBoost", "KNN","LightGBM","CatBoost","Extra Trees","SVR"]:
            model.fit(X_hist_scaled, y_hist)
        elif actual_model_name in ["LSTM", "CNN-1D", "GRU", "MLP"]:
            model.fit(X_hist_scaled, y_hist)
        elif actual_model_name == "Transformer":
            model.fit(X_hist_scaled, y_hist)

        # Evaluasi model pada data historis 2021–2023
        y_test_eval = historical_df["Demand"].values  # actual demand
        y_pred_eval = model.predict(X_hist_scaled)    # predicted demand
        
        # Prediksi masa depan (2024 dst)
        # Copy bagian future_df generation dari blok `if year_range[0] > 2023` ⬇️
        base_year = 2023
        n_years = year_range[1] - 2023

        if selected_freq == "monthly":
            n_periods = n_years * 12
            freq = "MS"
        elif selected_freq == "weekly":
            n_periods = n_years * 52
            freq = "W-SUN"
        elif selected_freq == "daily":
            n_periods = n_years * 365
            freq = "D"
        else:
            raise ValueError("Invalid frequency")

        date_range = pd.date_range(start="2024-01-01", periods=n_periods, freq=freq)
        future_df = pd.DataFrame({"Date": date_range})
        future_df["Year"] = future_df["Date"].dt.year
        future_df["Province"] = selected_province

        # Generate fitur masa depan sama seperti sebelumnya untuk All Parameter/Features:
        mean_hourly_values = historical_df[feature_cols].mean()
        np.random.seed(42)
        future_df["Temperature"] = mean_hourly_values["Temperature"] + np.random.normal(loc=0, scale=1.5, size=n_periods)
        future_df["Jumlah_Penduduk"] = [mean_hourly_values["Jumlah_Penduduk"] * ((1 + 0.05) ** (y - base_year)) for y in future_df["Year"]]
        future_df["Jumlah_Penduduk_Miskin"] = [mean_hourly_values["Jumlah_Penduduk_Miskin"] * ((1 - 0.05) ** (y - base_year)) for y in future_df["Year"]]
        future_df["PDRB"] = [mean_hourly_values["PDRB"] * ((1 + 0.05) ** (y - base_year)) for y in future_df["Year"]]
        future_df["Jumlah_Pelanggan_Listrik"] = [mean_hourly_values["Jumlah_Pelanggan_Listrik"] * ((1 + 0.05) ** (y - base_year)) for y in future_df["Year"]]
        start_val = mean_hourly_values["Listrik_Terjual"] * ((1 + 0.05) ** (2024 - base_year))
        week_growth = 0.001
        future_df["Listrik_Terjual"] = [start_val * ((1 + week_growth) ** i) for i in range(n_periods)]
        start_val_daya = mean_hourly_values["Daya_Terpasang"] * ((1 + 0.05) ** (2024 - base_year))
        future_df["Daya_Terpasang"] = [start_val_daya * ((1 + week_growth) ** i) for i in range(n_periods)]
        start_val_produksi = mean_hourly_values["Produksi_Listrik"] * ((1 + 0.05) ** (2024 - base_year))
        future_df["Produksi_Listrik"] = [start_val_produksi * ((1 + week_growth) ** i) for i in range(n_periods)]
        future_df["Persentase_Penduduk_Miskin"] = mean_hourly_values["Persentase_Penduduk_Miskin"]
        future_df["Month"] = future_df["Date"].dt.month
        future_df["Week"] = future_df["Date"].dt.isocalendar().week.astype(int)
        future_df["DayOfYear"] = future_df["Date"].dt.dayofyear

        # Seasonal Factor
        if selected_freq == "monthly":
            seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * future_df["Month"] / 12)
        elif selected_freq == "weekly":
            seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * future_df["Week"] / 52)
        elif selected_freq == "daily":
            seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * future_df["DayOfYear"] / 365)

        for col in ["Listrik_Terjual", "Daya_Terpasang", "Produksi_Listrik"]:
            future_df[col] *= seasonal_factor

        future_df = pd.concat([future_df.assign(Regency=reg) for reg in location_coords["Regency"].unique()], ignore_index=True)
        future_df = future_df.merge(location_coords, on=["Province", "Regency"], how="left")
        # Prediksi untuk prediksi demand setelah tahun > 2023:
        X_future = scaler.transform(future_df[feature_cols])
        if actual_model_name in ["Linear Regression", "Random Forest", "XGBoost", "KNN","LightGBM","CatBoost","Extra Trees","SVR"]:
            y_pred_future = model.predict(X_future)
        elif actual_model_name in ["LSTM", "CNN-1D", "GRU", "MLP"]:
            print("X_future type:", type(X_future))
            print("X_future shape:", getattr(X_future, 'shape', 'No shape'))
            print("First few rows of X_future:", X_future[:2] if hasattr(X_future, '__getitem__') else "Not indexable")
            X_future_reshaped = X_future.reshape((X_future.shape[0], X_future.shape[1]))
            y_pred_future = model.predict(X_future_reshaped)
        else:
            y_pred_future = model.predict(X_future).flatten() # NO expand_dims

        # Untuk mengukur nilai RMSE, MAE, dan R² saja:
        if actual_model_name in ["Linear Regression", "Random Forest", "XGBoost", "KNN","LightGBM","CatBoost","Extra Trees","SVR"]:
            y_pred_eval = model.predict(X_hist_scaled)
        elif actual_model_name in ["LSTM", "CNN-1D", "GRU", "MLP"]:
            X_hist_reshaped = X_hist_scaled.reshape((X_hist_scaled.shape[0], X_hist_scaled.shape[1]))
            y_pred_eval = model.predict(X_hist_reshaped)
        else:
            y_pred_eval = model.predict(X_hist_scaled).flatten() # NO expand_dims

        # Evaluasi metrik
        rmse = root_mean_squared_error(y_test_eval, y_pred_eval)
        mae = mean_absolute_error(y_test_eval, y_pred_eval)
        r2 = r2_score(y_test_eval, y_pred_eval)

        # Tampilkan hasil evaluasi sebagai tabel (html.Tbody)
        eval_table = html.Tbody([
            html.Tr([
                html.Td("Root Mean Squared Error (RMSE)"),
                html.Td("=", className="text-center"),
                html.Td(f"{rmse:.2f}", className="text-center")
            ]),
            html.Tr([
                html.Td("Mean Absolute Error (MAE)"),
                html.Td("=", className="text-center"),
                html.Td(f"{mae:.2f}", className="text-center")
            ]),
            html.Tr([
                html.Td("R² Score"),
                html.Td("=", className="text-center"),
                html.Td(f"{r2:.2f}", className="text-center")
            ]),
        ])

        future_df["Prediction"] = y_pred_future.flatten() if hasattr(y_pred_future, 'flatten') else y_pred_future

        # Output kartu indikator (akhir tahun terakhir)
        pred_population = future_df[future_df["Year"] == year_range[1]]['Jumlah_Penduduk'].iloc[-1]
        pred_poverty = future_df[future_df["Year"] == year_range[1]]['Jumlah_Penduduk_Miskin'].iloc[-1]
        pred_pdrb = future_df[future_df["Year"] == year_range[1]]['PDRB'].iloc[-1]

        # --- Siapkan data prediksi historis (2021–2023) ---
        historical_df_plot = historical_df.copy()
        historical_df_plot['Predicted_Demand'] = y_pred_eval

        # --- Siapkan data prediksi masa depan (2024 dst) ---
        future_df_plot = future_df.copy()
        future_df_plot['Predicted_Demand'] = y_pred_future.flatten() if hasattr(y_pred_future, 'flatten') else y_pred_future

        # Gabungkan data historis dan masa depan untuk prediksi keseluruhan
        combined_df = pd.concat([historical_df_plot, future_df_plot], axis=0)
        combined_df.set_index('Date', inplace=True)

        # Pilih kolom numerik untuk resampling
        numeric_cols = combined_df.select_dtypes(include='number').columns

        # Resampling berdasarkan frekuensi
        # if selected_freq == 'weekly':
        #     combined_resampled = combined_df[numeric_cols].resample('W-SUN').mean()
        # elif selected_freq == 'monthly':
        #     combined_resampled = combined_df[numeric_cols].resample('MS').mean()
        # elif selected_freq == 'daily':
        #     combined_resampled = combined_df[numeric_cols].resample('D').mean()
        # else:
        #     combined_resampled = combined_df[numeric_cols].copy()

        if selected_freq == 'monthly':
            resample_code = 'MS'  # match with future_df generation
        elif selected_freq == 'weekly':
            resample_code = 'W-SUN'  # match with future_df generation
        elif selected_freq == 'daily':
            resample_code = 'D'
        else:
            resample_code = selected_freq[0].upper()  # fallback
        # combined_resampled.reset_index(inplace=True)

        # --- PLOTTING ---
        fig_line = go.Figure()

        # 1. Actual Demand (2021–2023)
        actual_df = historical_df.copy()
        actual_df['Date'] = pd.to_datetime(actual_df['Date'])
        actual_df.set_index('Date', inplace=True)
        actual_resampled = actual_df[['Demand']].resample(resample_code).mean().reset_index()
        fig_line.add_trace(go.Scatter(
            x=actual_resampled['Date'], y=actual_resampled['Demand'],
            mode='lines', name='Actual Electricity Demand', line=dict(color='red')
        ))
        # 2. Modelled Demand (prediksi hasil training model 2021–2023)
        modelled_df = historical_df_plot.copy()
        modelled_df['Date'] = pd.to_datetime(modelled_df['Date'])
        modelled_df.set_index('Date', inplace=True)
        modelled_resampled = modelled_df[['Predicted_Demand']].resample(resample_code).mean().reset_index()
        fig_line.add_trace(go.Scatter(
            x=modelled_resampled['Date'], y=modelled_resampled['Predicted_Demand'],
            mode='lines', name='Modelled Electricity Demand', line=dict(color='blue', dash='solid')
        ))

        # 3. Predicted Demand (2024 dst)
        predicted_df = future_df_plot.copy()
        predicted_df['Date'] = pd.to_datetime(predicted_df['Date'])
        predicted_df.set_index('Date', inplace=True)
        predicted_resampled = predicted_df[['Predicted_Demand']].resample(resample_code).mean().reset_index()
        fig_line.add_trace(go.Scatter(
            x=predicted_resampled['Date'], y=predicted_resampled['Predicted_Demand'],
            mode='lines', name='Predicted Electricity Demand', line=dict(color='black', dash='dash')
        ))
        # Layout
        fig_line.update_layout(
            title='Actual vs Modelled vs Predicted Electricity Demand using All Important Parameter (in KWh)',
            title_x=0.5,
            xaxis_title='Date',
            yaxis_title='Electricity Demand (KWh)',
            height=500,
            xaxis=dict(showgrid=True, color='black', showline=True, linecolor='black', linewidth=2, tickfont=dict(size=14), titlefont=dict(size=16)),
            yaxis=dict(showgrid=True, color='black', showline=True, linecolor='black', linewidth=2, tickfont=dict(size=14), titlefont=dict(size=16)),
            legend=dict(x=1.1, y=1.15, xanchor='right', yanchor='top', bgcolor='snow'),
            margin=dict(l=40, r=40, t=100, b=60),
            plot_bgcolor='snow',
            paper_bgcolor='snow',
            font=dict(color='black')
        )

        # Untuk Top 5 Highest Parameter/Features:
        X_hist2 = historical_df2[feature_cols2]
        y_hist2 = historical_df2["Demand"]
        scaler2.fit(X_hist2)
        X_hist_scaled2 = scaler2.transform(X_hist2)
        y_test_eval2 = historical_df2["Demand"].values  # actual demand
        # Train model
        if actual_model_name in ["Linear Regression", "Random Forest", "XGBoost", "KNN","LightGBM","CatBoost","Extra Trees","SVR"]:
            model_top5.fit(X_hist_scaled2, y_hist2)
            # y_pred = model.predict(X_test_eval)
        elif actual_model_name in ["LSTM", "CNN-1D", "GRU", "MLP"]:
            model_top5.fit(X_hist_scaled2, y_hist2)
            # X_test_seq = np.expand_dims(X_test_eval, axis=-1)  # Ini penting! untuk model DL selain Transformer, karena TransformerBlock nya customize bukan bawaan library.
            # y_pred = model.predict(X_test_seq).flatten()
        elif actual_model_name == "Transformer":
            model_top5.fit(X_hist_scaled2, y_hist2)
        
        y_pred_eval2 = model_top5.predict(X_hist_scaled2)    # predicted demand

        # Prediksi masa depan (2024 dst)
        # Copy bagian future_df generation dari blok `if year_range[0] > 2023` ⬇️
        base_year = 2023
        n_years2 = year_range[1] - 2023

        if selected_freq == "monthly":
            n_periods2 = n_years2 * 12
            freq2 = "MS"
        elif selected_freq == "weekly":
            n_periods2 = n_years2 * 52
            freq2 = "W-SUN"
        elif selected_freq == "daily":
            n_periods2 = n_years2 * 365
            freq2 = "D"
        else:
            raise ValueError("Invalid frequency")

        date_range = pd.date_range(start="2024-01-01", periods=n_periods2, freq=freq2)
        future_df2 = pd.DataFrame({"Date": date_range})
        future_df2["Year"] = future_df2["Date"].dt.year
        future_df2["Province"] = selected_province

        # Generate fitur masa depan sama seperti sebelumnya untuk Top 5 Highest Parameter/Features:
        mean_hourly_values2 = historical_df2[feature_cols2].mean()
        np.random.seed(42)
        future_df2["Temperature"] = mean_hourly_values2["Temperature"] + np.random.normal(loc=0, scale=1.5, size=n_periods2)
        future_df2["Jumlah_Penduduk"] = [mean_hourly_values2["Jumlah_Penduduk"] * ((1 + 0.05) ** (y - base_year)) for y in future_df2["Year"]]
        # future_df2["PDRB"] = [mean_hourly_values2["PDRB"] * ((1 + 0.05) ** (y - base_year)) for y in future_df2["Year"]]
        future_df2["Jumlah_Penduduk_Miskin"] = [mean_hourly_values2["Jumlah_Penduduk_Miskin"] * ((1 - 0.05) ** (y - base_year)) for y in future_df2["Year"]]
        future_df2["Jumlah_Pelanggan_Listrik"] = [mean_hourly_values2["Jumlah_Pelanggan_Listrik"] * ((1 + 0.05) ** (y - base_year)) for y in future_df2["Year"]]
        # start_val2 = mean_hourly_values2["Listrik_Terjual"] * ((1 + 0.05) ** (2024 - base_year))
        # week_growth2 = 0.001
        # future_df2["Listrik_Terjual"] = [start_val2 * ((1 + week_growth2) ** i) for i in range(n_periods2)]
        # start_val_daya2 = mean_hourly_values2["Daya_Terpasang"] * ((1 + 0.05) ** (2024 - base_year))
        # future_df2["Daya_Terpasang"] = [start_val_daya2 * ((1 + week_growth2) ** i) for i in range(n_periods2)]
        # start_val_produksi2 = mean_hourly_values2["Produksi_Listrik"] * ((1 + 0.05) ** (2024 - base_year))
        # future_df2["Produksi_Listrik"] = [start_val_produksi2 * ((1 + week_growth2) ** i) for i in range(n_periods2)]
        future_df2["Persentase_Penduduk_Miskin"] = mean_hourly_values["Persentase_Penduduk_Miskin"]
        future_df2["Month"] = future_df2["Date"].dt.month
        future_df2["Week"] = future_df2["Date"].dt.isocalendar().week.astype(int)
        future_df2["DayOfYear"] = future_df2["Date"].dt.dayofyear

        future_df2 = pd.concat([future_df2.assign(Regency=reg) for reg in location_coords["Regency"].unique()], ignore_index=True)
        future_df2 = future_df2.merge(location_coords, on=["Province", "Regency"], how="left")

        missing = set(feature_cols2) - set(future_df.columns)
        if missing:
            print("⚠️ Fitur ini tidak tersedia di future_df:", missing)
        # Prediksi
        X_future2 = scaler2.transform(future_df2[feature_cols2])
        # Untuk memprediksi demand electricity setelah tahun > 2023:
        if actual_model_name in ["Linear Regression", "Random Forest", "XGBoost", "KNN","LightGBM","CatBoost","Extra Trees","SVR"]:
            y_pred_future2 = model_top5.predict(X_future2)
        elif actual_model_name in ["LSTM", "CNN-1D", "GRU", "MLP"]:
            # Paksa reshape ke (samples, timesteps=5, features=1)
            X_future_reshaped2 = X_future2.reshape((X_future2.shape[0], X_future2.shape[1]))
            print("Model input shape:", model_top5.input_shape)
            print("X_future2 shape:", X_future2.shape)
            print("X_future_reshaped2 shape:", X_future_reshaped2.shape)
            y_pred_future2 = model_top5.predict(X_future_reshaped2)
        else:
            y_pred_future2 = model_top5.predict(X_future2).flatten() # NO expand_dims

        # Untuk mengukur nilai RMSE, MAE, dan R² saja:
        if actual_model_name in ["Linear Regression", "Random Forest", "XGBoost", "KNN","LightGBM","CatBoost","Extra Trees","SVR"]:
            y_pred_eval2 = model_top5.predict(X_hist_scaled2)
        elif actual_model_name in ["LSTM", "CNN-1D", "GRU", "MLP"]:
            X_hist_reshaped2 = X_hist_scaled2.reshape((X_hist_scaled2.shape[0], X_hist_scaled2.shape[1]))
            y_pred_eval2 = model_top5.predict(X_hist_reshaped2)
        else:
            y_pred_eval2 = model_top5.predict(X_hist_scaled2).flatten() # NO expand_dims

        # Evaluasi metrik
        rmse_2 = root_mean_squared_error(y_test_eval2, y_pred_eval2)
        mae_2 = mean_absolute_error(y_test_eval2, y_pred_eval2)
        r2_2 = r2_score(y_test_eval2, y_pred_eval2)

        # Tampilkan hasil evaluasi sebagai tabel (html.Tbody)
        eval_table2 = html.Tbody([
            html.Tr([
                html.Td("Root Mean Squared Error (RMSE)"),
                html.Td("=", className="text-center"),
                html.Td(f"{rmse_2:.2f}", className="text-center")
            ]),
            html.Tr([
                html.Td("Mean Absolute Error (MAE)"),
                html.Td("=", className="text-center"),
                html.Td(f"{mae_2:.2f}", className="text-center")
            ]),
            html.Tr([
                html.Td("R² Score"),
                html.Td("=", className="text-center"),
                html.Td(f"{r2_2:.2f}", className="text-center")
            ]),
        ])
        
        future_df2["Prediction"] = y_pred_future2.flatten() if hasattr(y_pred_future2, 'flatten') else y_pred_future2
        # --- Siapkan data prediksi historis (2021–2023) ---
        historical_df_plot2 = historical_df2.copy()
        historical_df_plot2['Predicted_Demand'] = y_pred_eval2
        # --- Siapkan data prediksi masa depan (2024 dst) ---
        future_df_plot2 = future_df2.copy()
        future_df_plot2['Predicted_Demand'] = y_pred_future2.flatten() if hasattr(y_pred_future2, 'flatten') else y_pred_future2

        # Gabungkan data historis dan masa depan untuk prediksi keseluruhan
        combined_df2 = pd.concat([historical_df_plot2, future_df_plot2], axis=0)
        combined_df2.set_index('Date', inplace=True)

        # Pilih kolom numerik untuk resampling
        numeric_columns2 = combined_df2.select_dtypes(include='number').columns

        # Resampling berdasarkan frekuensi
        # if selected_freq == 'weekly':
        #     combined_resampled2 = combined_df2[numeric_columns2].resample('W-SUN').mean()
        # elif selected_freq == 'monthly':
        #     combined_resampled2 = combined_df2[numeric_columns2].resample('MS').mean()
        # elif selected_freq == 'daily':
        #     combined_resampled2 = combined_df2[numeric_columns2].resample('D').mean()
        # else:
        #     combined_resampled2 = combined_df2[numeric_columns2].copy()

        if selected_freq == 'monthly':
            resample_code2 = 'MS'  # match with future_df generation
        elif selected_freq == 'weekly':
            resample_code2 = 'W-SUN'  # match with future_df generation
        elif selected_freq == 'daily':
            resample_code2 = 'D'
        else:
            resample_code2 = selected_freq[0].upper()  # fallback

        # combined_resampled2.reset_index(inplace=True)

        # --- PLOTTING ---
        fig_line2 = go.Figure()

        # 1. Actual Demand (2021–2023)
        actual_df2 = historical_df2.copy()
        actual_df2['Date'] = pd.to_datetime(actual_df2['Date'])
        actual_df2.set_index('Date', inplace=True)
        actual_resampled2 = actual_df2[['Demand']].resample(resample_code2).mean().reset_index()
        fig_line2.add_trace(go.Scatter(
            x=actual_resampled2['Date'], y=actual_resampled2['Demand'],
            mode='lines', name='Actual Electricity Demand', line=dict(color='red')
        ))

        # 2. Modelled Demand (prediksi hasil training model 2021–2023)
        modelled_df2 = historical_df_plot2.copy()
        modelled_df2['Date'] = pd.to_datetime(modelled_df2['Date'])
        modelled_df2.set_index('Date', inplace=True)
        modelled_resampled2 = modelled_df2[['Predicted_Demand']].resample(resample_code2).mean().reset_index()
        fig_line2.add_trace(go.Scatter(
            x=modelled_resampled2['Date'], y=modelled_resampled2['Predicted_Demand'],
            mode='lines', name='Modelled Electricity Demand', line=dict(color='blue', dash='solid')
        ))

        # 3. Predicted Demand (2024 dst)
        predicted_df2 = future_df_plot2.copy()
        predicted_df2['Date'] = pd.to_datetime(predicted_df2['Date'])
        predicted_df2.set_index('Date', inplace=True)
        predicted_resampled2 = predicted_df2[['Predicted_Demand']].resample(resample_code2).mean().reset_index()
        fig_line2.add_trace(go.Scatter(
            x=predicted_resampled2['Date'], y=predicted_resampled2['Predicted_Demand'],
            mode='lines', name='Predicted Electricity Demand', line=dict(color='black', dash='dash')
        ))
        # Layout
        fig_line2.update_layout(
            title='Actual vs Modelled vs Predicted Electricity Demand using Top 5 Highest Important Parameter (in KWh)',
            title_x=0.5,
            xaxis_title='Date',
            yaxis_title='Electricity Demand (KWh)',
            height=500,
            xaxis=dict(showgrid=True, color='black', showline=True, linecolor='black', linewidth=2, tickfont=dict(size=14), titlefont=dict(size=16)),
            yaxis=dict(showgrid=True, color='black', showline=True, linecolor='black', linewidth=2, tickfont=dict(size=14), titlefont=dict(size=16)),
            legend=dict(x=1.1, y=1.15, xanchor='right', yanchor='top', bgcolor='snow'),
            margin=dict(l=40, r=40, t=100, b=60),
            plot_bgcolor='snow',
            paper_bgcolor='snow',
            font=dict(color='black')
        )
       
        X_hist3 = historical_df3[feature_cols3]
        y_hist3 = historical_df3["Demand"]
        scaler3.fit(X_hist3)
        X_hist_scaled3 = scaler3.transform(X_hist3)

        # Train model
        # Prediction for year 2021–2022 only → for matrix evaluation table (using actual y_test_eval)
        if actual_model_name in ["Linear Regression", "Random Forest", "XGBoost", "KNN","LightGBM","CatBoost","Extra Trees","SVR"]:
            model_map_predict.fit(X_hist_scaled3, y_hist3)
        elif actual_model_name in ["LSTM", "CNN-1D", "GRU", "MLP"]:
            #berhasil modified model sequential MLP:
            if actual_model_name == "MLP":
                input_dim = X_hist_scaled3.shape[1]
                model_map_predict = Sequential([
                    Dense(64, activation='relu', input_shape=(input_dim,)),
                    Dense(64, activation='relu'),
                    Dense(1)
                ])
                model_map_predict.compile(optimizer=Adam(learning_rate=0.005), loss='mse')
                model_map_predict.fit(X_hist_scaled3, y_hist3)
            elif actual_model_name == "CNN-1D":
                # Reshape to 3D for Conv1D: (samples, timesteps, features)
                X_hist_cnn = X_hist_scaled3.reshape((X_hist_scaled3.shape[0], X_hist_scaled3.shape[1], 1))
                model_map_predict = Sequential([
                    Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(X_hist_cnn.shape[1], 1)),
                    MaxPooling1D(pool_size=2),
                    Flatten(),
                    Dense(50, activation='relu'),
                    Dense(1)
                ])
                model_map_predict.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
                model_map_predict.fit(X_hist_cnn, y_hist3)
            else:
                model_map_predict.fit(X_hist_scaled3, y_hist3)
        elif actual_model_name == "Transformer":
            X_hist3 = historical_df3[feature_cols3]  # hanya 11 kolom
            y_hist3 = historical_df3["Demand"]
            X_hist_scaled3 = scaler3.transform(X_hist3) 
            X_hist_trans = X_hist_scaled3.reshape((X_hist_scaled3.shape[0], 11, 1))
            model_map_predict.fit(X_hist_trans, y_hist3)
        
        # Prediksi masa depan (2024 dst)
        # Copy bagian future_df generation dari blok `if year_range[0] > 2023` ⬇️
        base_year = 2023
        n_years3 = year_range[1] - 2023

        if selected_freq == "monthly":
            n_periods3 = n_years3 * 12
            freq3 = "MS"
        elif selected_freq == "weekly":
            n_periods3 = n_years3 * 52
            freq3 = "W-SUN"
        elif selected_freq == "daily":
            n_periods3 = n_years3 * 365
            freq3 = "D"
        else:
            raise ValueError("Invalid frequency")

        date_range3 = pd.date_range(start="2024-01-01", periods=n_periods3, freq=freq3)
        future_df3 = pd.DataFrame({"Date": date_range3})
        future_df3["Year"] = future_df3["Date"].dt.year
        future_df3["Province"] = selected_province

        # Generate fitur masa depan sama seperti sebelumnya untuk All Parameter/Features:
        mean_hourly_values3 = historical_df3[feature_cols3].mean()
        np.random.seed(42)
        future_df3["Temperature"] = mean_hourly_values3["Temperature"] + np.random.normal(loc=0, scale=1.5, size=n_periods3)
        future_df3["Jumlah_Penduduk"] = [mean_hourly_values3["Jumlah_Penduduk"] * ((1 + 0.05) ** (y - base_year)) for y in future_df3["Year"]]
        future_df3["Jumlah_Penduduk_Miskin"] = [mean_hourly_values3["Jumlah_Penduduk_Miskin"] * ((1 - 0.05) ** (y - base_year)) for y in future_df3["Year"]]
        future_df3["PDRB"] = [mean_hourly_values3["PDRB"] * ((1 + 0.05) ** (y - base_year)) for y in future_df3["Year"]]
        future_df3["Jumlah_Pelanggan_Listrik"] = [mean_hourly_values3["Jumlah_Pelanggan_Listrik"] * ((1 + 0.05) ** (y - base_year)) for y in future_df3["Year"]]
        start_val3 = mean_hourly_values3["Listrik_Terjual"] * ((1 + 0.05) ** (2024 - base_year))
        week_growth3 = 0.001
        future_df3["Listrik_Terjual"] = [start_val3 * ((1 + week_growth3) ** i) for i in range(n_periods3)]
        start_val_daya3 = mean_hourly_values3["Daya_Terpasang"] * ((1 + 0.05) ** (2024 - base_year))
        future_df3["Daya_Terpasang"] = [start_val_daya3 * ((1 + week_growth3) ** i) for i in range(n_periods3)]
        start_val_produksi3 = mean_hourly_values3["Produksi_Listrik"] * ((1 + 0.05) ** (2024 - base_year))
        future_df3["Produksi_Listrik"] = [start_val_produksi3 * ((1 + week_growth3) ** i) for i in range(n_periods3)]
        future_df3["Persentase_Penduduk_Miskin"] = mean_hourly_values3["Persentase_Penduduk_Miskin"]
        future_df3["Month"] = future_df3["Date"].dt.month
        future_df3["Week"] = future_df3["Date"].dt.isocalendar().week.astype(int)
        future_df3["DayOfYear"] = future_df3["Date"].dt.dayofyear

        # Seasonal Factor
        if selected_freq == "monthly":
            seasonal_factor3 = 1 + 0.1 * np.sin(2 * np.pi * future_df3["Month"] / 12)
        elif selected_freq == "weekly":
            seasonal_factor3 = 1 + 0.1 * np.sin(2 * np.pi * future_df3["Week"] / 52)
        elif selected_freq == "daily":
            seasonal_factor3 = 1 + 0.1 * np.sin(2 * np.pi * future_df3["DayOfYear"] / 365)

        for col in ["Listrik_Terjual", "Daya_Terpasang", "Produksi_Listrik"]:
            future_df3[col] *= seasonal_factor3

        future_df3 = pd.concat([future_df3.assign(Regency=reg) for reg in location_coords["Regency"].unique()], ignore_index=True)
        future_df3 = future_df3.merge(location_coords, on=["Province", "Regency"], how="left")
        # Prediksi untuk prediksi demand setelah tahun > 2023:
        X_future3 = scaler3.transform(future_df3[feature_cols3])
        if actual_model_name in ["Linear Regression", "Random Forest", "XGBoost", "KNN","LightGBM","CatBoost","Extra Trees","SVR"]:
            y_pred_future3 = model_map_predict.predict(X_future3)
        elif actual_model_name in ["LSTM", "CNN-1D", "GRU", "MLP"]:
            print("X_future type:", type(X_future3))
            print("X_future shape:", getattr(X_future3, 'shape', 'No shape'))
            print("First few rows of X_future:", X_future3[:2] if hasattr(X_future3, '__getitem__') else "Not indexable")
            X_future_reshaped3 = X_future3.reshape((X_future3.shape[0], X_future3.shape[1]))
            y_pred_future3 = model_map_predict.predict(X_future_reshaped3)
        else:
            y_pred_future3 = model_map_predict.predict(X_future3).flatten() # NO expand_dims
        
        future_df3["Prediction"] = y_pred_future3.flatten() if hasattr(y_pred_future3, 'flatten') else y_pred_future3

        # Agregasi rata-rata prediksi per lokasi (misal per Regency)
        top_n_df = future_df3.groupby(["Regency", "Latitude", "Longitude"], as_index=False)["Prediction"].sum()
        print(f"Data size before top_n: {top_n_df.shape}")
        print(f"Unique coordinate points: {top_n_df[['Latitude', 'Longitude']].drop_duplicates().shape}")
        top_n_df = top_n_df.sort_values("Prediction", ascending=False).head(20)  # top 10 demand tertinggi
        print(top_n_df.head(20))
        # Visualisasi Map Points:
        fig_map = att.predict_map_figure(top_n_df, selected_province, year_range, top_n=20) # untuk map predict lokasi koordinat di peta map

        return fig_map, fig_line, eval_table, fig_line2, eval_table2, f"{int(pred_population):,}", f"{int(pred_poverty):,}", f"{int(pred_pdrb):,}"
    
    # Kalau tahun <= 2023 → Pakai data test (ground truth tersedia)
    else:
        selected_year = year_range[-1]
        map_fig = att.get_province_point(selected_province, selected_year, top_n=20)

        # Untuk Line Chart Prediction with All Importance Features/Parameter:
        # Bagi test_df menjadi dua: untuk evaluasi (2021–2022) dan prediksi murni (2023)
        test_df_eval = test_df[test_df['Date'].dt.year <= 2022].copy()
        test_df_pred = test_df[test_df['Date'].dt.year == 2023].copy()

        X_test_eval = scaler.transform(test_df_eval[feature_cols])
        y_test_eval = test_df_eval['Demand']
        X_test_pred = scaler.transform(test_df_pred[feature_cols])
        X_train_scaled = scaler.transform(X_train)

        # Prediction for year 2021–2022 only → for matrix evaluation table (using actual y_test_eval)
        if actual_model_name in ["Linear Regression", "Random Forest", "XGBoost", "KNN","LightGBM","CatBoost","Extra Trees","SVR"]:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_eval)
        elif actual_model_name in ["LSTM", "CNN-1D", "GRU", "MLP"]:
            model.fit(X_train_scaled, y_train)
            X_test_seq = np.expand_dims(X_test_eval, axis=-1)  # Ini penting! untuk model DL selain Transformer, karena TransformerBlock nya customize bukan bawaan library.
            y_pred = model.predict(X_test_seq).flatten()
        elif actual_model_name == "Transformer":
            model.fit(X_train_scaled, y_train)
            # Transformer custom_model:
            y_pred = model.predict(X_test_eval).flatten() # NO expand_dims
        else:
            # Handle case when selected model is invalid or missing
            return map_fig, go.Figure(), "The selected model is invalid or unavailable. Please choose another model!", "", "", ""
        
        # Prediction for year 2023 only → for Pure Prediction (don't use ground truth)
        if actual_model_name in ["Linear Regression", "Random Forest", "XGBoost", "KNN","LightGBM","CatBoost","Extra Trees","SVR"]:
            y_pred_2023 = model.predict(X_test_pred)
        elif actual_model_name in ["LSTM", "CNN-1D", "GRU", "MLP"]:
            X_pred_seq = np.expand_dims(X_test_pred, axis=-1)
            y_pred_2023 = model.predict(X_pred_seq).flatten()
        elif actual_model_name == "Transformer":
            # Transformer custom_model:
            y_pred_2023 = model.predict(X_test_pred).flatten() # NO expand_dims
        else:
            # Handle case when selected model is invalid or missing
            return map_fig, go.Figure(), "The selected model is invalid or unavailable. Please choose another model!", "", "", ""
        
        # Evaluasi model
        rmse = root_mean_squared_error(y_test_eval, y_pred)
        mae = mean_absolute_error(y_test_eval, y_pred)
        r2 = r2_score(y_test_eval, y_pred)
        eval_table = html.Tbody([
            html.Tr([html.Td("Root Mean Squared Error (RMSE)"), html.Td("=",  className="text-center"), html.Td(f"{rmse:.2f}", className="text-center")]),
            html.Tr([html.Td("Mean Absolute Error (MAE)"), html.Td("=",  className="text-center"),  html.Td(f"{mae:.2f}", className="text-center")]),
            html.Tr([html.Td("R² Score"), html.Td("=",  className="text-center"), html.Td(f"{r2:.2f}", className="text-center")]),
        ])
        
        # Prediksi gabungan untuk plotting
        pred_eval_df = test_df_eval.copy()
        pred_eval_df['Predicted_Demand'] = y_pred

        pred_pred_df = test_df_pred.copy()
        pred_pred_df['Predicted_Demand'] = y_pred_2023

        # Gabungkan kedua prediksi jadi satu dataframe (hanya untuk plotting)
        pred_all_df = pd.concat([pred_eval_df, pred_pred_df], axis=0)
        pred_all_df.set_index('Date', inplace=True)
        # Pilih hanya kolom numerik untuk resampling
        numeric_columns = pred_all_df.select_dtypes(include=['number']).columns

        # Resampling berdasarkan frekuensi
        if selected_freq == 'weekly':
            pred_resampled = pred_all_df[numeric_columns].resample('W').mean()
        elif selected_freq == 'monthly':
            pred_resampled = pred_all_df[numeric_columns].resample('M').mean()
        elif selected_freq == 'daily':
            pred_resampled = pred_all_df[numeric_columns].resample('D').mean()
        else:
            pred_resampled = pred_all_df.copy()

        pred_resampled.reset_index(inplace=True)
        # --- PLOTTING --- Grafik Prediksi vs Real Demand
        demand_fig = go.Figure()
        # Actual line: hingga 2023 (berdasarkan test_df)
        test_df['Date'] = pd.to_datetime(test_df['Date'])
        test_df.set_index('Date', inplace=True)
        actual_resampled = test_df[['Demand']].resample(selected_freq[0].upper()).mean().reset_index()
        # Grafik actual electricity demand
        demand_fig.add_trace(go.Scatter(x=actual_resampled['Date'], y=actual_resampled['Demand'], mode='lines', name='Actual Electricity Demand', line=dict(color='red')))

        # Modelled line (2021–2022)
        pred_eval_df['Date'] = pd.to_datetime(pred_eval_df['Date'])
        pred_eval_df.set_index('Date', inplace=True)
        modelled_resampled = pred_eval_df[['Predicted_Demand']].resample(selected_freq[0].upper()).mean().reset_index()
        # Grafik predicted electricity demand
        demand_fig.add_trace(go.Scatter(x=modelled_resampled['Date'], y=modelled_resampled['Predicted_Demand'], mode='lines', name='Modelled Electricity Demand', line=dict(color='blue', dash='solid')))
        
        # Predicted line (2023 murni)
        pred_pred_df['Date'] = pd.to_datetime(pred_pred_df['Date'])
        pred_pred_df.set_index('Date', inplace=True)
        pred_only_resampled = pred_pred_df[['Predicted_Demand']].resample(selected_freq[0].upper()).mean().reset_index()
        demand_fig.add_trace(go.Scatter(x=pred_only_resampled['Date'], y=pred_only_resampled['Predicted_Demand'],
                                        mode='lines', name='Prediction Electricity Demand', line=dict(color='black', dash='dash')))
        # Layout pengaturan
        demand_fig.update_layout(
            title='Actual vs Modelled Electricity Demand using All Important Parameter: (in KWh)', title_x=0.5, height=500, 
            xaxis_title='Date', yaxis_title='Electricity Demand (KWh)',
            xaxis=dict(showgrid=True, color='black', showline=True, linecolor='black', linewidth=2, tickfont=dict(size=14), titlefont=dict(size=16)),
            yaxis=dict(range=[0, 800], showgrid=True, color='black', showline=True, linecolor='black', linewidth=2, tickfont=dict(size=14), titlefont=dict(size=16)),
            legend=dict(x=1, y=0.2, xanchor='right', yanchor='top', bgcolor='snow'), margin=dict(l=40, r=40, t=100, b=60), plot_bgcolor='snow', 
            paper_bgcolor='snow', font=dict(color='black')
        )

        # Untuk Line Chart Prediction with Top 5 Highest Importance Features/Parameter:
        # Bagi test_df menjadi dua: untuk evaluasi (2021–2022) dan prediksi murni (2023)
        test_df_eval2 = test_df2[test_df2['Date'].dt.year <= 2022].copy()
        test_df_pred2 = test_df2[test_df2['Date'].dt.year == 2023].copy()

        X_test_eval2 = scaler2.transform(test_df_eval2[feature_cols2])
        y_test_eval2 = test_df_eval2['Demand']
        X_test_pred2 = scaler2.transform(test_df_pred2[feature_cols2])
        X_train_scaled2 = scaler2.transform(X_train2)

        # Prediction for year 2021–2022 only → for matrix evaluation table (using actual y_test_eval)
        if actual_model_name in ["Linear Regression", "Random Forest", "XGBoost", "KNN","LightGBM","CatBoost","Extra Trees","SVR"]:
            model_top5.fit(X_train_scaled2, y_train2)
            y_pred2 = model_top5.predict(X_test_eval2)
        elif actual_model_name in ["LSTM", "CNN-1D", "GRU", "MLP"]:
            model_top5.fit(X_train_scaled2, y_train2)
            X_test_seq2 = np.expand_dims(X_test_eval2, axis=-1)  # Ini penting! untuk model DL selain Transformer, karena TransformerBlock nya customize bukan bawaan library.
            y_pred2 = model_top5.predict(X_test_seq2).flatten()
        elif actual_model_name == "Transformer":
            model_top5.fit(X_train_scaled2, y_train2)
            # Transformer custom_model:
            y_pred2 = model_top5.predict(X_test_eval2).flatten() # NO expand_dims
        else:
            # Handle case when selected model is invalid or missing
            return map_fig, go.Figure(), "The selected model is invalid or unavailable. Please choose another model!", "", "", ""
        
        # Prediction for year 2023 only → for Pure Prediction (don't use ground truth)
        if actual_model_name in ["Linear Regression", "Random Forest", "XGBoost", "KNN","LightGBM","CatBoost","Extra Trees","SVR"]:
            y_pred_2023_2 = model_top5.predict(X_test_pred2)
        elif actual_model_name in ["LSTM", "CNN-1D", "GRU", "MLP"]:
            X_pred_seq2 = np.expand_dims(X_test_pred2, axis=-1)
            y_pred_2023_2 = model_top5.predict(X_pred_seq2).flatten()
        elif actual_model_name == "Transformer":
            # Transformer custom_model:
            y_pred_2023_2 = model_top5.predict(X_test_pred2).flatten() # NO expand_dims
        else:
            # Handle case when selected model is invalid or missing
            return map_fig, go.Figure(), "The selected model is invalid or unavailable. Please choose another model!", go.Figure(), "", "", "", ""
        
        # Evaluasi model
        rmse_2 = root_mean_squared_error(y_test_eval2, y_pred2)
        mae_2 = mean_absolute_error(y_test_eval2, y_pred2)
        r2_2 = r2_score(y_test_eval2, y_pred2)
        eval_table2 = html.Tbody([
            html.Tr([html.Td("Root Mean Squared Error (RMSE)"), html.Td("=",  className="text-center"), html.Td(f"{rmse_2:.2f}", className="text-center")]),
            html.Tr([html.Td("Mean Absolute Error (MAE)"), html.Td("=",  className="text-center"),  html.Td(f"{mae_2:.2f}", className="text-center")]),
            html.Tr([html.Td("R² Score"), html.Td("=",  className="text-center"), html.Td(f"{r2_2:.2f}", className="text-center")]),
        ])
        
        # Prediksi gabungan untuk plotting
        pred_eval_df2 = test_df_eval2.copy()
        pred_eval_df2['Predicted_Demand'] = y_pred2

        pred_pred_df2 = test_df_pred2.copy()
        pred_pred_df2['Predicted_Demand'] = y_pred_2023_2

        # Gabungkan kedua prediksi jadi satu dataframe (hanya untuk plotting)
        pred_all_df2 = pd.concat([pred_eval_df2, pred_pred_df2], axis=0)
        pred_all_df2.set_index('Date', inplace=True)
        # Pilih hanya kolom numerik untuk resampling
        numeric_columns2 = pred_all_df2.select_dtypes(include=['number']).columns

        # Resampling berdasarkan frekuensi
        if selected_freq == 'weekly':
            pred_resampled2 = pred_all_df2[numeric_columns2].resample('W').mean()
        elif selected_freq == 'monthly':
            pred_resampled2 = pred_all_df2[numeric_columns2].resample('M').mean()
        elif selected_freq == 'daily':
            pred_resampled2 = pred_all_df2[numeric_columns2].resample('D').mean()
        else:
            pred_resampled2 = pred_all_df2.copy()

        pred_resampled2.reset_index(inplace=True)
        # --- PLOTTING --- Grafik Prediksi vs Real Demand
        demand_fig2 = go.Figure()
        # Actual line: hingga 2023 (berdasarkan test_df)
        test_df2['Date'] = pd.to_datetime(test_df2['Date'])
        test_df2.set_index('Date', inplace=True)
        actual_resampled2 = test_df2[['Demand']].resample(selected_freq[0].upper()).mean().reset_index()
        # Grafik actual electricity demand
        demand_fig2.add_trace(go.Scatter(x=actual_resampled2['Date'], y=actual_resampled2['Demand'], mode='lines', name='Actual Electricity Demand', line=dict(color='red')))

        # Modelled line (2021–2022)
        pred_eval_df2['Date'] = pd.to_datetime(pred_eval_df2['Date'])
        pred_eval_df2.set_index('Date', inplace=True)
        modelled_resampled2 = pred_eval_df2[['Predicted_Demand']].resample(selected_freq[0].upper()).mean().reset_index()
        # Grafik predicted electricity demand
        demand_fig2.add_trace(go.Scatter(x=modelled_resampled2['Date'], y=modelled_resampled2['Predicted_Demand'], mode='lines', name='Modelled Electricity Demand', line=dict(color='blue', dash='solid')))
        
        # Predicted line (2023 murni)
        pred_pred_df2['Date'] = pd.to_datetime(pred_pred_df2['Date'])
        pred_pred_df2.set_index('Date', inplace=True)
        pred_only_resampled2 = pred_pred_df2[['Predicted_Demand']].resample(selected_freq[0].upper()).mean().reset_index()
        demand_fig2.add_trace(go.Scatter(x=pred_only_resampled2['Date'], y=pred_only_resampled2['Predicted_Demand'],
                                        mode='lines', name='Prediction Electricity Demand', line=dict(color='black', dash='dash')))
        # Layout pengaturan
        demand_fig2.update_layout(
            title='Actual vs Modelled Electricity Demand using Top 5 Highest Important Parameter (in KWh)', title_x=0.5, height=500,
            xaxis_title='Date', yaxis_title='Electricity Demand (KWh)', 
            xaxis=dict(showgrid=True, color='black', showline=True, linecolor='black', linewidth=2, tickfont=dict(size=14), titlefont=dict(size=16)),
            yaxis=dict(range=[0, 800], showgrid=True, color='black', showline=True, linecolor='black', linewidth=2, tickfont=dict(size=14), titlefont=dict(size=16)),
            legend=dict(x=1, y=0.2, xanchor='right', yanchor='top', bgcolor='snow'), margin=dict(l=40, r=40, t=100, b=60), plot_bgcolor='snow', 
            paper_bgcolor='snow', font=dict(color='black')
        )

        test_df.reset_index(inplace=True)
        # --- CARD PERHITUNGAN --- Menampilkan data statistik demografi
        selected_year = max(year_range)
        filtered_card_df = test_df[test_df['Date'].dt.year == selected_year]
        avg_population = filtered_card_df['Jumlah_Penduduk'].max()
        avg_poor_population = filtered_card_df['Jumlah_Penduduk_Miskin'].min()
        avg_pdrb = filtered_card_df['PDRB'].max()

        # --- UPDATE DATE COLUMN BERDASARKAN FREQUENCY (Untuk Tahun Setelah 2023) UNTUK PREDICT WITH ALL IMPORTANCES PARAMETER/FEATURES ---
        update_pred_df = pred_resampled.copy()
        # Tentukan tanggal awal dan akhir eksplisit dari slider
        start_year = year_range[0]
        end_year = year_range[1]
        start_date = pd.Timestamp(f"{start_year}-01-01")
        final_end_date = pd.Timestamp(f"{end_year}-12-31")
        # Pastikan final_end_date tidak melampaui tanggal terakhir dalam data
        if final_end_date > update_pred_df['Date'].max():
            final_end_date = update_pred_df['Date'].max()
        if selected_freq == "monthly":
            date_range = pd.date_range(start=start_date, end=final_end_date, freq="M")
        elif selected_freq == "weekly":
            # Generate all weeks
            all_weeks = pd.date_range(start=start_date, end=final_end_date + pd.Timedelta(days=7), freq="W")
            # Sekarang filter supaya hanya tanggal <= 31 Dec end_year
            all_weeks = all_weeks[all_weeks <= final_end_date]
            date_range = all_weeks
        elif selected_freq == "daily":
            date_range = pd.date_range(start=start_date, end=final_end_date, freq="D")  # fallback harian
        else:
            date_range = pd.date_range(start=start_date, end=final_end_date, freq="YS") # fallback Tahunan
        # Pastikan panjang sesuai dengan prediksi
        if len(date_range) > len(update_pred_df):
            date_range = date_range[:len(update_pred_df)]
        elif len(date_range) < len(update_pred_df):
            update_pred_df = update_pred_df.iloc[:len(date_range)]
        # Assign tanggal
        update_pred_df["Date"] = date_range

        # --- UPDATE DATE COLUMN BERDASARKAN FREQUENCY (Untuk Tahun Setelah 2023) UNTUK PREDICT WITH TOP 5 HIGHEST IMPORTANCE PARAMETER/FEATURES ---
        update_pred_df2 = pred_resampled2.copy()
        # Tentukan tanggal awal dan akhir eksplisit dari slider
        start_year = year_range[0]
        end_year = year_range[1]
        start_date = pd.Timestamp(f"{start_year}-01-01")
        final_end_date = pd.Timestamp(f"{end_year}-12-31")
        # Pastikan final_end_date tidak melampaui tanggal terakhir dalam data
        if final_end_date > update_pred_df2['Date'].max():
            final_end_date = update_pred_df2['Date'].max()
        if selected_freq == "monthly":
            date_range = pd.date_range(start=start_date, end=final_end_date, freq="M")
        elif selected_freq == "weekly":
            # Generate all weeks
            all_weeks = pd.date_range(start=start_date, end=final_end_date + pd.Timedelta(days=7), freq="W")
            # Sekarang filter supaya hanya tanggal <= 31 Dec end_year
            all_weeks = all_weeks[all_weeks <= final_end_date]
            date_range = all_weeks
        elif selected_freq == "daily":
            date_range = pd.date_range(start=start_date, end=final_end_date, freq="D")  # fallback harian
        else:
            date_range = pd.date_range(start=start_date, end=final_end_date, freq="YS") # fallback Tahunan
        # Pastikan panjang sesuai dengan prediksi
        if len(date_range) > len(update_pred_df2):
            date_range = date_range[:len(update_pred_df2)]
        elif len(date_range) < len(update_pred_df2):
            update_pred_df2 = update_pred_df2.iloc[:len(date_range)]
        # Assign tanggal
        update_pred_df2["Date"] = date_range

        return (map_fig, demand_fig, eval_table, demand_fig2, eval_table2, f"{int(avg_population):,}", f"{int(avg_poor_population):,}", f"{int(avg_pdrb):,}")
@callback(
    Output('year-slider-warning', 'style'),
    Input('province-dropdown', 'value'),
    Input('frequency-dropdown', 'value'),
    Input('model-dropdown', 'value'),
    prevent_initial_call=True
)
def year_slider_warning(selected_province, selected_frequency, selected_model):
    if selected_province is None or selected_frequency is None or selected_model is None:
        return {'color': 'red', 'fontWeight': 'bold', 'display': 'block'}
    return {'display': 'none'}
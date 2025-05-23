import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, TimeDistributed, Dropout, LayerNormalization, MultiHeadAttention, Flatten, GRU, Conv1D, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor  # type: ignore
from catboost import CatBoostRegressor
from sklearn.svm import SVR
import optuna
from tensorflow.keras.callbacks import EarlyStopping
import joblib, os
from tensorflow.keras.utils import register_keras_serializable

# dataset all:
data_all = pd.read_parquet("C:/Users/Michael Adi/Documents/DATA SKRIPSI/SKRIPSI/dataset_after_merge/Final_Data_2021_2023.parquet")
# dataset latitude dan longitude:
data_top1 = pd.read_parquet("C:/Users/Michael Adi/Documents/DATA SKRIPSI/SKRIPSI/dataset_after_merge/Latitude_Longitude_All.parquet")

# Mengubah tampilan Data Numerikal dengan Koma, tanpa mengubah tipe data dari tiap Kolom:
# Daftar kolom yang ingin diformat
kolom_format = ["Jumlah_Penduduk", "Jumlah_Penduduk_Miskin", "Jumlah_Pelanggan_Listrik", "Listrik_Terjual", "Daya_Terpasang", "Produksi_Listrik" ]

# Konversi tipe data pada kolom Date:
data_top1['Date'] = pd.to_datetime(data_top1['Date'], format="%A, %B %d, %Y")
years = sorted(data_top1['Date'].dt.year.unique())
regencies = sorted(data_top1['Regency'].unique())
provinces = sorted(data_top1['Province'].unique())

pd.set_option('display.max_columns', None)  
pd.set_option('display.width', 1000)        # Atur lebar tampilan lebih besar
pd.set_option('display.max_rows', None)  

# Daftar ibu kota untuk tiap provinsi
capital_cities = {
    "Jakarta": "Jakarta Pusat",
    "Banten": "Serang",
    "West Java": "Bandung",
    "Central Java": "Semarang",
    "Yogyakarta": "Yogyakarta",
    "East Java": "Surabaya",
    "Bali": "Denpasar"
}

# Filter hanya untuk ibu kota provinsi
capital_data = data_top1[data_top1["Regency"].isin(capital_cities.values())]

# Menghitung titik pusat latitude dan longitude yg ada di tiap province dgn cara hitung rata-rata latitude dan longitude dari setiap regency
# province_centers = data_top1.groupby("Province")[["Latitude", "Longitude"]].mean().reset_index()

# Menghitung titik pusat latitude dan longitude yg ada di tiap province dgn cara hitung rata-rata latitude dan longitude dari ibukota provinsi saja
# province_centers = capital_data.groupby("Province")[["Latitude", "Longitude"]].mean().reset_index()
province_centers = {
    "Jakarta": (-6.175392, 106.827153),         # Monas / Kantor Gubernur
    "Banten": (-6.117027, 106.151755),               # Alun alun Kota Serang, Serang
    "West Java": (-6.921671, 107.611423),           # Asia Afrika, Bandung
    "Central Java": (-6.9906, 110.4237),          # Simpang Lima, Semarang
    "Yogyakarta": (-7.800227, 110.364540),        # Istana Yogya, Yogyakarta
    "East Java": (-7.24611, 112.73750),           # Tugu Pahlawan, Surabaya
    "Bali": (-8.652000, 115.22),                 # Kota Denpasar
}

# ini dipakai jika province_centers nya pakai data mean/rata2 latitude dan longitude dari dataset. Simpan hasil ke dalam dictionary (opsional)
# province_center_dict = {
#     row["Province"]: (row["Latitude"], row["Longitude"])
#     for _, row in province_centers.iterrows()
# }

# function mencari top 5 regency berdasarkan average demand pada selected provinsi dan selected year:
def get_top_5_regency_by_demand(selected_province, selected_year, selected_params):
    if not selected_province or not selected_year or "Demand" not in selected_params:
        print("⛔ Province or year not selected, cannot calculate Top 5")
        return pd.DataFrame()
    filtered_df = data_all[
        (data_all["Province"] == selected_province) & 
        (data_all["Year"] == selected_year)
    ]
    if filtered_df.empty:
        print("⛔ There is no matching data for the selected filter")
        return pd.DataFrame()
    if "Demand" not in filtered_df.columns:
        print("⛔ Demand column not found in dataset")
        return pd.DataFrame()
    # Hitung rata-rata Demand per Regency (Kabupaten/Kota)
    top_5_df = (filtered_df.groupby("Regency")["Demand"]
                .sum()
                .nlargest(5)
                .reset_index())
    return top_5_df

# function mencari top 3 regency berdasarkan highest temperature pada selected provinsi dan selected year:
def get_top_3_regency_by_temperature(selected_province, selected_year, selected_params):
    print(f"🟢 Debugging selected_province: {selected_province}")
    print(f"🟢 Debugging selected_year: {selected_year}")
    print(f"🟢 Debugging selected_params: {selected_params}")
    if not selected_province or not selected_year or "Temperature" not in selected_params:
        print("⛔ Province or year not selected, cannot calculate Top 3")
        return pd.DataFrame()
    filtered_df = data_all[
        (data_all["Province"] == selected_province) & 
        (data_all["Year"] == selected_year)
    ]
    if filtered_df.empty:
        print("⛔ There is no matching data for the selected filter")
        return pd.DataFrame()
    if "Temperature" not in filtered_df.columns:
        print("⛔ Temperature column not found in dataset")
        return pd.DataFrame()
    print(f"✅ Data available after filter: {filtered_df.shape}")
    print(filtered_df.head())
    # Hitung rata-rata Temperature per Regency (tanpa Date)
    top_3_df = (filtered_df.groupby("Regency")["Temperature"]
                .max()
                .nlargest(3)
                .reset_index())
    print("✅ Top 3 Regency by Highest Temperature:")
    print(top_3_df)
    return top_3_df

# function mencari bottom 3 regency berdasarkan lowest temperature pada selected provinsi dan selected year:
def get_bottom_3_regency_by_temperature(selected_province, selected_year, selected_params):
    print(f"🟢 Debugging selected_province: {selected_province}")
    print(f"🟢 Debugging selected_year: {selected_year}")
    print(f"🟢 Debugging selected_params: {selected_params}")
    if not selected_province or not selected_year or "Temperature" not in selected_params:
        print("⛔ Province or year not selected, cannot calculate Bottom 3")
        return pd.DataFrame()
    filtered_df = data_all[
        (data_all["Province"] == selected_province) & 
        (data_all["Year"] == selected_year) 
    ]
    if filtered_df.empty:
        print("⛔ There is no matching data for the selected filter")
        return pd.DataFrame()
    if "Temperature" not in filtered_df.columns:
        print("⛔ Temperature column not found in dataset")
        return pd.DataFrame()
    print(f"✅ Data available after filter: {filtered_df.shape}")
    print(filtered_df.head())
    # Cari nilai minimum Temperature per Regency (tanpa Date)
    bottom_3_df = (filtered_df.groupby("Regency")["Temperature"]
                   .min()
                   .nsmallest(3)
                   .reset_index())
    print("✅ Bottom 3 Regency by Lowest Temperature:")
    print(bottom_3_df)
    return bottom_3_df

# Function untuk memperoleh seluruh Regency di Selected Province
def get_all_regency(selected_province, selected_year, df):
    if not selected_province or not selected_year:
        print("⛔ Province or Year not selected, cannot retrieve Regency data")
        return pd.DataFrame()
    filtered_df = df[(df["Province"] == selected_province) & (df["Year"] == selected_year)].copy()
    if filtered_df.empty:
        print("⛔ There is no data for the selected province and year")
        return pd.DataFrame()
    if "Regency" not in filtered_df.columns or "Demand" not in filtered_df.columns:
        print("⛔ Column 'Regency' or 'Demand' not found in dataset")
        return pd.DataFrame()
    # Hitung total Electricity Demand per Regency
    regency_treemap_data = (
        filtered_df.groupby("Regency", as_index=False)["Demand"].sum()
    )
    # Tambahkan kolom Province untuk hover
    regency_treemap_data["Province"] = selected_province
    return regency_treemap_data

# Function untuk menentukan titik Koordinat Pusat Provinsi
def get_province_center(province):
    return province_centers.get(province, None)

def get_indonesia_map():
    center_lat, center_lon = -2.5, 120  # Pusat koordinat Indonesia
    # mapbox_access_token = "your_mapbox_access_token_here"
    fig = go.Figure()
    # Tambahkan titik pusat (opsional, hanya untuk referensi)
    fig.add_trace(go.Scattermapbox(
        lat=[center_lat], lon=[center_lon], mode="markers",
        # marker=go.scattermapbox.Marker(size=10, color="blue"),
        # text=["Pusat Indonesia"], hoverinfo="text"
    ))
    # Konfigurasi layout Mapbox
    fig.update_layout(
        mapbox=dict(
            # accesstoken=mapbox_access_token,  # Tambahkan token jika perlu
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=3.5
        ),
        margin=dict(t=50, b=0, l=0, r=0)
    )
    return fig

# Function untuk menghasilkan gambar map berdasarkan Selected Province:
def get_province_map(selected_province):
    center_coords = get_province_center(selected_province)
    center_lat, center_lon = center_coords
    # Ambil nama ibu kota dari dictionary
    capital_name = capital_cities.get(selected_province, "Unknown")

    # Buat teks hover
    hover_text = (
        f"<b>Province:</b> {selected_province}<br>"
        f"<b>Capital:</b> {capital_name}<br>"
        f"<b>Latitude:</b> {center_lat:.6f}<br>"
        f"<b>Longitude:</b> {center_lon:.6f}"
    )
    print(f"✅ Update Map: {selected_province} ({center_lat}, {center_lon}) - Capital: {capital_name}")
    fig = go.Figure(go.Scattermapbox(
        lat=[center_lat], lon=[center_lon], mode='markers',
        marker=go.scattermapbox.Marker(size=7, color="red"),
        text=[hover_text], hoverinfo="text"
    ))
    fig.update_layout(
        title=f"Maps Latitude and Longitude - {selected_province}",
        title_x=0.5,
        mapbox_style="open-street-map",
        mapbox=dict(center={"lat": center_lat, "lon": center_lon}, zoom=8),
        margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )
    return fig

def get_province_point(selected_province, selected_year, top_n=20):
    # Filter data berdasarkan provinsi dan tahun
    df_filtered = data_top1[
        (data_top1["Province"] == selected_province) & 
        (pd.to_datetime(data_top1["Date"]).dt.year == selected_year)
    ]
    print(f"Data filtered untuk {selected_province} tahun {selected_year}: {len(df_filtered)} baris")
    
    if df_filtered.empty:
        print(f"⚠️ Not found data for {selected_province} Province in {selected_year}")
        return go.Figure()

    # Agregasi nilai demand berdasarkan koordinat (jika per jam, maka sum atau mean)
    df_grouped = (
        df_filtered
        .groupby(["Latitude", "Longitude", "Regency"])
        .agg({"Demand": "sum"})  # atau "mean" sesuai kebutuhan
        .reset_index()
    )

    # Ambil top N lokasi dengan demand tertinggi
    top_points = df_grouped.sort_values(by="Demand", ascending=False).head(top_n)
    print(f"Top points count: {len(top_points)}")
    print(top_points)

    # Normalize size marker by total electricity demand
    min_size = 8
    max_size = 32
    min_demand = top_points["Demand"].min()
    max_demand = top_points["Demand"].max()
    if max_demand == min_demand:
        sizes = [15] * len(top_points)  # Jika semua demand sama
    else:
        sizes = ((top_points["Demand"] - min_demand) / (max_demand - min_demand)) * (max_size - min_size) + min_size

    # Hover text
    hover_texts = [
        f"<b>Regency:</b> {row['Regency']}<br>"
        f"<b>Latitude:</b> {row['Latitude']:.5f}<br>"
        f"<b>Longitude:</b> {row['Longitude']:.5f}<br>"
        f"<b>Total Electricity Demand (KWh):</b> {row['Demand']:.2f} in KWh"
        for _, row in top_points.iterrows()
    ]

    # Center Latitude and Center Longitude from top N as Center Maps 
    center_lat = top_points["Latitude"].mean()
    center_lon = top_points["Longitude"].mean()

    # Buat scattermapbox
    fig = go.Figure(go.Scattermapbox(
        lat=top_points["Latitude"],
        lon=top_points["Longitude"],
        mode='markers',
        marker=go.scattermapbox.Marker(size=sizes, color="red", symbol="circle"),
        text=hover_texts,
        hoverinfo="text"
    ))

    fig.update_layout(
        title=f"Top {top_n} Electricity Demand Points - {selected_province} ({selected_year})",
        title_x=0.5,
        mapbox_style="open-street-map",
        mapbox=dict(center={"lat": center_lat, "lon": center_lon}, zoom=8),
        margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )
    print(f"✅ Map updated: {selected_province} {selected_year}, top {top_n} points")
    return fig

def predict_map_figure(top_n_df, selected_province, year_range, top_n=20):
    top_points = top_n_df.head(top_n)
    # Normalize size marker by total electricity demand
    min_size = 8
    max_size = 24
    min_demand = top_points["Prediction"].min()
    max_demand = top_points["Prediction"].max()
    if max_demand == min_demand:
        sizes = [15] * len(top_points)  # Jika semua demand sama
    else:
        sizes = ((top_points["Prediction"] - min_demand) / (max_demand - min_demand)) * (max_size - min_size) + min_size


    hover_texts = [
        f"Province: {selected_province} <br> Regency: {row['Regency']} <br> Latitude: {row['Latitude']} <br> Longitude: {row['Longitude']} <br> Prediction of Electricity Demand: {row['Prediction']:.2f} in KWh"
        for _, row in top_points.iterrows()
    ]

    # Center map berdasarkan rata-rata koordinat top points
    center_lat = top_points["Latitude"].mean()
    center_lon = top_points["Longitude"].mean()

    fig = go.Figure(go.Scattermapbox(
        lat=top_points["Latitude"],
        lon=top_points["Longitude"],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=sizes,
            color="red"
        ),
        text=hover_texts,
        hoverinfo="text"
    ))

    fig.update_layout(
        title=f"The Top {top_n} Electricity Demand Coordinate Points in {selected_province} Province ({year_range[0]}–{year_range[1]})",
        title_x=0.5,
        mapbox_style="open-street-map",
        mapbox=dict(
            center={"lat": center_lat, "lon": center_lon},
            zoom=7
        ),
        margin={"r":0, "t":40, "l":0, "b":0}
    )

    return fig

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Function evaluate untuk Modelling Pages:
def evaluate_model(model, X_train, y_train, X_test, y_test, model_type="sklearn"):
    if model_type == "sklearn":
        y_pred = model.predict(X_test)
    elif model_type == "custom":
        # Reshape X_test untuk deep learning (LSTM dan Transformer)
        X_test_dl = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))
        y_pred = model.predict(X_test_dl).flatten()
    else:
        raise ValueError("Unknown model type")
    return {
        'RMSE': rmse(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }
# Function get param and model untuk Modelling Pages:
def get_model_and_param_grid(model_name):
    if model_name == 'Linear Regression (Machine Learning)':
        return LinearRegression(), {
            'fit_intercept': [True, False],
            'positive': [True, False]
        }
    elif model_name == 'Random Forest (Machine Learning)':
        return RandomForestRegressor(random_state=42), {
            "n_estimators": [50, 100],
            "max_depth": [5, 10, 15, 20], 
            "max_features": [0.3, 0.5, 1.0],
            "min_samples_leaf":[50,100,150], 
            "min_samples_split":[2,5,10],
        }
    elif model_name == 'XGBoost (Machine Learning)':
        return XGBRegressor(random_state=42), {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.05, 0.1, 0.5],
            "max_depth": [5, 10, 15, 20],
            "subsample": [0.3, 0.5, 0.8, 1.0],
            "colsample_bytree": [0.5, 0.8, 1.0]
        }
    elif model_name == 'KNN (Machine Learning)':
        return KNeighborsRegressor(), {
            # "n_neighbors": list(range(5, 1001, 50)),
            "n_neighbors": [5, 10, 15, 20, 25, 50, 100, 150, 200, 250, 300, 400, 450, 500, 550, 600, 700, 750, 800, 900, 1000],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]
        }
    elif model_name == 'SVR (Machine Learning)':
        return SVR(), {
            "kernel": ["linear", "rbf"],
            "C": [0.01, 0.1, 1.0],
            "epsilon": [0.01, 0.1, 1.0],
        }
    elif model_name == 'LightGBM (Machine Learning)':
        return LGBMRegressor(random_state=42), {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1, 0.5],
            "num_leaves": [15, 31, 51, 63],
            "boosting_type": ['gbdt', 'dart']
        }
    elif model_name == 'CatBoost (Machine Learning)':
        return CatBoostRegressor(verbose=0, random_state=42), {
            'iterations': [50, 100, 200],
            'depth': [4, 6, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'l2_leaf_reg': [1, 3, 5, 7]
        }
    elif model_name == 'Extra Trees (Machine Learning)':
        return ExtraTreesRegressor(random_state=42), {
            'n_estimators': [50, 100],
            'max_depth': [10, 15, 20],
            'max_features': [0.3, 0.5, 1.0],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf':[50,100,150],
        }
    elif model_name == 'LSTM (Deep Learning)':
        return train_lstm_with_optuna, None
    elif model_name == 'Transformer (Deep Learning)':
        return train_transformer_with_optuna, None
    elif model_name == 'CNN-1D (Deep Learning)':
        return train_cnn1d_with_optuna, None
    elif model_name == 'GRU (Deep Learning)':
        return train_gru_with_optuna, None
    elif model_name == 'MLP (Deep Learning)':
        return train_mlp_with_optuna, None

# Function prepare data untuk Modelling Pages:
def prepare_data(data_all, provinsi):
    df_filtered = data_all[data_all['Province'] == provinsi].copy()
    features = ['Temperature', 'Jumlah_Penduduk', 'Jumlah_Penduduk_Miskin', 'PDRB',
                'Jumlah_Pelanggan_Listrik', 'Listrik_Terjual', 'Daya_Terpasang', 'Produksi_Listrik','Persentase_Penduduk_Miskin']
    X = df_filtered[features].copy()
    y = df_filtered['Demand'].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

# Function Train LSTM untuk Modelling Pages:
def train_lstm_with_optuna(X_scaled, y, n_trials=10):
    X_lstm = np.array(X_scaled).reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    y_lstm = np.array(y)
    def objective(trial):
        num_units = trial.suggest_int("num_units", 10, 128)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-3, 1e-1)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        model = Sequential([
            LSTM(num_units, activation='relu', return_sequences=True, input_shape=(X_lstm.shape[1], 1)),
            LSTM(num_units, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        history = model.fit(X_lstm, y_lstm, epochs=10, batch_size=batch_size, validation_split=0.2, verbose=0)
        return min(history.history['val_loss'])
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    model = Sequential([
        LSTM(best_params["num_units"], activation='relu', return_sequences=True, input_shape=(X_lstm.shape[1], 1)),
        LSTM(best_params["num_units"], activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=best_params["learning_rate"]), loss='mse')
    model.fit(X_lstm, y_lstm, epochs=20, batch_size=best_params["batch_size"], verbose=0, callbacks=[EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)],)
    return model, study.best_value, best_params

# Tambahan: Transformer Block untuk Train Transformer Model pada Modelling Pages:
@register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.attention = [
            tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
            for _ in range(num_layers)
        ]
        self.dense_ffn = [
            tf.keras.Sequential([
                Dense(dff, activation="relu"),
                Dense(d_model)
            ]) for _ in range(num_layers)
        ]
        self.layernorm = [tf.keras.layers.LayerNormalization() for _ in range(num_layers)]
        self.dropout = [tf.keras.layers.Dropout(rate) for _ in range(num_layers)]

    def call(self, inputs, training=False):
        x = inputs
        for i in range(self.num_layers):
            attn_output = self.attention[i](x, x)
            attn_output = self.dropout[i](attn_output, training=training)
            x = self.layernorm[i](x + attn_output)

            ffn_output = self.dense_ffn[i](x)
            ffn_output = self.dropout[i](ffn_output, training=training)
            x = self.layernorm[i](x + ffn_output)
        return x
    
    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            'num_layers': self.num_layers,
            'd_model': self.attention[0].key_dim,  # assuming all layers have the same key_dim
            'num_heads': self.attention[0].num_heads,
            'dff': self.dense_ffn[0].layers[0].units,  # assuming all layers have the same size
            'rate': self.dropout[0].rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
# Function Train Transformer untuk Modelling Pages:
def train_transformer_with_optuna(X_scaled, y, n_trials=10):
    X_transformer = np.array(X_scaled).reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    y_transformer = np.array(y)
    def transformer_objective(trial):
        num_layers = trial.suggest_int("num_layers", 1, 6)
        d_model = trial.suggest_int("d_model", 32, 224, step=16)  # Harus kelipatan 16
        num_heads = trial.suggest_int("num_heads", 2, 6, d_model // 32)  # Harus bisa dibagi d_model
        dff = trial.suggest_int("dff", 64, 224, step=16)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.8)

        # Membangun model Transformer
        inputs = Input(shape=(X_transformer.shape[1], X_transformer.shape[2]))  # Pastikan 3D input
        encoded_inputs = TimeDistributed(Dense(d_model))(inputs)  # Transformasi fitur agar cocok

        transformer_block = TransformerBlock(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, rate=dropout_rate)
        transformer_output = transformer_block(encoded_inputs)
        flatten_output = Flatten()(transformer_output)  # Menyesuaikan dimensi sebelum Dense terakhir
        outputs = Dense(1)(flatten_output)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

        # Training model dengan early stopping
        history = model.fit(
            X_transformer, y_transformer, epochs=10, batch_size=trial.suggest_categorical("batch_size", [32, 64, 128, 256]), validation_split=0.2,
            callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
        )
        
        return min(history.history['val_loss'])
    # Mencari hyperparameter terbaik
    study_transformer = optuna.create_study(direction="minimize")
    study_transformer.optimize(transformer_objective, n_trials=10)
    print("Best Transformer Params:", study_transformer.best_params)

    # Menggunakan parameter terbaik dari Optuna
    best_params = study_transformer.best_params
    best_transformer_block = TransformerBlock(
        num_layers=best_params["num_layers"],
        d_model=best_params["d_model"],
        num_heads=best_params["num_heads"],
        dff=best_params["dff"],
        rate=best_params["dropout_rate"]
    )
    # Final training dengan parameter terbaik
    # Membangun model Transformer dengan parameter terbaik
    inputs = Input(shape=(X_transformer.shape[1], X_transformer.shape[2]))
    encoded_inputs = TimeDistributed(Dense(best_params["d_model"]))(inputs)
    transformer_output = best_transformer_block(encoded_inputs)
    flatten_output = Flatten()(transformer_output)
    outputs = Dense(1)(flatten_output)

    model_transformer = Model(inputs=inputs, outputs=outputs)
    model_transformer.compile(optimizer=Adam(learning_rate=best_params["dropout_rate"]), loss='mse')
    print(model_transformer)
    model_transformer.fit(
        X_transformer, y_transformer,
        epochs=20,
        batch_size=best_params["batch_size"],
        callbacks=[EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)],
        verbose=0
    )
    return model_transformer, study_transformer.best_value, best_params

# Function Train CNN-1D untuk Modelling Pages:
def train_cnn1d_with_optuna(X_scaled, y, n_trials=10):
    X_cnn = np.array(X_scaled).reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    y_cnn = np.array(y)
    def objective(trial):
        num_filters = trial.suggest_int("num_filters", 16, 128, step=16)
        kernel_size = trial.suggest_int("kernel_size", 2, 10)
        dense_units = trial.suggest_int("dense_units", 16, 128, step=16)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-3, 1e-1)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        model = Sequential([
            Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu', input_shape=(X_cnn.shape[1], 1)),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(dense_units, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        history = model.fit(X_cnn, y_cnn, epochs=10, batch_size=batch_size, validation_split=0.2, verbose=0)
        return min(history.history['val_loss'])

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    final_model = Sequential([
        Conv1D(filters=best_params["num_filters"], kernel_size=best_params["kernel_size"], activation='relu', input_shape=(X_cnn.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(best_params["dense_units"], activation='relu'),
        Dense(1)
    ])
    final_model.compile(optimizer=Adam(learning_rate=best_params["learning_rate"]), loss='mse')
    final_model.fit(X_cnn, y_cnn, epochs=20, batch_size=best_params["batch_size"], verbose=0,
                    callbacks=[EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)])
    return final_model, study.best_value, best_params

# Function Train GRU untuk Modelling Pages:
def train_gru_with_optuna(X_scaled, y, n_trials=10):
    X_gru = np.array(X_scaled).reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    y_gru = np.array(y)
    def objective(trial):
        num_units = trial.suggest_int("num_units", 10, 128)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-3, 1e-1)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        model = Sequential([
            GRU(num_units, activation='relu', return_sequences=True, input_shape=(X_gru.shape[1], 1)),
            GRU(num_units, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        history = model.fit(X_gru, y_gru, epochs=10, batch_size=batch_size, validation_split=0.2, verbose=0)
        return min(history.history['val_loss'])

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    final_model = Sequential([
        GRU(best_params["num_units"], activation='relu', return_sequences=True, input_shape=(X_gru.shape[1], 1)),
        GRU(best_params["num_units"], activation='relu'),
        Dense(1)
    ])
    final_model.compile(optimizer=Adam(learning_rate=best_params["learning_rate"]), loss='mse')
    final_model.fit(X_gru, y_gru, epochs=20, batch_size=best_params["batch_size"], verbose=0,
                    callbacks=[EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)])
    return final_model, study.best_value, best_params

# Function Train MLP untuk Modelling Pages:
def train_mlp_with_optuna(X_scaled, y, n_trials=10):
    X_mlp = np.array(X_scaled)
    y_mlp = np.array(y)
    def objective(trial):
        hidden_units = trial.suggest_int("hidden_units", 32, 256, step=32)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-3, 1e-1)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        model = Sequential([
            Dense(hidden_units, activation='relu', input_shape=(X_mlp.shape[1],)),
            Dense(hidden_units, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        history = model.fit(X_mlp, y_mlp, epochs=10, batch_size=batch_size, validation_split=0.2, verbose=0)
        return min(history.history['val_loss'])

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    final_model = Sequential([
        Dense(best_params["hidden_units"], activation='relu', input_shape=(X_mlp.shape[1],)),
        Dense(best_params["hidden_units"], activation='relu'),
        Dense(1)
    ])
    final_model.compile(optimizer=Adam(learning_rate=best_params["learning_rate"]), loss='mse')
    final_model.fit(X_mlp, y_mlp, epochs=20, batch_size=best_params["batch_size"], verbose=0,
                    callbacks=[EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)])
    return final_model, study.best_value, best_params

# dictionary untuk Manggil File-File Model pada Prediction Pages untuk ALL Parameter:
models = {
    "Bali": {
        "Linear Regression": "model/models/Bali/Bali_Linear_Regression.pkl", #1
        "Random Forest": "model/models/Bali/Bali_Random_Forest.pkl", #2
        "XGBoost": "model/models/Bali/Bali_XGBoost.pkl", #3
        "SVR": "model/models/Bali/Bali_SVR.pkl", #4
        "KNN": "model/models/Bali/Bali_KNN.pkl", #5
        "LightGBM": "model/models/Bali/Bali_LightGBM.pkl", #6
        "CatBoost": "model/models/Bali/Bali_CatBoost.pkl", #7
        "Extra Trees": "model/models/Bali/Bali_Extra_Trees.pkl", #8
        "LSTM": "model/models/Bali/Bali_LSTM.keras", #9
        "Transformer": "model/models/Bali/Bali_Transformer.keras", #10
        "GRU": "model/models/Bali/Bali_GRU.keras", #11
        "CNN-1D": "model/models/Bali/Bali_CNN-1D.keras", #12
        "MLP": "model/models/Bali/Bali_MLP.keras" #13
    },
    "Banten": {
        "Linear Regression": "model/models/Banten/Banten_Linear Regression.pkl", #1
        "Random Forest": "model/models/Banten/Banten_Random Forest.pkl", #2
        "XGBoost": "model/models/Banten/Banten_XGBoost.pkl", #3
        "SVR": "model/models/Banten/Banten_SVR.pkl", #4
        "KNN": "model/models/Banten/Banten_KNN.pkl", #5
        "LightGBM": "model/models/Banten/Banten_LightGBM.pkl", #6
        "CatBoost": "model/models/Banten/Banten_CatBoost.pkl", #7
        "Extra Trees": "model/models/Banten/Banten_Extra Trees.pkl", #8
        "LSTM": "model/models/Banten/Banten_LSTM.keras", #9
        "Transformer": "model/models/Banten/Banten_Transformer.keras", #10
        "GRU": "model/models/Banten/Banten_GRU.keras", #11
        "CNN-1D": "model/models/Banten/Banten_CNN-1D.keras", #12
        "MLP": "model/models/Banten/Banten_MLP.keras" #13
    },
    "Central Java": {
        "Linear Regression": "model/models/Central Java/Central Java_Linear Regression.pkl", #1
        "Random Forest": "model/models/Central Java/Central Java_Random Forest.pkl", #2
        "XGBoost": "model/models/Central Java/Central Java_XGBoost.pkl", #3
        "SVR": "model/models/Central Java/Central Java_SVR.pkl", #4
        "KNN": "model/models/Central Java/Central Java_KNN.pkl", #5
        "LightGBM": "model/models/Central Java/Central Java_LightGBM.pkl", #6
        "CatBoost": "model/models/Central Java/Central Java_CatBoost.pkl", #7
        "Extra Trees": "model/models/Central Java/Central Java_Extra Trees.pkl", #8
        "LSTM": "model/models/Central Java/Central Java_LSTM.keras", #9
        "Transformer": "model/models/Central Java/Central Java_Transformer.keras", #10
        "GRU": "model/models/Central Java/Central Java_GRU.keras", #11
        "CNN-1D": "model/models/Central Java/Central Java_CNN-1D.keras", #12
        "MLP": "model/models/Central Java/Central Java_MLP.keras" #13
    },
    "East Java": {
        "Linear Regression": "model/models/East Java/East Java_Linear Regression.pkl", #1
        "Random Forest": "model/models/East Java/East Java_Random Forest.pkl", #2
        "XGBoost": "model/models/East Java/East Java_XGBoost.pkl", #3
        "SVR": "model/models/East Java/East Java_SVR.pkl", #4
        "KNN": "model/models/East Java/East Java_KNN.pkl", #5
        "LightGBM": "model/models/East Java/East Java_LightGBM.pkl", #6
        "CatBoost": "model/models/East Java/East Java_CatBoost.pkl", #7
        "Extra Trees": "model/models/East Java/East Java_Extra Trees.pkl", #8
        "LSTM": "model/models/East Java/East Java_LSTM.keras", #9
        "Transformer": "model/models/East Java/East Java_Transformer.keras", #10
        "GRU": "model/models/East Java/East Java_GRU.keras", #11
        "CNN-1D": "model/models/East Java/East Java_CNN-1D.keras", #12
        "MLP": "model/models/East Java/East Java_MLP.keras" #13
    },
    "Jakarta": {
        "Linear Regression": "model/models/Jakarta/Jakarta_Linear_Regression.pkl", #1
        "Random Forest": "model/models/Jakarta/Jakarta_Random_Forest.pkl", #2
        "XGBoost": "model/models/Jakarta/Jakarta_XGBoost.pkl", #3
        "SVR": "model/models/Jakarta/Jakarta_SVR.pkl", #4
        "KNN": "model/models/Jakarta/Jakarta_KNN.pkl", #5
        "LightGBM": "model/models/Jakarta/Jakarta_LightGBM.pkl", #6
        "CatBoost": "model/models/Jakarta/Jakarta_CatBoost.pkl", #7
        "Extra Trees": "model/models/Jakarta/Jakarta_Extra_Trees.pkl", #8
        "LSTM": "model/models/Jakarta/Jakarta_LSTM.keras", #9
        "Transformer": "model/models/Jakarta/Jakarta_Transformer.keras", #10
        "GRU": "model/models/Jakarta/Jakarta_GRU.keras", #11
        "CNN-1D": "model/models/Jakarta/Jakarta_CNN-1D.keras", #12
        "MLP": "model/models/Jakarta/Jakarta_MLP.keras" #13
    },
    "West Java": {
        "Linear Regression": "model/models/West Java/West Java_Linear_Regression.pkl", #1
        "Random Forest": "model/models/West Java/West Java_Random_Forest.pkl", #2
        "XGBoost": "model/models/West Java/West Java_XGBoost.pkl", #3
        "SVR": "model/models/West Java/West Java_SVR.pkl", #4
        "KNN": "model/models/West Java/West Java_KNN.pkl", #5
        "LightGBM": "model/models/West Java/West Java_LightGBM.pkl", #6
        "CatBoost": "model/models/West Java/West Java_CatBoost.pkl", #7
        "Extra Trees": "model/models/West Java/West Java_Extra_Trees.pkl", #8
        "LSTM": "model/models/West Java/West Java_LSTM.keras", #9
        "Transformer": "model/models/West Java/West Java_Transformer.keras", #10
        "GRU": "model/models/West Java/West Java_GRU.keras", #11
        "CNN-1D": "model/models/West Java/West Java_CNN-1D.keras", #12
        "MLP": "model/models/West Java/West Java_MLP.keras" #13
    },
    "Yogyakarta": {
        "Linear Regression": "model/models/Yogyakarta/Yogyakarta_Linear_Regression.pkl", #1
        "Random Forest": "model/models/Yogyakarta/Yogyakarta_Random_Forest.pkl", #2
        "XGBoost": "model/models/Yogyakarta/Yogyakarta_XGBoost.pkl", #3
        "SVR": "model/models/Yogyakarta/Yogyakarta_SVR.pkl", #4
        "KNN": "model/models/Yogyakarta/Yogyakarta_KNN.pkl", #5
        "LightGBM": "model/models/Yogyakarta/Yogyakarta_LightGBM.pkl", #6
        "CatBoost": "model/models/Yogyakarta/Yogyakarta_CatBoost.pkl", #7
        "Extra Trees": "model/models/Yogyakarta/Yogyakarta_Extra_Trees.pkl", #8
        "LSTM": "model/models/Yogyakarta/Yogyakarta_LSTM.keras", #9
        "Transformer": "model/models/Yogyakarta/Yogyakarta_Transformer.keras", #10
        "GRU": "model/models/Yogyakarta/Yogyakarta_GRU.keras", #11
        "CNN-1D": "model/models/Yogyakarta/Yogyakarta_CNN-1D.keras", #12
        "MLP": "model/models/Yogyakarta/Yogyakarta_MLP.keras" #13
    },
}

# dictionary untuk Manggil File-File Model pada Prediction Pages untuk Top 5 Parameter:
models_top5 = {
    "Bali": {
        "Linear Regression": "model/models/Bali/Bali_Linear_Regression_Top5.pkl", #1
        "Random Forest": "model/models/Bali/Bali_Random_Forest_Top5.pkl", #2
        "XGBoost": "model/models/Bali/Bali_XGBoost_Top5.pkl", #3
        "SVR": "model/models/Bali/Bali_SVR_Top5.pkl", #4
        "KNN": "model/models/Bali/Bali_KNN_Top5.pkl", #5
        "LightGBM": "model/models/Bali/Bali_LightGBM_Top5.pkl", #6
        "CatBoost": "model/models/Bali/Bali_CatBoost_Top5.pkl", #7
        "Extra Trees": "model/models/Bali/Bali_Extra_Trees_Top5.pkl", #8
        "LSTM": "model/models/Bali/Bali_LSTM_Top5.keras", #9
        "Transformer": "model/models/Bali/Bali_Transformer_Top5.keras", #10
        "GRU": "model/models/Bali/Bali_GRU_Top5.keras", #11
        "CNN-1D": "model/models/Bali/Bali_CNN-1D_Top5.keras", #12
        "MLP": "model/models/Bali/Bali_MLP_Top5.keras" #13
    },
    "Banten": {
        "Linear Regression": "model/models/Banten/Banten_Linear Regression_Top5.pkl", #1
        "Random Forest": "model/models/Banten/Banten_Random Forest_Top5.pkl", #2
        "XGBoost": "model/models/Banten/Banten_XGBoost_Top5.pkl", #3
        "SVR": "model/models/Banten/Banten_SVR_Top5.pkl", #4
        "KNN": "model/models/Banten/Banten_KNN_Top5.pkl", #5
        "LightGBM": "model/models/Banten/Banten_LightGBM_Top5.pkl", #6
        "CatBoost": "model/models/Banten/Banten_CatBoost_Top5.pkl", #7
        "Extra Trees": "model/models/Banten/Banten_Extra Trees_Top5.pkl", #8
        "LSTM": "model/models/Banten/Banten_LSTM_Top5.keras", #9
        "Transformer": "model/models/Banten/Banten_Transformer_Top5.keras", #10
        "GRU": "model/models/Banten/Banten_GRU_Top5.keras", #11
        "CNN-1D": "model/models/Banten/Banten_CNN-1D_Top5.keras", #12
        "MLP": "model/models/Banten/Banten_MLP_Top5.keras" #13
    },
    "Central Java": {
        "Linear Regression": "model/models/Central Java/Central Java_Linear Regression_Top5.pkl", #1
        "Random Forest": "model/models/Central Java/Central Java_Random Forest_Top5.pkl", #2
        "XGBoost": "model/models/Central Java/Central Java_XGBoost_Top5.pkl", #3
        "SVR": "model/models/Central Java/Central Java_SVR_Top5.pkl", #4
        "KNN": "model/models/Central Java/Central Java_KNN_Top5.pkl", #5
        "LightGBM": "model/models/Central Java/Central Java_LightGBM_Top5.pkl", #6
        "CatBoost": "model/models/Central Java/Central Java_CatBoost_Top5.pkl", #7
        "Extra Trees": "model/models/Central Java/Central Java_Extra Trees_Top5.pkl", #8
        "LSTM": "model/models/Central Java/Central Java_LSTM_Top5.keras", #9
        "Transformer": "model/models/Central Java/Central Java_Transformer_Top5.keras", #10
        "GRU": "model/models/Central Java/Central Java_GRU_Top5.keras", #11
        "CNN-1D": "model/models/Central Java/Central Java_CNN-1D_Top5.keras", #12
        "MLP": "model/models/Central Java/Central Java_MLP_Top5.keras" #13
    },
    "East Java": {
        "Linear Regression": "model/models/East Java/East Java_Linear Regression_Top5.pkl", #1
        "Random Forest": "model/models/East Java/East Java_Random Forest_Top5.pkl", #2
        "XGBoost": "model/models/East Java/East Java_XGBoost_Top5.pkl", #3
        "SVR": "model/models/East Java/East Java_SVR_Top5.pkl", #4
        "KNN": "model/models/East Java/East Java_KNN_Top5.pkl", #5
        "LightGBM": "model/models/East Java/East Java_LightGBM_Top5.pkl", #6
        "CatBoost": "model/models/East Java/East Java_CatBoost_Top5.pkl", #7
        "Extra Trees": "model/models/East Java/East Java_Extra Trees_Top5.pkl", #8
        "LSTM": "model/models/East Java/East Java_LSTM_Top5.keras", #9
        "Transformer": "model/models/East Java/East Java_Transformer_Top5.keras", #10
        "GRU": "model/models/East Java/East Java_GRU_Top5.keras", #11
        "CNN-1D": "model/models/East Java/East Java_CNN-1D_Top5.keras", #12
        "MLP": "model/models/East Java/East Java_MLP_Top5.keras" #13
    },
    "Jakarta": {
        "Linear Regression": "model/models/Jakarta/Jakarta_Linear_Regression_Top5.pkl", #1
        "Random Forest": "model/models/Jakarta/Jakarta_Random_Forest_Top5.pkl", #2
        "XGBoost": "model/models/Jakarta/Jakarta_XGBoost_Top5.pkl", #3
        "SVR": "model/models/Jakarta/Jakarta_SVR_Top5.pkl", #4
        "KNN": "model/models/Jakarta/Jakarta_KNN_Top5.pkl", #5
        "LightGBM": "model/models/Jakarta/Jakarta_LightGBM_Top5.pkl", #6
        "CatBoost": "model/models/Jakarta/Jakarta_CatBoost_Top5.pkl", #7
        "Extra Trees": "model/models/Jakarta/Jakarta_Extra_Trees_Top5.pkl", #8
        "LSTM": "model/models/Jakarta/Jakarta_LSTM_Top5.keras", #9
        "Transformer": "model/models/Jakarta/Jakarta_Transformer_Top5.keras", #10
        "GRU": "model/models/Jakarta/Jakarta_GRU_Top5.keras", #11
        "CNN-1D": "model/models/Jakarta/Jakarta_CNN-1D_Top5.keras", #12
        "MLP": "model/models/Jakarta/Jakarta_MLP_Top5.keras" #13
    },
    "West Java": {
        "Linear Regression": "model/models/West Java/West Java_Linear_Regression_Top5.pkl", #1
        "Random Forest": "model/models/West Java/West Java_Random_Forest_Top5.pkl", #2
        "XGBoost": "model/models/West Java/West Java_XGBoost_Top5.pkl", #3
        "SVR": "model/models/West Java/West Java_SVR_Top5.pkl", #4
        "KNN": "model/models/West Java/West Java_KNN_Top5.pkl", #5
        "LightGBM": "model/models/West Java/West Java_LightGBM_Top5.pkl", #6
        "CatBoost": "model/models/West Java/West Java_CatBoost_Top5.pkl", #7
        "Extra Trees": "model/models/West Java/West Java_Extra_Trees_Top5.pkl", #8
        "LSTM": "model/models/West Java/West Java_LSTM_Top5.keras", #9
        "Transformer": "model/models/West Java/West Java_Transformer_Top5.keras", #10
        "GRU": "model/models/West Java/West Java_GRU_Top5.keras", #11
        "CNN-1D": "model/models/West Java/West Java_CNN-1D_Top5.keras", #12
        "MLP": "model/models/West Java/West Java_MLP_Top5.keras" #13
    },
    "Yogyakarta": {
        "Linear Regression": "model/models/Yogyakarta/Yogyakarta_Linear_Regression_Top5.pkl", #1
        "Random Forest": "model/models/Yogyakarta/Yogyakarta_Random_Forest_Top5.pkl", #2
        "XGBoost": "model/models/Yogyakarta/Yogyakarta_XGBoost_Top5.pkl", #3
        "SVR": "model/models/Yogyakarta/Yogyakarta_SVR_Top5.pkl", #4
        "KNN": "model/models/Yogyakarta/Yogyakarta_KNN_Top5.pkl", #5
        "LightGBM": "model/models/Yogyakarta/Yogyakarta_LightGBM_Top5.pkl", #6
        "CatBoost": "model/models/Yogyakarta/Yogyakarta_CatBoost_Top5.pkl", #7
        "Extra Trees": "model/models/Yogyakarta/Yogyakarta_Extra_Trees_Top5.pkl", #8
        "LSTM": "model/models/Yogyakarta/Yogyakarta_LSTM_Top5.keras", #9
        "Transformer": "model/models/Yogyakarta/Yogyakarta_Transformer_Top5.keras", #10
        "GRU": "model/models/Yogyakarta/Yogyakarta_GRU_Top5.keras", #11
        "CNN-1D": "model/models/Yogyakarta/Yogyakarta_CNN-1D_Top5.keras", #12
        "MLP": "model/models/Yogyakarta/Yogyakarta_MLP_Top5.keras" #13
    },
}

# Function load model pada Prediction Pages untuk ALL Parameter:
def load_model_dynamic(province, model_name):
    base_path = "C:/Users/Michael Adi/Documents/DATA SKRIPSI/SKRIPSI/model/models"
    # Define the model types with correct naming
    model_types = {
        "Linear Regression": ".pkl", #1
        "Random Forest": ".pkl", #2
        "XGBoost": ".pkl", #3
        "KNN": ".pkl", #4
        "SVR": ".pkl", #5
        "LightGBM": ".pkl", #6
        "CatBoost": ".pkl", #7
        "Extra Trees": ".pkl", #8
        "LSTM": ".keras", #9
        "CNN-1D": ".keras", #10
        "Transformer": ".keras", #11
        "GRU": ".keras", #12
        "MLP": ".keras" #13
    }
    # Normalize the model_name by replacing spaces with underscores
    model_name_normalized = model_name.replace(" ", "_")
    # Try to get the correct model extension based on the normalized name
    model_extension = model_types.get(model_name_normalized, None)

    # If the normalized model name doesn't match, try the original name
    if model_extension is None:
        model_extension = model_types.get(model_name, None)
    if model_extension is None:
        raise ValueError(f"Unsupported model type: {model_name}")
    
    # Construct the full path to the model
    model_path = os.path.join(base_path, province, f"{province}_{model_name}{model_extension}")
    if not os.path.exists(model_path):
        raise ValueError(f"No model '{model_name}' available for province: {province} at {model_path}")
    
    # Load the model based on its extension
    if model_extension == ".pkl":
        print(f"Loading model from: {model_path}")
        return joblib.load(model_path)
    elif model_extension == ".keras":
        print(f"Loading model from: {model_path}")
        # Siapkan custom_objects sesuai kebutuhan
        custom_objects = {}
        if model_name == "Transformer":
            custom_objects = {"TransformerBlock": TransformerBlock}
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    else:
        raise ValueError(f"Unknown model file type for {model_path}")
    
# Function load model pada Prediction Pages
def load_model_dynamic_top5(province, model_name):
    base_path = "C:/Users/Michael Adi/Documents/DATA SKRIPSI/SKRIPSI/model/models"
    # Define the model types with correct naming
    model_types = {
        "Linear Regression": ".pkl", #1
        "Random Forest": ".pkl", #2
        "XGBoost": ".pkl", #3
        "KNN": ".pkl", #4
        "SVR": ".pkl", #5
        "LightGBM": ".pkl", #6
        "CatBoost": ".pkl", #7
        "Extra Trees": ".pkl", #8
        "LSTM": ".keras", #9
        "CNN-1D": ".keras", #10
        "Transformer": ".keras", #11
        "GRU": ".keras", #12
        "MLP": ".keras" #13
    }
    # Normalize the model_name by replacing spaces with underscores
    model_name_normalized = model_name.replace(" ", "_")
    # Try to get the correct model extension based on the normalized name
    model_extension = model_types.get(model_name_normalized, None)

    # If the normalized model name doesn't match, try the original name
    if model_extension is None:
        model_extension = model_types.get(model_name, None)
    if model_extension is None:
        raise ValueError(f"Unsupported model type: {model_name}")
    
    # Construct the full path to the model
    model_path = os.path.join(base_path, province, f"{province}_{model_name}_Top5{model_extension}")
    if not os.path.exists(model_path):
        raise ValueError(f"No model '{model_name}' available for province: {province} at {model_path}")
    
    # Load the model based on its extension
    if model_extension == ".pkl":
        print(f"Loading model from: {model_path}")
        return joblib.load(model_path)
    elif model_extension == ".keras":
        print(f"Loading model from: {model_path}")
        # Siapkan custom_objects sesuai kebutuhan
        custom_objects = {}
        if model_name == "Transformer":
            custom_objects = {"TransformerBlock": TransformerBlock}
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    else:
        raise ValueError(f"Unknown model file type for {model_path}")
    
# Function ambil nama-nama model untuk Dash UI Prediction Pages untuk ALL Parameter:
def get_available_model_names():
    # Ambil dari salah satu provinsi saja
    first_province = next(iter(models.values()))
    return list(first_province.keys())

# Function ambil nama-nama model untuk Dash UI Prediction Pages untuk Top 5 Parameter:
def get_available_model_names_top5():
    # Ambil dari salah satu provinsi saja
    first_province_top5 = next(iter(models_top5.values()))
    return list(first_province_top5.keys())
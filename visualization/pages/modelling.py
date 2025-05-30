from dash import dcc, html, Input, Output, State, callback, register_page
import dash_bootstrap_components as dbc
from lib import attribute as att
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer,mean_squared_error, mean_absolute_error, r2_score
import time, dash, json
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import datetime
import joblib
import os
register_page(__name__, path='/modelling')
df = att.data_all

layout = html.Div([
    html.Br(),
    html.H4("Modelling", className="text-center border border-dark p-2 fw-bold"),
    html.Br(),
    # Dropdowns dan tombol Train
    html.Div([
        dbc.Row([
            # Warning untuk Model SVR:
            html.Div(
                    id='svr-warning',
                    children="⚠️ If you select the SVR model, training will take approximately 12 to 24 hours.",
                    style={'color': 'red', 'fontWeight': 'bold', 'display': 'none'}
                ),
            # Warning untuk Model Random Forest, XGBoost, LSTM, Transformer:
            html.Div(
                    id='rf-xgb-lstm-transformer-warning',
                    children="⚠️ If you choose one of the models from Random Forest, XGBoost, LSTM, and Transformer, each model will require 1 to 3 hours to train.",
                    style={'color': 'red', 'fontWeight': 'bold', 'display': 'none'}
                ),
            # Warning untuk Province East Java, West Java, and Central Java:
            html.Div(
                    id='province-warning',
                    children="⚠️ If you choose one of the provinces from East Java, West Java, or Central Java, each model will require 2 to 4 hours to train.",
                    style={'color': 'red', 'fontWeight': 'bold', 'display': 'none'}
                ),
            dbc.Col(
                dcc.Dropdown(
                    id="model-provinsi-dropdown",
                    options=[{'label': prov, 'value': prov} for prov in sorted(df['Province'].dropna().unique())],
                    placeholder="Choose Province"
                ),width=4
            ),
            dbc.Col(
                dcc.Dropdown(
                    id="model-selection-dropdown",
                    options=[
                        {'label': 'Linear Regression (Machine Learning)', 'value': 'Linear Regression (Machine Learning)'},
                        {'label': 'Random Forest (Machine Learning)', 'value': 'Random Forest (Machine Learning)'},
                        {'label': 'XGBoost (Machine Learning)', 'value': 'XGBoost (Machine Learning)'},
                        {'label': 'KNN (Machine Learning)', 'value': 'KNN (Machine Learning)'},
                        {'label': 'SVR (Machine Learning)', 'value': 'SVR (Machine Learning)'},
                        {'label': 'LightGBM (Machine Learning)', 'value': 'LightGBM (Machine Learning)'},
                        {'label': 'CatBoost (Machine Learning)', 'value': 'CatBoost (Machine Learning)'},
                        {'label': 'Extra Trees (Machine Learning)', 'value': 'Extra Trees (Machine Learning)'},
                        {'label': 'LSTM (Deep Learning)', 'value': 'LSTM (Deep Learning)'},
                        {'label': 'Transformer (Deep Learning)', 'value': 'Transformer (Deep Learning)'},
                        {'label': 'CNN-1D (Deep Learning)', 'value': 'CNN-1D (Deep Learning)'},
                        {'label': 'GRU (Deep Learning)', 'value': 'GRU (Deep Learning)'},
                        {'label': 'MLP (Deep Learning)', 'value': 'MLP (Deep Learning)'}
                    ],
                    placeholder="Choose Model"
                ),width=5
            ),
            dbc.Col(
                dcc.Dropdown(
                    id="train-size-dropdown",
                    options=[{'label': f'{int(size*100)}%', 'value': size} for size in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]],
                    placeholder="Choose Train Size"
                ),width=2
            ),
            dbc.Col(
                html.Button(
                    "Train",
                    id='train-button',
                    n_clicks=0,
                    className="btn btn-success",
                    style={"width": "100%"}
                ),width=1
            ),
        ], justify="center")
    ]),
    html.Br(), html.Br(),
    html.Div(id='evaluation-table-div', style={'display': 'none'}, children=[
        html.H6("Matrix Evaluation Table", className="text-center bg-light p-2"),
        dbc.Table([
            html.Thead(html.Tr([html.Th("Metric"), html.Th("=", className="text-center"), html.Th("Value", className="text-center")])),
            html.Tbody([
                html.Tr([html.Td("RMSE"), html.Td("=", className="text-center"), html.Td(id="rmse-val", className="text-center")]),
                html.Tr([html.Td("MAE"), html.Td("=", className="text-center"), html.Td(id="mae-val", className="text-center")]),
                html.Tr([html.Td("R²"), html.Td("=", className="text-center"), html.Td(id="r2-val", className="text-center")]),
                html.Tr([html.Td("Training Time"), html.Td("=", className="text-center"), html.Td(id="time-val", className="text-center")])
            ])
        ], bordered=True, hover=True)
    ]),
    # Matrix Correlation
    html.Div(id='correlation-graph-div',className="text-center"),
    # Tabel model yang pernah dicoba
    html.Div(id='model-history-div', children=[
        html.H6("Table of Tried Models", className="text-center bg-light p-2"),
        html.Div(id='model-history-table', className="text-center")
    ]),
    html.Div([
        html.Button("Download CSV", id='download-csv-btn', className='btn btn-outline-primary mx-2'),
        html.Button("Download Excel", id='download-excel-btn', className='btn btn-outline-success mx-2'),
        html.Button("Download Trained Model", id='download-model-manual-btn', className='btn btn-dark'),
        dcc.Download(id='download-model-manual'),
        dcc.Download(id="download-model-history")
    ], className="text-center my-3"),
    # Store untuk menyimpan model history
    dcc.Store(id='model-history-store', data=[]),
    dcc.Store(id='model-path-store')
])
@callback(
    Output('evaluation-table-div', 'style'),
    Output('rmse-val', 'children'),
    Output('mae-val', 'children'),
    Output('r2-val', 'children'),
    Output('time-val', 'children'),
    Output('correlation-graph-div', 'children'),
    Output('model-history-store', 'data'),
    Output('model-history-table', 'children'),
    Output('model-path-store', 'data'),
    Input('train-button', 'n_clicks'),
    State('model-provinsi-dropdown', 'value'),
    State('model-selection-dropdown', 'value'),
    State('train-size-dropdown', 'value'),
    State('model-history-store', 'data')
)
def update_model_result(n_clicks, provinsi, selected_model, train_size, model_history):
    if n_clicks == 0 or not (provinsi and selected_model and train_size):
        return {'display': 'none'}, "", "", "", "", None, model_history, None, None

    df_filtered = df[df['Province'] == provinsi].copy()
    if df_filtered.empty:
        return {'display': 'none'}, "", "", "", "", html.Div("There is no data for this province!"), model_history, None, None
    X, y = att.prepare_data(df_filtered, provinsi)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=False)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model_or_func, param_grid = att.get_model_and_param_grid(selected_model)

    # Path untuk save model:
    filename_clean = f"{selected_model.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')}_{provinsi.replace(' ', '_')}"
    model_ext = ".keras" if "Deep Learning" in selected_model else ".pkl"
    model_path = f"C:/Users/Michael Adi/Documents/DATA SKRIPSI/SKRIPSI/saved_models/{filename_clean}{model_ext}"

    # Cek apakah model adalah fungsi deep learning
    if selected_model in ["LSTM (Deep Learning)", "Transformer (Deep Learning)", "CNN-1D (Deep Learning)", "GRU (Deep Learning)", "MLP (Deep Learning)"]:
        # Deep learning model, jangan pakai GridSearchCV
        start_time = time.time()
        trained_model, best_loss, best_params = model_or_func(X_train_scaled, y_train)
        # reshape untuk Conv1D, GRU, LSTM, Transformer, MLP jika perlu
        if selected_model in ["LSTM (Deep Learning)", "Transformer (Deep Learning)", "CNN-1D (Deep Learning)", "GRU (Deep Learning)"]:
            X_test_dl = np.array(X_test_scaled).reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
        # khusus model MLP tidak perlu reshape ke 3D
        else:  
            X_test_dl = np.array(X_test_scaled)

        y_pred = trained_model.predict(X_test_dl)
        evaluation = {
            'RMSE': att.rmse(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }
        best_model = trained_model
        end_time = time.time()
        trained_model.save(model_path) # save models
    else:
        # Model klasik (sklearn), pakai GridSearchCV
        model, param_grid = att.get_model_and_param_grid(selected_model)
        scoring = {
            'rmse': make_scorer(att.rmse, greater_is_better=False),
            'mae': 'neg_mean_absolute_error',
            'r2': 'r2'
        }
        start_time = time.time()
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, refit='r2', cv=5)
        grid.fit(X_train_scaled, y_train)
        end_time = time.time()
        best_model = grid.best_estimator_
        evaluation = att.evaluate_model(best_model, X_train_scaled, y_train, X_test_scaled, y_test)
        joblib.dump(best_model, model_path) # save models

    duration = end_time - start_time
    # Pastikan best_params selalu ada
    if selected_model not in ["LSTM (Deep Learning)", "Transformer (Deep Learning)", "CNN-1D (Deep Learning)", "GRU (Deep Learning)", "MLP (Deep Learning)"]:
        best_params = grid.best_params_
    elif selected_model in ["LSTM (Deep Learning)", "Transformer (Deep Learning)", "CNN-1D (Deep Learning)", "GRU (Deep Learning)", "MLP (Deep Learning)"]:
        best_params = best_params if best_params else {}
    # Tambahkan hasil ke history
    model_entry = {
        "Model": selected_model,
        "Best Parameters": json.dumps(best_params, indent=2),  # Ubah dict jadi string agar bisa ditampilkan
        "RMSE": round(evaluation["RMSE"], 2),
        "MAE": round(evaluation["MAE"], 2),
        "R²": round(evaluation["R2"], 4),
        "Training Time": f"{duration:.2f} s"
    }
    model_history.append(model_entry)
    # Matrix Correlation Figure:
    features = ['Demand','Geo','Temperature', 'Jumlah_Penduduk', 'Jumlah_Penduduk_Miskin', 'PDRB',
            'Jumlah_Pelanggan_Listrik', 'Listrik_Terjual', 'Daya_Terpasang', 'Produksi_Listrik',
            'Persentase_Penduduk_Miskin']
    df_corr = df_filtered[features].corr(numeric_only=True)
    fig_corr = px.imshow(df_corr, text_auto=True, aspect='auto', title='Correlation Matrix')
    # Set the title alignment to center
    fig_corr.update_layout(
        title={'x': 0.5, 'xanchor': 'center'}
    )
    dcc.Graph(figure=fig_corr)
    # Buat tabel model yang sudah dicoba
    history_df = pd.DataFrame(model_history)
    history_table = dbc.Table.from_dataframe(history_df, striped=True, bordered=True, hover=True)

    return (
        {'display': 'block'},
        round(evaluation["RMSE"], 2),
        round(evaluation["MAE"], 2),
        round(evaluation["R2"], 4),
        f"{duration:.2f} s",
        dcc.Graph(figure=fig_corr),
        model_history,
        history_table,
        model_path
    )
# Callback download model history:
@callback(
    Output("download-model-history", "data"),
    Input("download-csv-btn", "n_clicks"),
    Input("download-excel-btn", "n_clicks"),
    State("model-history-store", "data"),
    State("model-provinsi-dropdown", "value"), 
    State('train-size-dropdown', 'value'),
    prevent_initial_call=True
)
def download_model_history(csv_clicks, excel_clicks, model_history,selected_province,train_size):
    ctx = dash.callback_context
    trigger_id = ctx.triggered_id if ctx.triggered_id else None
    df = pd.DataFrame(model_history)
    # Gunakan nama provinsi dalam nama file (hapus spasi atau ubah jadi _ agar lebih aman)
    provinsi_clean = selected_province.replace(" ", "_") if selected_province else "UnknownProvince"
    filename_base = f"Model_History_{provinsi_clean}_TrainSize_{train_size}"
    if trigger_id == "download-csv-btn":
        return dcc.send_data_frame(df.to_csv, filename=f"{filename_base}_GridSearch.csv", index=False)
    elif trigger_id == "download-excel-btn":
        return dcc.send_data_frame(df.to_excel, filename=f"{filename_base}_GridSearch.xlsx", index=False, sheet_name="ModelHistory")
# Callback warning Model SVR:
@callback(
    Output('svr-warning', 'style'),
    Input('model-selection-dropdown', 'value'),
    Input('train-button', 'n_clicks'),
    prevent_initial_call=True
)
def toggle_svr_warning(selected_model, n_clicks):
    if selected_model == "SVR (Machine Learning)" and n_clicks == 0:
        return {'color': 'red', 'fontWeight': 'bold', 'display': 'block'}
    return {'display': 'none'}

# Callback warning Province East Java - West Java - Central Java:
@callback(
    Output('province-warning', 'style'),
    Input('model-provinsi-dropdown', 'value'),
    Input('train-button', 'n_clicks'),
    prevent_initial_call=True
)
def toggle_province_warning(selected_province, n_clicks):
    if (selected_province == "East Java" or selected_province == "West Java" or selected_province == "Central Java") and n_clicks == 0:
        return {'color': 'red', 'fontWeight': 'bold', 'display': 'block'}
    return {'display': 'none'}

# Warning untuk Model Random Forest, XGBoost, LSTM, Transformer:
@callback(
    Output('rf-xgb-lstm-transformer-warning', 'style'),
    Input('model-selection-dropdown', 'value'),
    Input('train-button', 'n_clicks'),
    prevent_initial_call=True
)
def toggle_province_warning(selected_model, n_clicks):
    if (selected_model == "Random Forest (Machine Learning)" or selected_model == "XGBoost (Machine Learning)" or selected_model == "LSTM (Deep Learning)" or selected_model == "Transformer (Deep Learning)") and n_clicks == 0:
        return {'color': 'red', 'fontWeight': 'bold', 'display': 'block'}
    return {'display': 'none'}

# Download Trained Model:
@callback(
    Output('download-model-manual', 'data'),
    Input('download-model-manual-btn', 'n_clicks'),
    State('model-path-store', 'data'),
    prevent_initial_call=True
)
def download_model_manual(n_clicks, model_path):
    if model_path and os.path.exists(model_path):
        return dcc.send_file(model_path)
    raise dash.exceptions.PreventUpdate
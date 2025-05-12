
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import os

st.set_page_config(page_title="Controle de Qualidade - Aço Inoxidável")

st.title("Dashboard - Classificação de Defeitos em Chapas de Aço")

uploaded_file = st.file_uploader("Faça o upload do arquivo Excel com os dados", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("Pré-visualização dos Dados")
    st.dataframe(df.head())

    model_path = "modelo_random_forest.pkl"
    scaler_path = "scaler.pkl"

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        X = df.drop(columns=["id"])
        X = X.replace({'nÃ£o': 0, 'NÃ£o': 0, 'Sim': 1, 'True': 1, 'False': 0})
        X = X.fillna(0)
        X_scaled = scaler.transform(X)

        st.subheader("Predições")
        pred = model.predict(X_scaled)
        pred_df = pd.DataFrame(pred, columns=["falha_1", "falha_2", "falha_3", "falha_4", "falha_5", "falha_6", "falha_outros"])
        pred_df.insert(0, "id", df["id"].values)
        st.dataframe(pred_df)

        st.download_button("Baixar predições", data=pred_df.to_csv(index=False).encode(), file_name="predicoes.csv", mime="text/csv")
    else:
        st.warning("O modelo e scaler treinados não foram encontrados. Coloque os arquivos 'modelo_random_forest.pkl' e 'scaler.pkl' na mesma pasta do app.")

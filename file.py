import streamlit as st
import pandas as pd

from bot_m15_m30 import BotConfig, generate_signals

st.set_page_config(
    page_title="Bot M15 e M30",
    page_icon="📊",
    layout="wide",
)

st.title("Bot de sinais M15 e M30")
st.warning(
    "Ferramenta experimental. Não há garantia de lucro. "
    "Teste os sinais em conta demonstrativa."
)

arquivo = st.file_uploader(
    "Envie candles de 1 minuto em CSV",
    type=["csv"],
)

with st.sidebar:
    st.header("Configuração")
    adx_min = st.number_input(
        "ADX mínimo",
        min_value=5.0,
        max_value=60.0,
        value=20.0,
    )
    rsi_call = st.number_input(
        "RSI mínimo para CALL",
        min_value=1.0,
        max_value=99.0,
        value=52.0,
    )
    rsi_put = st.number_input(
        "RSI máximo para PUT",
        min_value=1.0,
        max_value=99.0,
        value=48.0,
    )

if arquivo is None:
    st.info(
        "Envie um CSV com as colunas: "
        "time, open, high, low e close."
    )
    st.stop()

try:
    candles = pd.read_csv(arquivo, parse_dates=["time"])
    candles = candles.set_index("time")

    config = BotConfig(
        adx_min=adx_min,
        rsi_call=rsi_call,
        rsi_put=rsi_put,
    )

    resultado = generate_signals(candles, config)
    ultimo = resultado.iloc[-1]

    sinal = ultimo["signal"]
    cor = {
        "CALL": "green",
        "PUT": "red",
        "NONE": "gray",
    }[sinal]

    st.subheader("Sinal mais recente")
    st.markdown(
        f"## :{cor}[{sinal}]"
    )

    coluna1, coluna2, coluna3 = st.columns(3)
    coluna1.metric("RSI M15", f"{ultimo['rsi']:.2f}")
    coluna2.metric("ADX M15", f"{ultimo['adx']:.2f}")
    coluna3.metric("ADX M30", f"{ultimo['adx_m30']:.2f}")

    st.write(
        "Expiração:",
        ultimo["expiration"],
    )

    tabela = resultado[
        [
            "open",
            "high",
            "low",
            "close",
            "rsi",
            "adx",
            "adx_m30",
            "signal",
            "expiration",
        ]
    ].tail(100)

    st.subheader("Histórico")
    st.dataframe(tabela, use_container_width=True)

except Exception as erro:
    st.error(f"Não foi possível analisar o arquivo: {erro}")

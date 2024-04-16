import streamlit as st
import numpy as np
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Configuração inicial da página
st.set_page_config(page_title="Calculadora de Opções Avançada", layout="wide", page_icon="📈")

# Definindo a função para obter dados do Yahoo Finance
def get_stock_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    hist = stock.history(period="1y")  # Obtém dados históricos do último ano
    last_price = hist['Close'].iloc[-1]  # Preço de fechamento mais recente
    daily_returns = hist['Close'].pct_change().dropna()  # Mudança percentual diária
    volatilidade = np.std(daily_returns) * np.sqrt(252)  # Volatilidade anualizada
    return last_price, volatilidade

# Interface do usuário
st.title('Calculadora de Opções Avançada')
simbolo = st.text_input("Digite o símbolo do ativo (ex: AAPL):")

if simbolo:
    S, sigma = get_stock_data(simbolo)
    st.write(f"Preço Atual do Ativo: {S:.2f}")
    st.write(f"Volatilidade Anualizada: {sigma:.2%}")

    K = st.number_input("Preço de Exercício (K):", min_value=0.0, value=S, format="%.2f")
    T = st.number_input("Tempo até a Expiração (T) em anos:", min_value=0.0, value=1.0, step=0.1, format="%.2f")
    r = st.number_input("Taxa de Juros Sem Risco (r):", min_value=0.0, value=0.05, step=0.01, format="%.2f")
    opcao_tipo = st.selectbox("Tipo de Opção:", ["Europeia", "Americana", "Asiática"])

    # Demais elementos da interface aqui...

# Restante do código (implementação das funções, plots, etc.)

import streamlit as st
import numpy as np
import yfinance as yf
from scipy.stats import norm
import plotly.graph_objects as go

# Configuração inicial da página
st.set_page_config(page_title="Calculadora de Opções Avançada", layout="wide", page_icon="📈")

def get_stock_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    hist = stock.history(period="1y")  # Dados históricos do último ano
    last_price = hist['Close'].iloc[-1]  # Último preço de fechamento
    daily_returns = hist['Close'].pct_change().dropna()  # Mudança percentual diária
    volatilidade = np.std(daily_returns) * np.sqrt(252)  # Volatilidade anualizada
    return last_price, volatilidade

def calc_greeks(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) * 0.01  # Multiplicado por 0.01 para converter de % para pontos base
    return delta, gamma, vega

def plot_simulation(S, sigma, T):
    times = np.linspace(0, T, int(T*365))
    prices = S * np.exp((r - 0.5 * sigma**2) * times + sigma * np.sqrt(times) * np.random.normal(size=len(times)))
    fig = go.Figure(data=[go.Line(x=list(range(int(T*365))), y=prices)])
    fig.update_layout(title='Simulação de Preço do Ativo', xaxis_title='Dias', yaxis_title='Preço do Ativo')
    st.plotly_chart(fig)

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

    if st.button('Calcular Greeks'):
        delta, gamma, vega = calc_greeks(S, K, T, r, sigma, opcao_tipo)
        col1, col2, col3 = st.columns(3)
        col1.metric("Delta", f"{delta:.4f}")
        col2.metric("Gamma", f"{gamma:.4f}")
        col3.metric("Vega", f"{vega:.4f}")

    if st.button('Simular Preço do Ativo'):
        plot_simulation(S, sigma, T)

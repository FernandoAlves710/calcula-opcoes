import streamlit as st
import numpy as np
import yfinance as yf
from scipy.stats import norm

# Configuração inicial da página
st.set_page_config(page_title="Calculadora de Opções Avançada", layout="wide", page_icon="📈")

def get_stock_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    hist = stock.history(period="1y")  # Dados históricos do último ano
    last_price = hist['Close'].iloc[-1]  # Último preço de fechamento
    daily_returns = hist['Close'].pct_change().dropna()  # Mudança percentual diária
    volatilidade = np.std(daily_returns) * np.sqrt(252)  # Volatilidade anualizada
    return last_price, volatilidade

def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

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

    if st.button('Calcular Preço da Opção'):
        preco_opcao = black_scholes(S, K, T, r, sigma, opcao_tipo)
        st.success(f"Preço da Opção Calculada: ${preco_opcao:.2f}")



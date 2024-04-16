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

def black_scholes(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def binomial_tree(S, K, T, r, sigma, n=1000):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    prices = np.zeros((n+1, n+1))
    prices[0, 0] = S
    for i in range(1, n+1):
        prices[i, 0] = prices[i-1, 0] * u
        for j in range(1, i+1):
            prices[i, j] = prices[i-1, j-1] * d
    option_values = np.maximum(prices - K, 0)
    for j in range(n-1, -1, -1):
        for i in range(j+1):
            option_values[i, j] = np.exp(-r * dt) * (p * option_values[i, j+1] + (1 - p) * option_values[i+1, j+1])
    return option_values[0, 0]

def monte_carlo(S, K, T, r, sigma, num_simulacoes=10000):
    dt = T / 365
    Z = np.random.normal(0, 1, num_simulacoes)
    ST = S * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    payoff = np.maximum(ST - K, 0)
    preco_opcao = np.exp(-r * T) * np.mean(payoff)
    return preco_opcao

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

    if opcao_tipo == "Europeia":
        if st.button('Calcular Preço da Opção'):
            preco_opcao = black_scholes(S, K, T, r, sigma)
            st.success(f"Preço da Opção Calculada: ${preco_opcao:.2f}")
    elif opcao_tipo == "Americana":
        if st.button('Calcular Preço da Opção'):
            preco_opcao = binomial_tree(S, K, T, r, sigma)
            st.success(f"Preço da Opção Calculada: ${preco_opcao:.2f}")
    elif opcao_tipo == "Asiática":
        if st.button('Calcular Preço da Opção'):
            preco_opcao = monte_carlo(S, K, T, r, sigma)
            st.success(f"Preço da Opção Calculada: ${preco_opcao:.2f}")


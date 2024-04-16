import streamlit as st
import numpy as np
import yfinance as yf
from scipy.stats import norm

# Função para obter os dados do ativo (ação ou ETF) do Yahoo Finance
def get_stock_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    hist = stock.history(period="1y")  # Dados históricos do último ano
    last_price = hist['Close'].iloc[-1]  # Último preço de fechamento
    daily_returns = hist['Close'].pct_change().dropna()  # Mudança percentual diária
    volatility = np.std(daily_returns) * np.sqrt(252)  # Volatilidade anualizada
    return last_price, volatility

# Função para calcular o preço da opção usando o modelo de Black-Scholes
def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Função para calcular o preço da opção americana usando o modelo binomial
def binomial_option_pricing(S, K, T, r, sigma, n=100):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    prices = np.zeros((n + 1, n + 1))
    prices[0, 0] = S

    for i in range(1, n + 1):
        prices[i, 0] = prices[i - 1, 0] * u
        for j in range(1, i + 1):
            prices[i, j] = prices[i - 1, j - 1] * d

    option_values = np.zeros((n + 1, n + 1))
    for j in range(n + 1):
        option_values[n, j] = max(0, prices[n, j] - K)

    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            option_values[i, j] = np.exp(-r * dt) * (p * option_values[i + 1, j] + (1 - p) * option_values[i + 1, j + 1])

    return option_values[0, 0]

# Função para calcular o preço da opção asiática usando o método de Monte Carlo
def monte_carlo_option_pricing(S, K, T, r, sigma, num_simulations=10000):
    dt = T / 365
    Z = np.random.normal(0, 1, num_simulations)
    ST = S * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    payoff = np.maximum(ST - K, 0)
    option_price = np.exp(-r * T) * np.mean(payoff)
    return option_price

# Função para calcular o delta da opção usando o modelo de Black-Scholes
def delta(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1

# Função para calcular o gamma da opção usando o modelo de Black-Scholes
def gamma(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

# Função para calcular o vega da opção usando o modelo de Black-Scholes
def vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


# Interface do usuário
st.title('Calculadora de Opções Avançada')
simbolo = st.text_input("Digite o símbolo do ativo (ex: AAPL ou AGRI11):")

if simbolo:
    S, volatility = get_stock_data(simbolo)
    st.write(f"Preço Atual do Ativo: {S:.2f}")
    st.write(f"Volatilidade Anualizada: {volatility:.2%}")

    K = st.number_input("Preço de Exercício (K):", min_value=0.0, value=S, format="%.2f")
    T = st.number_input("Tempo até a Expiração (T) em anos:", min_value=0.0, value=1.0, step=0.1, format="%.2f")
    r = st.number_input("Taxa de Juros Sem Risco (r):", min_value=0.0, value=0.05, step=0.01, format="%.2f")
    option_type = st.selectbox("Tipo de Opção:", ["Europeia", "Americana", "Asiática"])

    if option_type == "Europeia":
        if st.button('Calcular Preço da Opção'):
            preco_opcao = black_scholes(S, K, T, r, volatility)
            st.success(f"Preço da Opção Calculada: ${preco_opcao:.2f}")
            st.write(f"Delta: {delta(S, K, T, r, volatility):.4f}")
            st.write(f"Gamma: {gamma(S, K, T, r, volatility):.4f}")
            st.write(f"Vega: {vega(S, K, T, r, volatility):.4f}")
    elif option_type == "Americana":
        if st.button('Calcular Preço da Opção'):
            preco_opcao = binomial_option_pricing(S, K, T, r, volatility)
            st.success(f"Preço da Opção Calculada: ${preco_opcao:.2f}")
            st.write(f"Delta: {delta(S, K, T, r, volatility):.4f}")
            st.write(f"Gamma: {gamma(S, K, T, r, volatility):.4f}")
            st.write(f"Vega: {vega(S, K, T, r, volatility):.4f}")
    elif option_type == "Asiática":
        if st.button('Calcular Preço da Opção'):
            preco_opcao = monte_carlo_option_pricing(S, K, T, r, volatility)
            st.success(f"Preço da Opção Calculada: ${preco_opcao:.2f}")
            st.write(f"Delta: {delta(S, K, T, r, volatility):.4f}")
            st.write(f"Gamma: {gamma(S, K, T, r, volatility):.4f}")
            st.write(f"Vega: {vega(S, K, T, r, volatility):.4f}")


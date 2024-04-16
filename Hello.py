import streamlit as st
import numpy as np
from scipy.stats import norm
import yfinance as yf

# Configuração inicial da página
st.set_page_config(page_title="Calculadora de Opções", layout="wide", page_icon="📈")

# Função para obter os dados do ativo do Yahoo Finance
def get_stock_data(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="1y")  # Dados históricos do último ano
    last_price = hist['Close'].iloc[-1]  # Último preço de fechamento
    daily_returns = hist['Close'].pct_change().dropna()  # Mudança percentual diária
    volatility = np.std(daily_returns) * np.sqrt(252)  # Volatilidade anualizada
    return last_price, volatility

# Função para calcular o preço da opção usando o modelo Black-Scholes
def black_scholes(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Função para calcular o preço da opção usando o método Binomial
def binomial_pricing(S, K, T, r, sigma, n, option_type):
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
    option_values = np.maximum(prices - K, 0)
    for j in range(n - 1, -1, -1):
        for i in range(j + 1):
            option_values[i, j] = np.exp(-r * dt) * (p * option_values[i, j + 1] + (1 - p) * option_values[i + 1, j + 1])
    return option_values[0, 0]

# Função para calcular o preço da opção usando o método de Monte Carlo
def monte_carlo_pricing(S, K, T, r, sigma, num_simulations, option_type):
    dt = T / 365
    Z = np.random.normal(0, 1, num_simulations)
    ST = S * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    payoff = np.maximum(ST - K, 0)
    if option_type == 'call':
        option_price = np.exp(-r * T) * np.mean(payoff)
    else:
        option_price = np.exp(-r * T) * np.mean(payoff)
    return option_price

# Interface do Streamlit
st.title('Calculadora de Opções')

# Sidebar para entrada de dados
st.sidebar.title('Parâmetros da Opção')
symbol = st.sidebar.text_input('Digite o símbolo do ativo (ex: AAPL):')
strike_price = st.sidebar.number_input('Preço de Exercício (K):', min_value=0.0, format="%.2f")
expiry_time = st.sidebar.number_input('Tempo até a Expiração (T) em anos:', min_value=0.0, step=0.01, format="%.2f")
risk_free_rate = st.sidebar.number_input('Taxa de Juros Sem Risco (r):', min_value=0.0, step=0.01, format="%.2f")
option_type = st.sidebar.selectbox('Tipo de Opção:', ['Call', 'Put'])

# Obter os dados do ativo do Yahoo Finance
if symbol:
    S, volatility = get_stock_data(symbol)
    st.write(f'Preço Atual do Ativo: ${S:.2f}')
    st.write(f'Volatilidade Anualizada: {volatility:.2%}')

    # Método de cálculo do preço da opção
    option_method = st.selectbox('Escolha o Método de Cálculo:', ['Black-Scholes', 'Binomial', 'Monte Carlo'])

    # Botão para calcular o preço da opção
    if st.button('Calcular Preço da Opção'):
        if option_method == 'Black-Scholes':
            option_price = black_scholes(S, strike_price, expiry_time, risk_free_rate, volatility, option_type.lower())
        elif option_method == 'Binomial':
            option_price = binomial_pricing(S, strike_price, expiry_time, risk_free_rate, volatility, 1000, option_type.lower())
        elif option_method == 'Monte Carlo':
            option_price = monte_carlo_pricing(S, strike_price, expiry_time, risk_free_rate, volatility, 10000, option_type.lower())
        st.success(f'Preço da Opção Calculado: ${option_price:.2f}')



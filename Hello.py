import streamlit as st
import numpy as np
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
import plotly.graph_objects as go

# Função para obter os dados do ativo (ação ou ETF) do Yahoo Finance
def get_stock_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    hist = stock.history(period="1y")  # Dados históricos do último ano
    last_price = hist['Close'].iloc[-1]  # Último preço de fechamento
    daily_returns = hist['Close'].pct_change().dropna()  # Mudança percentual diária
    volatility = np.std(daily_returns) * np.sqrt(252)  # Volatilidade anualizada
    return last_price, volatility, hist

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

# Função para calcular a volatilidade implícita
def implied_volatility(S, K, T, r, price, option_type='call'):
    """
    Retorna a volatilidade implícita para uma opção europeia usando o método de Brentq.
    """
    def bsm_call_imp_vol(S, K, T, r, price):
        MAX_ITERATIONS = 100
        PRECISION = 1.0e-5

        sigma = 0.5
        for i in range(MAX_ITERATIONS):
            price_est = black_scholes(S, K, T, r, sigma)
            vega = np.exp(-r * T) * norm.pdf(d1) * np.sqrt(T)
            diff = price_est - price
            if abs(diff) < PRECISION:
                return sigma
            sigma -= diff / vega
        return sigma

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return brentq(lambda x: black_scholes(S, K, T, r, x) - price, 0.001, 1)
    else:
        return brentq(lambda x: black_scholes(S, K, T, r, x, 'put') - price, 0.001, 1)

# Interface do usuário
st.set_page_config(page_title="Calculadora de Opções Avançada", layout="wide", page_icon="📈")

# Estilos personalizados
st.markdown("""
<style>
    .big-font {
        font-size:25px !important;
        font-weight: bold;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #0e1117;
    }
    .result {
        font-size: 20px;
        font-weight: bold;
        color: #036;
    }
    .description {
        font-size: 16px;
        color: #444;
    }
</style>
""", unsafe_allow_html=True)

# Título
st.title('Calculadora de Opções Avançada')

# Sidebar para entrada de dados
st.write("## Parâmetros de Entrada")
simbolo = st.text_input("Digite o símbolo do ativo desde moedas, ações, commodities e etfs (ex: AAPL):")

if simbolo:
    S, volatility, hist = get_stock_data(simbolo)
    st.write(f"Preço Atual do Ativo: {S:.2f}")
    st.write(f"Volatilidade Anualizada: {volatility:.2%}")

    K = st.number_input("Preço de Exercício (K):", min_value=0.0, value=S, format="%.2f")
    T = st.number_input("Tempo até a Expiração (T) em anos:", min_value=0.0, value=1.0, step=0.1, format="%.2f")
    r = st.number_input("Taxa de Juros Sem Risco (r):", min_value=0.0, value=0.05, step=0.01, format="%.2f")
    option_type = st.selectbox("Tipo de Opção:", ["Europeia", "Americana", "Asiática"])

    if option_type == "Europeia":
        if st.button('Calcular Preço da Opção'):
            preco_opcao = black_scholes(S, K, T, r, volatility)
            vol_imp = implied_volatility(S, K, T, r, preco_opcao)
            st.success(f"Preço da Opção Calculada: ${preco_opcao:.2f}")
            st.write("### Resultados:")
            st.write(f"Delta: {delta(S, K, T, r, volatility):.4f}")
            st.write(f"Gamma: {gamma(S, K, T, r, volatility):.4f}")
            st.write(f"Vega: {vega(S, K, T, r, volatility):.4f}")
            st.write("### Volatilidade Implícita:")
            st.write(f"{vol_imp:.2%}", cls="result")
            st.write("### Descrição:")
            st.write("Volatilidade implícita é a volatilidade futura do ativo subjacente, inferida do preço atual da opção.")

    st.write("## Histórico de Preços")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Close Price'))
    fig.update_layout(title='Histórico de Preços do Ativo nos Últimos 12 Meses',
                      xaxis_title='Data', yaxis_title='Preço')
    st.plotly_chart(fig)

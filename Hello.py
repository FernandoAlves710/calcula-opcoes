import streamlit as st
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def calcular_preco_opcao_BS(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    preco_opcao = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    return preco_opcao

def calcular_preco_opcao_MonteCarlo(S, K, T, r, sigma, num_simulacoes=10000):
    dt = T / 365
    Z = np.random.normal(0, 1, num_simulacoes)
    ST = S * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    payoff = np.maximum(ST - K, 0)
    preco_opcao = np.exp(-r * T) * np.mean(payoff)
    return preco_opcao

def calcular_preco_opcao_Binomial(S, K, T, r, sigma, n=1000):
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

def main():
    st.title("Calculadora de Opções")
    st.sidebar.header("Parâmetros da Opção")

    mercado_escolhido = st.sidebar.selectbox(
        "Escolha o mercado que deseja analisar:",
        ["Ações", "Moedas", "ETFs", "Offshore"]
    )
    simbolo = st.sidebar.text_input("Digite o símbolo do ativo ou a paridade de moeda (ex: AAPL ou EURUSD)")
    S = st.sidebar.number_input("Preço do Ativo (S):", value=100.0)
    K = st.sidebar.number_input("Preço de Exercício (K):", value=100.0)
    T = st.sidebar.number_input("Tempo até a Expiração (T) em anos:", value=1.0)
    r = st.sidebar.number_input("Taxa de Juros Sem Risco (r):", value=0.05)
    sigma = st.sidebar.number_input("Volatilidade (sigma):", value=0.2)
    
    opcao_metodo = st.sidebar.selectbox(
        "Escolha o método de solução:",
        ["Opções Europeias (Black-Scholes)", "Opções Flexíveis (Monte Carlo)", "Opções Americanas e Europeias (Binomial)", "Opções Asiáticas (Monte Carlo)"]
    )

    if st.sidebar.button('Calcular preço da opção'):
        st.subheader(f"Resultados para {simbolo} no mercado de {mercado_escolhido}")
        with st.spinner('Calculando...'):
            if opcao_metodo == "Opções Europeias (Black-Scholes)":
                preco_opcao = calcular_preco_opcao_BS(S, K, T, r, sigma)
            elif opcao_metodo == "Opções Flexíveis (Monte Carlo)":
                preco_opcao = calcular_preco_opcao_MonteCarlo(S, K, T, r, sigma)
            elif opcao_metodo == "Opções Americanas e Europeias (Binomial)":
                preco_opcao = calcular_preco_opcao_Binomial(S, K, T, r, sigma)
            elif opcao_metodo == "Opções Asiáticas (Monte Carlo)":
                preco_opcao = calcular_preco_opcao_Asiatica(S, K, T, r, sigma)

            st.success(f"Preço da Opção Calculado: {preco_opcao:.2f}")

if __name__ == "__main__":
    main()

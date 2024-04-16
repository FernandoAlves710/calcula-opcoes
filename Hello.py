import streamlit as st
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def main():
    st.set_page_config(page_title="Calculadora de Opções", layout="wide")

    # CSS para personalizar alguns estilos
    st.markdown("""
        <style>
            .big-font {
                font-size:30px !important;
                font-weight: bold;
            }
            .text-color {
                color: #0083B8;
            }
            .header-container {
                background-color: #f1f1f1;
                padding: 10px;
                border-radius: 10px;
                box-shadow: 5px 5px 5px #888888;
            }
        </style>
        """, unsafe_allow_html=True)

    # Cabeçalho
    st.markdown('<div class="header-container"><h1 class="big-font text-color">Calculadora de Opções</h1></div>', unsafe_allow_html=True)

    # Sidebar - Parâmetros de Entrada
    with st.sidebar:
        st.subheader("Parâmetros da Opção")
        S = st.number_input("Preço do Ativo (S):", value=100.0, step=0.1)
        K = st.number_input("Preço de Exercício (K):", value=100.0, step=0.1)
        T = st.number_input("Tempo até a Expiração (T) em anos:", value=1.0, step=0.1)
        r = st.number_input("Taxa de Juros Sem Risco (r):", value=0.05, step=0.01)
        sigma = st.number_input("Volatilidade (sigma):", value=0.2, step=0.01)
        option_type = st.selectbox("Tipo de Opção:", ["Europeia", "Americana", "Asiática"])

    # Área principal - Resultados
    st.markdown('## Resultados da Simulação')
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button('Calcular Preço da Opção'):
            # Aqui você chamaria a função de cálculo de opção com base no tipo selecionado
            # preco_opcao = calcular_preco_opcao(option_type, S, K, T, r, sigma)
            # Simulação de um preço para exemplo
            preco_opcao = S * np.exp(-r * T)
            st.metric(label="Preço da Opção", value=f"${preco_opcao:.2f}")

    with col2:
        st.write("Gráfico da Simulação de Preços do Ativo")
        fig, ax = plt.subplots()
        ax.plot([np.random.normal(1, 0.1) for _ in range(100)])
        ax.set_title("Simulação de Preços")
        st.pyplot(fig)

# Funções para cálculo viriam aqui...

if __name__ == "__main__":
    main()


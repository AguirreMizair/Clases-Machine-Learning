import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

# --- CONFIGURACIÓN DE PÁGINA --
st.set_page_config(page_title="Aprendizaje de maquina - FCIENCIAS", layout="wide")

st.title("🛡️ Modelo de Aprendizaje Consistente")

st.write("""
Un banco busca predecir el comportamiento crediticio basado en dos variables (atributos): 
**Ingresos Mensuales** ($x_1$) y **Puntaje de Crédito** ($x_2$). 

Existe una "regla ideal" desconocida llamada **concepto objetivo** ($c$), que mapea cada instancia 
a una etiqueta $\{0, 1\}$. Como el banco no conoce $c$, utiliza un algoritmo para 
navegar en el **espacio de hipótesis** ($H$) y encontrar una **hipótesis** ($h$) que mejor aproxime 
la función objetivo.
""")

st.subheader("Ajuste de la Hipótesis ($h$) mediante el Modelo Consistente")

st.info("""
**Nota Teórica:** Según el modelo consistente, buscamos una regla de predicción derivada de 
un conjunto de ejemplos que sea totalmente coherente con las etiquetas observadas ($c(x_i) = y_i$).
""")

# --- 1. GESTIÓN DE DATOS ---
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame({
        'Ingresos': [10.0, 15.0, 20.0, 2.0, 5.0, 8.0],
        'Puntaje': [80.0, 70.0, 90.0, 30.0, 40.0, 20.0],
        'Clase': [1, 1, 1, 0, 0, 0]
    })

with st.sidebar:
    st.header("📚 Conceptos")
    st.info("""
    - **Instancia:** El objeto a clasificar (un cliente).
    - **Atributos:** Características del cliente (Ingresos y Score).
    - **Concepto (c):** Función desconocida $X \\rightarrow \{0,1\}$ que define el éxito.
    - **Espacio de Hipótesis (H):** Conjunto de posibles reglas (líneas) que el banco puede usar.
    - **Modelo Consistente:** La regla debe ser 100% fiel a los datos de entrenamiento vistos.
    """)
    
    st.subheader("➕ Añadir Instancia")
    with st.form("new_point"):
        ni = st.number_input("Ingreso", 0.0, 30.0, 12.0)
        ns = st.number_input("Puntaje", 0.0, 100.0, 50.0)
        nl = st.selectbox("Clase", [1, 0], format_func=lambda x: "Pagó (+)" if x==1 else "Falló (-)")
        if st.form_submit_button("Agregar"):
            st.session_state.data = pd.concat([st.session_state.data, pd.DataFrame({'Ingresos': [ni], 'Puntaje': [ns], 'Clase': [nl]})], ignore_index=True)
            st.rerun()
    
    if st.button("🗑️ Reiniciar"):
        st.session_state.data = pd.DataFrame(columns=['Ingresos', 'Puntaje', 'Clase'])
        st.rerun()

# --- 2. MOTOR DE ENTRENAMIENTO ---
if not st.session_state.data.empty and len(st.session_state.data['Clase'].unique()) > 1:
    X = st.session_state.data[['Ingresos', 'Puntaje']].values
    y = st.session_state.data['Clase'].values

    clf = Perceptron(tol=None, max_iter=10000, random_state=42, eta0=1.0)
    clf.fit(X, y)
    
    acc = clf.score(X, y)
    is_consistent = (acc == 1.0)

    # --- 3. VISUALIZACIÓN ---
    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Rango dinámico para el gráfico
        x_pad, y_pad = 2, 10
        x_min, x_max = X[:,0].min()-x_pad, X[:,0].max()+x_pad
        y_min, y_max = X[:,1].min()-y_pad, X[:,1].max()+y_pad
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        
        cmap = ListedColormap(['#ff4d4d', '#4dff4d'])  # 0 → rojo, 1 → verde

        ax.contourf(
            xx, yy, Z,
            levels=[-0.5, 0.5, 1.5],
            cmap=cmap,
            alpha=0.4
        )
        ax.scatter(X[y==1, 0], X[y==1, 1], c='green', label='Pago (+)', s=120, edgecolors='k')
        ax.scatter(X[y==0, 0], X[y==0, 1], c='red', marker='x', label='Falló (-)', s=120)
        
        ax.set_title("Espacio de Hipótesis: Funciones Lineales")
        ax.set_xlabel("Ingresos Mensuales (miles)")
        ax.set_ylabel("Puntaje de Crédito (0-100)")
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("🔎 Análisis de Consistencia")
        if is_consistent:
            st.success(f"**¡LOGRADO!** Precisión: {acc*100}%")
            st.write("Se encontró una $h \\in H$ que es **consistente** con los ejemplos")
        else:
            st.error(f"**NO CONSISTENTE** (Precisión: {acc*100:.1f}%)")
            st.write("No se encontro un modelo lineal perfectamente separable con los clientes")
            
# --- SECCIÓN 3: PREDICCIÓN INTERACTIVA ---
st.subheader("💡 Prueba la Hipótesis con un Nuevo Cliente")
st.write("Modifica los atributos para ver cómo generaliza el modelo aprendido.")

c1, c2 = st.columns(2)
with c1:
    ingreso_test = st.slider("Ingresos Mensuales", 0, 25, 12)
    score_test = st.slider("Puntaje de Crédito", 0, 100, 50)
    
    cliente_nuevo = np.array([[ingreso_test, score_test]])
    prediccion = clf.predict(cliente_nuevo)
    
    resultado = "✅ CRÉDITO APROBADO" if prediccion[0] == 1 else "❌ CRÉDITO RECHAZADO"
    st.metric("Decisión del Algoritmo", resultado)

with c2:
    st.write("**Relación con el Aprendizaje PAC**")
    st.write("""
    Aunque el modelo sea consistente con los 6 clientes anteriores, el **Error de Generalización** $e(h)$ 
    es la probabilidad de que este nuevo cliente sea clasificado mal.
    
    Para que el banco esté **Probablemente Aproximadamente Correcto (PAC)**, necesitaría que este cliente 
    provenga de la misma distribución de probabilidad $D$ que los anteriores.
    """)

# --- 4. TABLA INTERACTIVA ---
st.divider()
st.subheader("📝 Tabla de datos")
st.session_state.data = st.data_editor(st.session_state.data, num_rows="dynamic", use_container_width=True)

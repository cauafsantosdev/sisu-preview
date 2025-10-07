import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px


# Sets the browser tab's title, icon, and layout. "wide" makes better use of screen space.
st.set_page_config(
    page_title="SISU Preview | Previsor de Notas de Corte",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS injection to customize the colors and create a "dark purple" theme.
st.markdown("""
<style>
    .main {
        background-color: #0E0B16;
    }
    [data-testid="stSidebar"] {
        background-color: #1A1032;
    }
    body, h1, h2, h3, h4, h5, h6, p, .st-emotion-cache-16txtl3, .st-emotion-cache-1xarl3l {
        color: #E0E0E0;
    }
    .stButton>button {
        background-color: #7B2CBF;
        color: white;
        border: none;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #9D4EDD;
        color: white;
    }
    [data-testid="stMetricValue"] {
        color: #A450F3;
    }
    [data-testid="stMetricDelta"] svg {
        fill: #9D4EDD;
    }
</style>
""", unsafe_allow_html=True)

# @st.cache_resource and @st.cache_data ensure that the model and data are loaded only once.
@st.cache_resource
def load_model():
    """Loads the trained model."""
    model_path = os.path.join('saved_models', 'lgbm_sisu_predictor.joblib')
    model = joblib.load(model_path)
    return model

@st.cache_data
def load_data():
    """Loads and prepares the dataset to populate the filters."""
    data_url = "https://github.com/cauafsantosdev/sisu-preview/releases/download/v1.0.0/final_data.parquet"
    df_full = pd.read_parquet(data_url)
    df_specialist = df_full.query("ds_mod_concorrencia == 'AMPLA CONCORRÊNCIA' and qt_vagas_concorrencia >= 10").copy()
    return df_specialist

model = load_model()
df = load_data()

st.sidebar.title("🔮 SISU Preview")
st.sidebar.markdown("Selecione os parâmetros para estimar a nota de corte:")

# Dynamic filters that adjust based on the previous selection
ies_options = sorted(df['sg_ies'].dropna().unique())
selected_ies = st.sidebar.selectbox('Universidade (IES)', ies_options)

df_filtered_ies = df[df['sg_ies'] == selected_ies]
course_options = sorted(df_filtered_ies['no_curso'].dropna().unique())
selected_course = st.sidebar.selectbox('Curso', course_options)

df_filtered_course = df_filtered_ies[df_filtered_ies['no_curso'] == selected_course]
degree_options = sorted(df_filtered_course['ds_grau'].dropna().unique())
selected_degree = st.sidebar.selectbox('Grau', degree_options)

shift_options = sorted(df_filtered_course['ds_turno'].dropna().unique())
selected_shift = st.sidebar.selectbox('Turno', shift_options)

predict_button = st.sidebar.button('✨ Estimar Nota de Corte', type='primary', use_container_width=True)

# Results section
st.title("Estimador de Nota de Corte do SISU")
st.markdown("Este simulador utiliza um modelo de Machine Learning para prever a nota de corte para a modalidade **Ampla Concorrência** com base em dados históricos.")
st.markdown("---")

if predict_button:
    # Find the most recent data for the selected combination
    latest_data = df[
        (df['sg_ies'] == selected_ies) &
        (df['no_curso'] == selected_course) &
        (df['ds_grau'] == selected_degree) &
        (df['ds_turno'] == selected_shift)
    ].sort_values('edicao', ascending=False).iloc[0:1]

    if latest_data.empty:
        st.error("Não foram encontrados dados históricos para a combinação selecionada. A previsão não é possível.")
    else:
        # Prediction logic remains the same
        train_features = model.feature_name_
        features_for_prediction = latest_data[train_features]

        for col in ['sg_ies', 'no_curso', 'ds_grau', 'ds_turno']:
            if col in features_for_prediction.columns:
                features_for_prediction[col] = features_for_prediction[col].astype('category')
        
        predicted_score = model.predict(features_for_prediction)[0]
        last_score = latest_data['nu_notacorte'].iloc[0]

        # Display results in columns
        st.subheader(f"Estimativa para {selected_course} ({selected_shift}) na {selected_ies}")
        col1, col2 = st.columns(2)
        col1.metric("Última Nota Registrada", f"{last_score:.2f}")
        col2.metric("Estimativa para a Próxima Edição", f"{predicted_score:.2f}", delta=f"{predicted_score - last_score:.2f}")
        
        # Filter all historical data for the selected course
        historical_data = df[
            (df['sg_ies'] == selected_ies) &
            (df['no_curso'] == selected_course) &
            (df['ds_grau'] == selected_degree) &
            (df['ds_turno'] == selected_shift)
        ].sort_values('edicao')

        if historical_data.shape[0] > 1:
            # Create the figure with Plotly Express
            fig = px.line(
                historical_data,
                x='edicao',
                y='nu_notacorte',
                markers=True,
                labels={'edicao': 'Edição do SISU', 'nu_notacorte': 'Nota de Corte'},
                title=f"Evolução da Nota de Corte para {selected_course} ({selected_shift}) na {selected_ies}"
            )

            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                title_font_color='#E0E0E0',
                font_color='#E0E0E0',
                xaxis_title=None
            )

            fig.update_traces(
                line=dict(color='#A450F3', width=3),
                marker=dict(size=8)
            )
            
            # Display the chart in Streamlit
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Não há dados históricos suficientes (mínimo 2 pontos) para gerar um gráfico de evolução.")

        # Communicate the model's accuracy
        st.info(f"**Sobre a Estimativa:** O erro médio deste modelo é de aproximadamente **{16:.0f} pontos** para mais ou para menos. Use este valor como uma referência e não como uma garantia.", icon="💡")
else:
    st.info("Por favor, utilize os filtros na barra lateral e clique no botão para gerar uma estimativa.")

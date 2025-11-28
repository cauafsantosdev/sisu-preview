import re
import duckdb
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# Page configuration
st.set_page_config(
    page_title="SISU Preview | Estimador de Notas de Corte do SISU",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom purple on main values
st.markdown("""
<style>
div[data-testid="stMetricValue"] {
    color: #A450F3 !important;
}
</style>
""", unsafe_allow_html=True)

colour_pallete = [
    '#6600FF',
    '#E0B0FF',
    '#301934',
    '#DA70D6',
    '#00008B',
    '#FF00FF',
    '#9370DB',
    '#F8D0FF',
    '#4B0082',
    '#EE82EE',
    '#8A2BE2',
    '#C71585',
    '#D8BFD8',
    '#800080',
    '#483D8B'
]

MAX_COMPARISONS = 5
MODELS_DIR = "saved_models"

@st.cache_data
def load_data():
    """
    Loads database on cache.
    """
    db_path = "data/database/sisu_preview.db"
    conn = duckdb.connect(database=str(db_path), read_only=True)
    df = conn.sql("SELECT * FROM sisu_data").df()
    df = df[df["nu_notacorte"] != 0].copy()

    # String normalization
    for c in ["ds_mod_concorrencia", "no_curso", "sg_ies", "no_campus", "ds_grau", "ds_turno"]:
        df[c] = df[c].astype(str)
    return df

def edition_key(edition):
    """
    Converts an edition string to an integer for sorting
    """
    edition_string = str(edition)
    match_year = re.search(r"(19|20)\d{2}", edition_string)
    year = int(match_year.group(0)) if match_year else 0
    
    # Search for 1 or 2 after a separator
    match_semester = re.search(r"[\/_\-\s](1|2)\b", edition_string)
    if match_semester:
        half = int(match_semester.group(1))
    else:
        half = 0

    return year * 10 + half

# Initial state
if "comparisons" not in st.session_state:
    st.session_state.comparisons = []

# Loads data and models
df = load_data()
model = joblib.load('saved_models/lgbm_sisu_predictor.joblib')
modalities_all = sorted(df["ds_mod_concorrencia"].unique().tolist())

# Sidebar
st.sidebar.title("🔮 SISU Preview")
st.sidebar.caption("Adicione até 5 combinações (IES + Curso + Campus + Grau + Turno + Modalidade(s))")

ies_options = sorted(df["sg_ies"].unique().tolist())
selected_ies = st.sidebar.selectbox("Universidade (IES)", [""] + ies_options)

df_ies = df[df["sg_ies"] == selected_ies] if selected_ies else df.copy()
course_options = sorted(df_ies["no_curso"].unique().tolist())
selected_course = st.sidebar.selectbox("Curso", [""] + course_options)

df_course = df_ies[df_ies["no_curso"] == selected_course] if selected_course else df_ies.copy()
campus_options = sorted(df_course["no_campus"].unique().tolist())
selected_campus = st.sidebar.selectbox("Campus", ["Todos"] + campus_options)

degree_options = sorted(df_course["ds_grau"].unique().tolist())
selected_degree = st.sidebar.selectbox("Grau", [""] + degree_options)

shift_options = sorted(df_course["ds_turno"].unique().tolist())
selected_shift = st.sidebar.selectbox("Turno", [""] + shift_options)

modalities_present = sorted(df_course["ds_mod_concorrencia"].unique().tolist()) or modalities_all
selected_modalities = st.sidebar.multiselect(
    "Modalidade(s)", modalities_present, default=["AC"] if "AC" in modalities_present else modalities_present[:1]
)

# Buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("➕ Adicionar curso", use_container_width=True):
        if len(st.session_state.comparisons) >= MAX_COMPARISONS:
            st.sidebar.error(f"Máximo de {MAX_COMPARISONS} comparações atingido.")
        elif not selected_ies or not selected_course or not selected_degree or not selected_shift or not selected_modalities:
            st.sidebar.error("Preencha todos os campos antes de adicionar.")
        else:
            st.session_state.comparisons.append({
                "sg_ies": selected_ies,
                "no_curso": selected_course,
                "no_campus": selected_campus,
                "ds_grau": selected_degree,
                "ds_turno": selected_shift,
                "modalities": selected_modalities
            })
with col2:
    if st.button("🧹 Remover cursos", use_container_width=True):
        st.session_state.comparisons = []

# Prediction button
estimate_all = st.sidebar.button("✨ Estimar Notas",  type='primary', use_container_width=True)

# List of courses added to comparison
st.sidebar.markdown("**Comparações adicionadas:**")
if st.session_state.comparisons:
    for i, c in enumerate(st.session_state.comparisons):
        st.sidebar.markdown(f"{i+1}. {c['sg_ies']} — {c['no_curso']} ({c['ds_grau']}, {c['ds_turno']}) | Modalidades: {', '.join(c['modalities'])}")
else:
    st.sidebar.info("Nenhuma comparação adicionada.")

# Main Title
st.title("Estimador de Notas de Corte do SISU")
st.markdown("Este simulador utiliza um modelo de Machine Learning para prever a nota de corte para qualquer curso com base em dados históricos.")
st.markdown("---")

# Run prediction
if estimate_all:
    # Check if there are any comparisons added
    if not st.session_state.comparisons:
        st.warning("Adicione pelo menos uma comparação na barra lateral.")
    else:
        results = []
        history = []

        # Iterate over all comparisons and modalities
        for comp in st.session_state.comparisons:
            for mod in comp["modalities"]:
                # Filter dataset for the selected combination
                subset = df[
                    (df["sg_ies"] == comp["sg_ies"]) &
                    (df["no_curso"] == comp["no_curso"]) &
                    (df["ds_grau"] == comp["ds_grau"]) &
                    (df["ds_turno"] == comp["ds_turno"]) &
                    (df["ds_mod_concorrencia"] == mod)
                ].copy()
                if comp["no_campus"] != "Todos":
                    subset = subset[subset["no_campus"] == comp["no_campus"]]

                # If no data is available for this combination
                if subset.empty:
                    results.append({
                        "IES": comp["sg_ies"], "Campus": comp["no_campus"], "Curso": comp["no_curso"],
                        "Modalidade": mod, "Última Nota": np.nan, "Previsão": np.nan, "Observação": "Sem dados"
                    })
                    continue
                
                # If model is not loaded
                if model is None:
                    pred = np.nan
                else:
                    # Get the latest known historical record
                    latest = subset.sort_values("edicao", ascending=False).iloc[0:1].copy()
                    last_year = int(latest["ano"].iloc[0])
                    last_real = float(latest["nu_notacorte"].iloc[0])

                    # Create base row for 2026
                    latest_future = latest.copy()
                    latest_future["edicao"] = str(last_year + 1)
                    latest_future["ano"] = last_year + 1
                    latest_future["nu_notacorte"] = 0.0  # target to predict

                    # Recalculate lags
                    subset_sorted = subset.sort_values("edicao").copy()

                    internal_lags = ["lag1_nota", "lag2_nota", "lag1_vagas", "lag2_vagas", "lag1_inscritos", "lag2_inscritos"]

                    latest_future["lag1_nota"] = subset_sorted["nu_notacorte"].iloc[-1]
                    latest_future["lag2_nota"] = subset_sorted["nu_notacorte"].iloc[-2] if len(subset_sorted) >= 2 else np.nan

                    latest_future["lag1_vagas"] = subset_sorted["qt_vagas_concorrencia"].iloc[-1]
                    latest_future["lag2_vagas"] = subset_sorted["qt_vagas_concorrencia"].iloc[-2] if len(subset_sorted) >= 2 else np.nan

                    latest_future["lag1_inscritos"] = subset_sorted["qt_inscricao"].iloc[-1]
                    latest_future["lag2_inscritos"] = subset_sorted["qt_inscricao"].iloc[-2] if len(subset_sorted) >= 2 else np.nan

                    # Cleaned names
                    latest_future["vagas_edicao_anterior"] = latest_future["lag1_vagas"]
                    latest_future["inscritos_edicao_anterior"] = latest_future["lag1_inscritos"]

                    # Trend
                    if latest_future["lag1_nota"] is not None and latest_future["lag2_nota"] is not None:
                        latest_future["tendencia_nota"] = latest_future["lag1_nota"] - latest_future["lag2_nota"]
                    else:
                        latest_future["tendencia_nota"] = 0.0

                    # Demand
                    latest_future["demanda_anterior"] = latest_future["lag1_inscritos"] / (latest_future["lag1_vagas"] + 1)

                    # Delta features
                    latest_future["delta_vagas"] = (
                        latest_future["lag1_vagas"].iloc[0] - latest_future["lag2_vagas"].iloc[0]
                        if pd.notna(latest_future["lag1_vagas"].iloc[0]) and pd.notna(latest_future["lag2_vagas"].iloc[0])
                        else 0
                    )

                    latest_future["delta_inscritos"] = (
                        latest_future["lag1_inscritos"].iloc[0] - latest_future["lag2_inscritos"].iloc[0]
                        if pd.notna(latest_future["lag1_inscritos"].iloc[0]) and pd.notna(latest_future["lag2_inscritos"].iloc[0])
                        else 0
                    )

                    # Growth rate
                    latest_future["taxa_crescimento_nota"] = (
                        (latest_future["lag1_nota"].iloc[0] - latest_future["lag2_nota"]).iloc[0] / latest_future["lag2_nota"].iloc[0]
                        if pd.notna(latest_future["lag2_nota"]).iloc[0] and latest_future["lag2_nota"].iloc[0] > 0
                        else 0
                    )

                    # National mean by degree
                    subset_grau = df[df["ds_grau"] == latest["ds_grau"].iloc[0]]
                    latest_future["media_nacional_grau"] = subset_grau["nu_notacorte"].mean()

                    # Region mean and std
                    subset_regiao = df[df["regiao"] == latest["regiao"].iloc[0]]
                    latest_future["media_regiao"] = subset_regiao["nu_notacorte"].mean()
                    latest_future["desvio_regiao"] = subset_regiao["nu_notacorte"].std(ddof=0)

                    # Relative deltas
                    def safe_rel_diff(col):
                        if len(subset_sorted) >= 2:
                            prev = subset_sorted[col].iloc[-2]
                            last = subset_sorted[col].iloc[-1]
                            return (last - prev) / (prev + 1)
                        return 0.0
                    
                    latest_future["delta_vagas_rel"] = safe_rel_diff("qt_vagas_concorrencia")
                    latest_future["delta_inscritos_rel"] = safe_rel_diff("qt_inscricao")
                    latest_future["demanda_ratio"] = float(latest["qt_inscricao"].iloc[0]) / (
                        float(latest["qt_vagas_concorrencia"].iloc[0]) + 1
                    )

                    # Normalize year using min/max of the entire course history
                    years = subset["ano"].astype(float)
                    min_year = years.min()
                    max_year = years.max()

                    latest_future["ano_norm"] = (latest_future["ano"].astype(float) - min_year) / (max_year - min_year + 1e-9)

                    # Lists all model features
                    model_features = list(model.feature_name_)

                    # Cleans columns that are not features
                    for col in internal_lags:
                        if col not in model_features and col in latest_future.columns:
                            latest_future.drop(columns=[col], inplace=True)

                    # Ensure all model features exist
                    feats = [f for f in model.feature_name_ if f in latest_future.columns]
                    missing = [f for f in model.feature_name_ if f not in latest_future.columns]

                    for m in missing:
                        latest_future[m] = 0

                    X_pred = latest_future[model.feature_name_].copy()

                    for c in X_pred.select_dtypes(include="object").columns:
                        X_pred[c] = X_pred[c].astype("category")

                    try:
                        # Model predicts the cutoff score
                        pred = float(model.predict(X_pred)[0])

                    except Exception as e:
                        print("Prediction error:", e)
                        pred = np.nan

                # Append prediction result
                results.append({
                    "IES": comp["sg_ies"], "Campus": latest["no_campus"].iloc[0],
                    "Curso": comp["no_curso"], "Modalidade": mod,
                    "Última Nota": last_real, "Previsão": round(pred, 2)
                })

                # Store historical data for plotting
                for _, r in subset.iterrows():
                    history.append({
                        "Edição": r["edicao"], "Nota": r["nu_notacorte"],
                        "Label": f"{comp['no_curso']} — {mod} — {comp['sg_ies']}"
                    })

        df_results = pd.DataFrame(results)
        df_hist = pd.DataFrame(history)

        # Display prediction metrics
        if not df_results.empty:
            # Sort by IES and course to keep grouped
            for (ies, curso), group in df_results.groupby(by=["IES", "Curso"], sort=False):
                st.markdown(f"#### {curso} — {ies}")

                # Create columns dynamically (up to 3 per row)
                cols = st.columns(min(len(group), 3))
                for i, (_, row) in enumerate(group.iterrows()):
                    with cols[i % len(cols)]:
                        modality = row["Modalidade"]
                        last_score = row["Última Nota"]
                        predicted = row["Previsão"]
                        delta = predicted - last_score if pd.notna(predicted) and pd.notna(last_score) else None
                        st.metric(
                            label=f"{modality}",
                            value=f"{predicted:.2f}" if pd.notna(predicted) else "–",
                            delta=f"{delta:+.2f}" if delta is not None else None,
                            help=f"Última nota: {last_score:.2f}" if pd.notna(last_score) else "Sem histórico de notas"
                        )

            st.markdown("---")
    
        # Historical line chart
        if not df_hist.empty:
            # Calculate sorting key and sort
            df_hist["_ed_key"] = df_hist["Edição"].apply(edition_key)
            ordered_edicoes = (df_hist[["Edição", "_ed_key"]].drop_duplicates()
                            .sort_values("_ed_key")["Edição"].tolist())

            # Force categorical order for edition so plotly respects sequence
            df_hist["Edição"] = pd.Categorical(df_hist["Edição"], categories=ordered_edicoes, ordered=True)

            fig_hist = px.line(
                df_hist.sort_values("_ed_key"),
                x="Edição",
                y="Nota",
                color="Label",
                line_group="Label",
                markers=True,
                color_discrete_sequence=colour_pallete,
                title="Histórico de Notas de Corte"
            )

            fig_hist.update_traces(line=dict(width=2))
            fig_hist.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#E0E0E0",
                xaxis=dict(categoryorder="array", categoryarray=ordered_edicoes, tickangle=-45),
                height=650,
                margin=dict(t=60, b=140)
            )
            fig_hist.update_xaxes(title_text="")
            fig_hist.update_yaxes(title_text="Nota de Corte")
            fig_hist.update_layout(
                legend=dict(
                    title=None,
                    orientation="h",
                    yanchor="bottom",
                    y=-0.3,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10)
                )
            )

            st.plotly_chart(fig_hist, use_container_width=True)

            # Removes auxiliary column
            df_hist.drop(columns=["_ed_key"], inplace=True, errors="ignore")

        # Vertical bar chart
        if not df_results.empty:
            df_bar = df_results.dropna(subset=["Previsão"])
            df_bar["Curso/IES"] = df_bar["Curso"] + " (" + df_bar["IES"] + ")"

            unique_labels = df_bar["Curso/IES"].unique()

            fig_bar = px.bar(
                df_bar, x="Curso/IES", y="Previsão", color="Modalidade",
                text_auto=".2f", barmode="group",
                title="Notas Estimadas por Curso, Modalidade e IES",
                color_discrete_sequence=colour_pallete
            )
            fig_bar.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#E0E0E0",
                xaxis_tickangle=-30
            )
            fig_bar.update_xaxes(title_text="")
            fig_bar.update_yaxes(title_text="Estimated Score")
            st.plotly_chart(fig_bar, use_container_width=True)

    
        # Results table
        if not df_results.empty:
            df_results["Diferença"] = df_results["Previsão"] - df_results["Última Nota"]
            st.dataframe(
                df_results.style.format({
                    "Última Nota": "{:.2f}", "Previsão": "{:.2f}", "Diferença": "{:+.2f}"
                }),
                use_container_width=True
            )

        st.markdown("---")
        st.info("As estimativas podem apresentar diferenças comparadas às notas reais.", icon="💡")

else:
    st.info("Adicione combinações na barra lateral e clique em **Estimar Notas** para gerar as previsões.")
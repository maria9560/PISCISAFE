# PISCISAFE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
import streamlit as st
from datetime import datetime
import plotly.express as px


# ============================================================
# 0) CLASSIFICAÇÃO DE RISCO FUTURO
# ============================================================
def classify_risk(ph: float) -> str:
    if 6.5 <= ph <= 8.5:
        return "bom"
    elif 6.0 <= ph < 6.5 or 8.5 < ph <= 9.0:
        return "alerta"
    else:
        return "critico"


# ============================================================
# A) FUNÇÕES DE INFERÊNCIA — TILÁPIA E PIRARUCU PELO pH
# ============================================================
def inferir_tilapia_pct(ph: float) -> float:
    ph_lim = max(6.0, min(9.0, ph))
    escala = (ph_lim - 6.0) / 3.0
    tilapia = 0.10 + escala * 0.75
    return tilapia

def inferir_pirarucu_pct(ph: float) -> float:
    return 1 - inferir_tilapia_pct(ph)


# ============================================================
# 1) CARREGAMENTO & TRATAMENTO DA BASE
# ============================================================
@st.cache_data
def load_data(json_path: str = "dados_aqua.json") -> pd.DataFrame:
    df = pd.read_json(json_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["tanque_id", "timestamp"])

    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek

    df["ph_future"] = df.groupby("tanque_id")["ph"].shift(-1)
    df = df.dropna(subset=["ph_future"]).reset_index(drop=True)

    df["risk_future"] = df["ph_future"].apply(classify_risk)

    return df


# ============================================================
# 2) TREINO DOS MODELOS — REGRESSÃO + CLASSIFICAÇÃO
# ============================================================
@st.cache_resource
def treinar_modelos(df: pd.DataFrame):

    feature_cols = ["water_temp_c", "feed_amount_g", "feed_quality", "hour", "dayofweek"]
    X = df[feature_cols]
    y_ph = df["ph_future"]
    y_risk = df["risk_future"]

    numeric_features = ["water_temp_c", "feed_amount_g", "hour", "dayofweek"]
    categorical_features = ["feed_quality"]

    preprocessor = ColumnTransformer([
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])

    reg = RandomForestRegressor(n_estimators=200, random_state=42)
    pipe_reg = Pipeline([("prep", preprocessor), ("model", reg)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_ph, test_size=0.2, random_state=42
    )

    pipe_reg.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, pipe_reg.predict(X_test))

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    pipe_clf = Pipeline([("prep", preprocessor), ("model", clf)])

    strat = None if (y_risk.value_counts() < 2).any() else y_risk

    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X, y_risk, test_size=0.2, random_state=42, stratify=strat
    )

    pipe_clf.fit(X_train2, y_train2)
    preds = pipe_clf.predict(X_test2)
    acc = accuracy_score(y_test2, preds)

    metrics = {
        "mae_regressao": mae,
        "acc_classificacao": acc,
        "relatorio_classificacao": classification_report(y_test2, preds)
    }

    return pipe_reg, pipe_clf, metrics


# ============================================================
# 3) SIMULAÇÃO DE CENÁRIO
# ============================================================
def prever_cenario(pipe_reg, pipe_clf, temp, feed_g, feed_q, dt: datetime):

    dados = pd.DataFrame([{
        "water_temp_c": temp,
        "feed_amount_g": feed_g,
        "feed_quality": feed_q,
        "hour": dt.hour,
        "dayofweek": dt.weekday()
    }])

    ph_prev = pipe_reg.predict(dados)[0]
    risk_prev = pipe_clf.predict(dados)[0]

    return ph_prev, risk_prev


# ============================================================
# 4) ASSISTENTE VIRTUAL
# ============================================================
def gerar_resumo(df):
    riscos = df["risk_future"].value_counts()
    ph_medio = df["ph"].mean()
    temp_media = df["water_temp_c"].mean()

    return f"""
Resumo do Cenário:

• pH médio: {ph_medio:.2f}
• Temperatura média: {temp_media:.2f}°C
• Bom: {riscos.get('bom', 0)}
• Alerta: {riscos.get('alerta', 0)}
• Crítico: {riscos.get('critico', 0)}
"""


def relatorio_tanque(df, tanque):
    return f"""
RELATÓRIO DO TANQUE {tanque}

• Mínimo pH = {df['ph'].min():.2f}
• Máximo pH = {df['ph'].max():.2f}
• Média pH  = {df['ph'].mean():.2f}

{gerar_resumo(df)}
"""


def interpretar_metricas(metrics):
    return f"""
• MAE (Regressão): {metrics['mae_regressao']:.3f}
• Acurácia: {metrics['acc_classificacao']:.3f}
"""


# ============================================================
# 5) INTERFACE STREAMLIT
# ============================================================
def main():

    st.title("Piscisafe 🐟💧 — Dashboard Inteligente")

    menu = st.sidebar.radio(
        "Menu",
        ["Dashboard", "Simulação", "Assistente Virtual"]
    )

    df = load_data("dados_aqua.json")

    st.sidebar.header("Filtros")

    min_date = df["timestamp"].min().date()
    max_date = df["timestamp"].max().date()

    data_inicio, data_fim = st.sidebar.date_input(
        "Intervalo de datas",
        [min_date, max_date],
        min_value=min_date, max_value=max_date
    )

    hora_inicio = st.sidebar.time_input("Hora inicial", datetime.strptime("00:00", "%H:%M").time())
    hora_fim = st.sidebar.time_input("Hora final", datetime.strptime("23:59", "%H:%M").time())

    tanque = st.sidebar.selectbox("Tanque", df["tanque_id"].unique())

    df_filtrado = df[
        (df["tanque_id"] == tanque) &
        (df["timestamp"].dt.date >= data_inicio) &
        (df["timestamp"].dt.date <= data_fim) &
        (df["timestamp"].dt.time >= hora_inicio) &
        (df["timestamp"].dt.time <= hora_fim)
    ]

    if df_filtrado.empty:
        st.warning("Sem dados nesse intervalo.")
        return


    # ============================================================
    # DASHBOARD
    # ============================================================
    if menu == "Dashboard":

        st.subheader("Distribuição do risco futuro")

        cont = df_filtrado["risk_future"].value_counts().reset_index()
        cont.columns = ["categoria", "quantidade"]

        fig_risco = px.bar(
            cont,
            x="categoria",
            y="quantidade",
            color="categoria",
            title="Distribuição — Bom / Alerta / Crítico"
        )
        st.plotly_chart(fig_risco, use_container_width=True)

        st.subheader("Treinando modelos…")
        pipe_reg, pipe_clf, metrics = treinar_modelos(df_filtrado)

        col1, col2 = st.columns(2)
        col1.metric("MAE", f"{metrics['mae_regressao']:.3f}")
        col2.metric("Acurácia", f"{metrics['acc_classificacao']:.3f}")

        # Previsão futura
        df_filtrado["ph_prev_model"] = pipe_reg.predict(
            df_filtrado[["water_temp_c", "feed_amount_g", "feed_quality", "hour", "dayofweek"]]
        )

        # Colunas temporais
        df_filtrado["ano"] = df_filtrado["timestamp"].dt.year
        df_filtrado["mes"] = df_filtrado["timestamp"].dt.to_period("M").astype(str)
        df_filtrado["dia"] = df_filtrado["timestamp"].dt.date
        df_filtrado["hora"] = df_filtrado["timestamp"].dt.hour

        dias_span = (data_fim - data_inicio).days

        st.subheader("📊 Evolução dinâmica do pH")

        if dias_span <= 1:
            eixo = "hora"
        elif dias_span <= 60:
            eixo = "dia"
        elif dias_span <= 365:
            eixo = "mes"
        else:
            eixo = "ano"

        df_plot = df_filtrado.groupby(eixo).agg(
            ph_real=("ph", "mean"),
            ph_prev=("ph_prev_model", "mean")
        ).reset_index()

        fig = px.line(
            df_plot, x=eixo, y=["ph_real", "ph_prev"],
            markers=True, title="Evolução do pH"
        )

        fig.add_hline(y=6.5, line_dash="dot", annotation_text="Ideal mínimo")
        fig.add_hline(y=8.5, line_dash="dot", annotation_text="Ideal máximo")

        st.plotly_chart(fig, use_container_width=True)


        # ============================================================
        # NOVO GRÁFICO — POPULAÇÃO ESTIMADA EM COLUNAS
        # ============================================================
        st.subheader("📈 Composição estimada de Tilápias × Pirarucus")

        df_filtrado["tilapia_est"] = df_filtrado["ph"].apply(inferir_tilapia_pct)
        df_filtrado["pirarucu_est"] = 1 - df_filtrado["tilapia_est"]

        # Escolher granularidade automática
        if dias_span > 365:
            df_filtrado["bucket"] = df_filtrado["timestamp"].dt.year
        elif dias_span > 60:
            df_filtrado["bucket"] = df_filtrado["timestamp"].dt.to_period("M").astype(str)
        elif dias_span > 10:
            df_filtrado["bucket"] = df_filtrado["timestamp"].dt.to_period("W").astype(str)
        else:
            df_filtrado["bucket"] = df_filtrado["timestamp"].dt.date

        df_pop = df_filtrado.groupby("bucket").agg(
            tilapia=("tilapia_est", "mean"),
            pirarucu=("pirarucu_est", "mean")
        ).reset_index()

        fig_pop = px.bar(
            df_pop,
            x="bucket",
            y=["tilapia", "pirarucu"],
            barmode="group",
            labels={"value": "Proporção", "variable": "Espécie"},
            title="Composição estimada por período"
        )

        st.plotly_chart(fig_pop, use_container_width=True)


    # ============================================================
    # SIMULAÇÃO
    # ============================================================
    elif menu == "Simulação":

        st.subheader("🔮 Simular Futuro")
        pipe_reg, pipe_clf, _ = treinar_modelos(df_filtrado)

        colA, colB = st.columns(2)

        with colA:
            temp = st.slider("Temperatura (°C)", 20.0, 35.0, 28.0)
            feed_g = st.slider("Ração (g)", 0, 300, 150)

        with colB:
            feed_q = st.selectbox("Qualidade da ração", ["boa", "media", "ruim"])
            d_input = st.date_input("Data")
            h_input = st.time_input("Hora")

        if st.button("Prever cenário"):
            dt = datetime.combine(d_input, h_input)
            ph_prev, risk_prev = prever_cenario(pipe_reg, pipe_clf, temp, feed_g, feed_q, dt)
            st.success(f"pH previsto: **{ph_prev:.2f}** • Risco: **{risk_prev.upper()}**")


    # ============================================================
    # ASSISTENTE VIRTUAL
    # ============================================================
    elif menu == "Assistente Virtual":

        st.subheader("🤖 Assistente Virtual Piscisafe")

        colA, colB, colC = st.columns(3)

        if colA.button("📄 Resumo"):
            st.info(gerar_resumo(df_filtrado))

        if colB.button("📘 Relatório"):
            st.info(relatorio_tanque(df_filtrado, tanque))

        if colC.button("📊 Métricas"):
            _, _, m = treinar_modelos(df_filtrado)
            st.info(interpretar_metricas(m))

        pergunta = st.text_input("Pergunte ao sistema:")

        if pergunta:
            p = pergunta.lower()

            if "ph" in p:
                st.info(f"O pH médio é {df_filtrado['ph'].mean():.2f}")
            elif "temperatura" in p:
                st.info(f"A temperatura média é {df_filtrado['water_temp_c'].mean():.2f}°C")
            elif "risco" in p:
                st.info(gerar_resumo(df_filtrado))
            else:
                st.info("Posso ajudar com análises e previsões!")


# ============================================================
# EXECUTAR
# ============================================================
if __name__ == "__main__":
    main()

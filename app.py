from datetime import date

import streamlit as st
import MODISyVIIRS_prot1 as core

st.set_page_config(page_title="Focos | MODIS + VIIRS", page_icon="◉", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #1c2924;
        --forest: #173a31;
        --moss: #dce8d8;
        --paper: #f4f5ef;
        --line: #dce2d8;
        --muted: #68766e;
        --signal: #df633e;
        --signal-light: #fff0e9;
    }
    html, body, [class*="css"] {
        font-family: 'Manrope', 'Segoe UI', sans-serif;
        color: var(--ink);
    }
    [data-testid="stAppViewContainer"] {
        background-color: var(--paper);
        background-image: repeating-linear-gradient(135deg, rgba(23,58,49,.025) 0, rgba(23,58,49,.025) 1px, transparent 1px, transparent 10px);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] { max-width: 1440px; padding-top: 2.2rem; }
    [data-testid="stSidebar"] > div:first-child {
        background: var(--forest);
        border-right: 1px solid rgba(255,255,255,.12);
    }
    [data-testid="stSidebar"] * { color: #eff3e9; }
    [data-testid="stSidebar"] [data-baseweb="input"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-testid="stDateInput"] input {
        background: #24483e;
        border-color: #48675d;
        color: #fff;
    }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #c5d2c8;
        font-size: .78rem;
        font-weight: 700;
    }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,.82);
        border: 1px solid var(--line);
        border-top: 3px solid #9aad9c;
        border-radius: 5px;
        padding: 15px 18px;
        min-height: 112px;
    }
    [data-testid="stMetric"] label p {
        color: var(--muted);
        font-size: .75rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    [data-testid="stMetricValue"] { color: var(--ink); font-weight: 800; }
    .hero {
        position: relative;
        overflow: hidden;
        background: var(--forest);
        color: #f7f8f1;
        border-radius: 5px;
        padding: 30px 34px 27px;
        margin-bottom: 22px;
        border-bottom: 4px solid var(--signal);
    }
    .hero:after {
        content: '';
        position: absolute;
        right: 0; top: 0; bottom: 0; width: 34%;
        opacity: .19;
        background: repeating-linear-gradient(135deg, transparent 0 12px, #dce8d8 12px 13px);
        mask-image: linear-gradient(90deg, transparent, black);
        pointer-events: none;
    }
    .eyebrow, .micro, .side-mark {
        font-family: 'DM Mono', Consolas, monospace;
        letter-spacing: .04em;
        text-transform: uppercase;
    }
    .eyebrow { color: #c4d8c5; font-size: .72rem; margin-bottom: 8px; }
    .hero-title { font-size: clamp(1.8rem, 3vw, 2.65rem); font-weight: 800; line-height: 1.08; }
    .hero-meta { color: #d1ddd3; font-size: .84rem; margin-top: 13px; }
    .side-mark { color: #b4c8b9; font-size: .68rem; margin: 4px 0 23px; }
    .section-heading {
        color: var(--forest);
        font-size: 1.06rem;
        font-weight: 800;
        margin: 4px 0 2px;
    }
    .section-rule { border-top: 1px solid var(--line); margin: 12px 0 18px; }
    [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 7px; }
    [data-testid="stTabs"] button[role="tab"] {
        color: var(--muted);
        font-weight: 700;
        border-radius: 4px 4px 0 0;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--forest);
        border-bottom-color: var(--signal);
    }
    div.stButton > button[kind="primary"] {
        background: var(--signal);
        border: 1px solid var(--signal);
        color: white;
        border-radius: 4px;
        font-weight: 800;
        min-height: 44px;
    }
    div.stButton > button[kind="primary"]:hover { background: #c94f2e; border-color: #c94f2e; }
    .empty-state {
        border: 1px dashed #b8c6b9;
        background: rgba(255,255,255,.62);
        border-radius: 5px;
        padding: 36px;
        color: var(--muted);
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("<div class='side-mark'>FIRE / OBSERVATION / 01</div>", unsafe_allow_html=True)
    st.markdown("### Parámetros")
    key = st.text_input("FIRMS MAP_KEY", type="password")
    bbox_text = st.text_input(
        "BBox · O, S, E, N",
        "-56.5502,-33.3985,-56.4950,-33.3520",
    )
    inicio = st.date_input("Fecha inicial", date(2023, 1, 1))
    fin = st.date_input("Fecha final", date(2023, 12, 30))
    ejecutar = st.button("Consultar FIRMS", type="primary", use_container_width=True)
    st.caption("NASA FIRMS · MODIS / VIIRS")

if ejecutar:
    if not key:
        st.session_state["dashboard_error"] = "Ingresa tu MAP_KEY de FIRMS."
    elif inicio > fin:
        st.session_state["dashboard_error"] = "La fecha inicial debe ser anterior a la final."
    else:
        try:
            bbox = core.parse_bbox(bbox_text)
            with st.spinner("Consultando detecciones satelitales…"):
                det = core.descargar_firms(key, bbox, inicio.isoformat(), fin.isoformat())
                mlt = core.mlt_empirico(det)
                conteos = core.conteos_diarios(det)
                indice, fuente, factores, referencia, escalados = core.armonizar(conteos, mlt)
                matriz = core.matriz_anual(
                    indice.rolling(7, min_periods=4, center=True).mean()
                )
                z, valores = core.anomalias(matriz)
                criticos = core.periodos_criticos(z, valores, matriz.index.values)
            st.session_state["dashboard"] = {
                "detecciones": det,
                "conteos": conteos,
                "indice": indice,
                "mlt": mlt,
                "criticos": criticos,
                "bbox": bbox,
                "inicio": inicio,
                "fin": fin,
            }
            st.session_state.pop("dashboard_error", None)
        except SystemExit as error:
            st.session_state["dashboard_error"] = str(error)
        except Exception as error:
            st.session_state["dashboard_error"] = f"No se pudieron procesar los datos: {error}"

data = st.session_state.get("dashboard")
if data:
    bbox_label = ", ".join(f"{value:.4f}" for value in data["bbox"])
    st.markdown(
        f"<div class='hero'><div class='eyebrow'>Observatorio de actividad · FIRMS</div>"
        f"<div class='hero-title'>Fuego visto desde<br>órbita.</div>"
        f"<div class='hero-meta'>{data['inicio']:%d %b %Y} — {data['fin']:%d %b %Y}"
        f" &nbsp; / &nbsp; bbox {bbox_label}</div></div>",
        unsafe_allow_html=True,
    )

    detecciones = data["detecciones"]
    conteos = data["conteos"]
    criticos = data["criticos"]
    total = len(detecciones)
    confirmadas = int(detecciones["conf_ok"].sum())
    dias_activos = int(conteos.sum(axis=1).gt(0).sum())
    dias_totales = len(conteos.index)

    metricas = st.columns(4)
    metricas[0].metric("Detecciones", f"{total:,}")
    metricas[1].metric("Confianza aceptada", f"{confirmadas:,}")
    metricas[2].metric("Días con actividad", f"{dias_activos:,}", f"de {dias_totales} días")
    metricas[3].metric("Periodos críticos", f"{len(criticos):,}")

    st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)
    mapa_tab, actividad_tab, sensores_tab, eventos_tab = st.tabs(
        ["Mapa", "Actividad", "Satélites", "Periodos críticos"]
    )

    with mapa_tab:
        st.markdown("<div class='section-heading'>Distribución espacial</div>", unsafe_allow_html=True)
        puntos = detecciones.loc[:, ["lat", "lon"]].dropna()
        if len(puntos) > 5000:
            puntos = puntos.sample(5000, random_state=42)
        st.map(puntos, latitude="lat", longitude="lon", zoom=7, size=12, color="#df633e")
        if len(detecciones) > len(puntos):
            st.caption(f"Mapa de una muestra de {len(puntos):,} detecciones para mantenerlo ágil.")

    with actividad_tab:
        st.markdown("<div class='section-heading'>Evolución de detecciones</div>", unsafe_allow_html=True)
        st.line_chart(conteos, height=340, use_container_width=True)
        st.markdown("<div class='section-heading'>Índice armonizado</div>", unsafe_allow_html=True)
        st.line_chart(data["indice"].rename("Índice equivalente a VIIRS"), height=280, use_container_width=True)

    with sensores_tab:
        izquierda, derecha = st.columns([1.2, 1])
        with izquierda:
            st.markdown("<div class='section-heading'>Detecciones por sensor</div>", unsafe_allow_html=True)
            totales = conteos.sum().sort_values(ascending=False).rename("Detecciones")
            st.bar_chart(totales, height=300, use_container_width=True)
        with derecha:
            st.markdown("<div class='section-heading'>Hora local estimada</div>", unsafe_allow_html=True)
            st.line_chart(data["mlt"], height=300, use_container_width=True)

    with eventos_tab:
        st.markdown("<div class='section-heading'>Anomalías persistentes detectadas</div>", unsafe_allow_html=True)
        if criticos.empty:
            st.info("No se detectaron periodos críticos para el intervalo seleccionado.")
        else:
            st.dataframe(criticos, hide_index=True, use_container_width=True)

elif st.session_state.get("dashboard_error"):
    st.markdown(
        "<div class='hero'><div class='eyebrow'>Observatorio de actividad · FIRMS</div>"
        "<div class='hero-title'>Fuego visto desde<br>órbita.</div></div>",
        unsafe_allow_html=True,
    )
    st.error(st.session_state["dashboard_error"])
else:
    st.markdown(
        "<div class='hero'><div class='eyebrow'>Observatorio de actividad · FIRMS</div>"
        "<div class='hero-title'>Fuego visto desde<br>órbita.</div>"
        "<div class='hero-meta'>MODIS + VIIRS &nbsp; / &nbsp; Uruguay</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='empty-state'><span class='micro'>Sin consulta activa</span></div>",
        unsafe_allow_html=True,
    )

if st.session_state.get("dashboard_error") and data:
    st.error(st.session_state["dashboard_error"])
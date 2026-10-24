from datetime import date

import streamlit as st

import MODISyVIIRS_prot1 as core

try:
    import folium
    from folium.plugins import Draw
    from streamlit_folium import st_folium
except Exception:  # pragma: no cover - optional map extras
    folium = None
    Draw = None
    st_folium = None

st.set_page_config(page_title="Focos | MODIS + VIIRS", page_icon="◉", layout="wide")

REGIONES_PREDEFINIDAS = {
    "Uruguay": (-58.5, -35.2, -53.0, -30.0),
    "Argentina": (-73.5, -55.0, -53.0, -21.0),
    "Brasil": (-74.0, -34.0, -34.0, 7.0),
    "América del Sur": (-81.3, -56.0, -34.0, 12.5),
    "Personalizada": None,
}


def format_bbox(bbox):
    return ",".join(f"{value:.4f}" for value in bbox)


def validar_bbox(raw_bbox):
    if raw_bbox is None:
        raise ValueError("Selecciona un BBOX válido.")
    txt = str(raw_bbox).strip()
    if not txt:
        raise ValueError("El BBOX no puede estar vacío.")
    parts = [p.strip() for p in txt.split(",")]
    if len(parts) != 4:
        raise ValueError("El BBOX debe tener 4 valores: O, S, E, N.")
    try:
        west, south, east, north = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError("El BBOX debe contener solo números.") from exc

    if not (-180 <= west <= 180 and -180 <= east <= 180):
        raise ValueError("Las longitudes deben estar entre -180 y 180.")
    if not (-90 <= south <= 90 and -90 <= north <= 90):
        raise ValueError("Las latitudes deben estar entre -90 y 90.")
    if south >= north:
        raise ValueError("La latitud sur debe ser menor que la de norte.")
    if west >= east:
        raise ValueError(
            "La longitud oeste debe ser menor que la de este. Si cruza el antimeridiano, requiere tratamiento especial."
        )
    return (west, south, east, north)


def aplicar_region(region_name):
    if region_name in (None, "Personalizada"):
        return st.session_state.get("bbox_text", "")
    bbox = REGIONES_PREDEFINIDAS[region_name]
    if bbox is None:
        return st.session_state.get("bbox_text", "")
    return format_bbox(bbox)


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

st.session_state.setdefault("bbox_text", "-56.5502,-33.3985,-56.4950,-33.3520")
st.session_state.setdefault("selected_region", "Uruguay")


with st.sidebar:
    st.markdown("<div class='side-mark'>FIRE / OBSERVATION / 01</div>", unsafe_allow_html=True)
    st.markdown("### Paso 1 · Configurar NASA FIRMS")
    st.caption("NASA FIRMS requiere una MAP_KEY para consultar focos activos de MODIS y VIIRS.")
    st.link_button("Obtener mi NASA FIRMS API Key (Gratis)", "https://firms.modaps.eosdis.nasa.gov/api/map_key/")

    with st.expander("¿Cómo obtener mi API Key?", expanded=False):
        st.markdown(
            """
            1. Entrá a la página oficial de NASA FIRMS.\n
            2. Ingresá tu correo electrónico.\n
            3. Solicitá la clave MAP_KEY gratuita.\n
            4. Revisá tu correo para obtener la clave.\n
            5. Pegala en el campo correcto.\n
            La clave solo se usa para consultar los datos y no se comparte en la aplicación.
            """
        )

    with st.form("consulta_firms"):
        key = st.text_input("FIRMS MAP_KEY", type="password", key="map_key_input")
        st.caption("Se mantiene solo durante la sesión actual.")

        st.markdown("### Paso 2 · Seleccionar región")
        regiones = list(REGIONES_PREDEFINIDAS.keys())
        region_choice = st.selectbox(
            "Región predefinida",
            regiones,
            index=regiones.index(st.session_state.get("selected_region", "Uruguay")),
            key="region_select",
        )
        if region_choice != "Personalizada":
            st.session_state["bbox_text"] = aplicar_region(region_choice)
        st.session_state["selected_region"] = region_choice

        with st.expander("¿Qué es un BBOX y cómo funciona?", expanded=False):
            st.markdown(
                """
                Un BBOX (Bounding Box) es un rectángulo geográfico que delimita la zona de análisis.\n
                NASA FIRMS usa cuatro valores en este orden: OESTE, SUR, ESTE, NORTE.\n
                Ejemplo aproximado para Uruguay: `-58.5,-35.2,-53,-30`\n
                - Oeste: límite occidental\n
                - Sur: latitud mínima\n
                - Este: límite oriental\n
                - Norte: latitud máxima\n
                El rectángulo se forma con esos cuatro límites.
                """
            )

        bbox_text = st.text_input(
            "BBox · O, S, E, N",
            st.session_state["bbox_text"],
            help="Formato esperado: oeste,sur,este,norte",
            key="bbox_input",
        )
        st.session_state["bbox_text"] = bbox_text

        try:
            validar_bbox(bbox_text)
            st.success("BBox válido para NASA FIRMS.")
        except ValueError as exc:
            st.warning(str(exc))

        inicio = st.date_input("Fecha inicial", date(2023, 1, 1), key="inicio_input")
        fin = st.date_input("Fecha final", date(2023, 12, 30), key="fin_input")

        ejecutar = st.form_submit_button("Consultar FIRMS", type="primary", use_container_width=True)
        if ejecutar:
            try:
                parsed_bbox = validar_bbox(bbox_text)
                if not key:
                    st.session_state["dashboard_error"] = "Ingresa tu MAP_KEY de FIRMS."
                elif inicio > fin:
                    st.session_state["dashboard_error"] = "La fecha inicial debe ser anterior a la final."
                else:
                    with st.spinner("Consultando detecciones satelitales…"):
                        det = core.descargar_firms(key, parsed_bbox, inicio.isoformat(), fin.isoformat())
                        mlt = core.mlt_empirico(det)
                        conteos = core.conteos_diarios(det)
                        indice, _, _, _, _ = core.armonizar(conteos, mlt)
                        matriz = core.matriz_anual(indice.rolling(7, min_periods=4, center=True).mean())
                        z, valores = core.anomalias(matriz)
                        criticos = core.periodos_criticos(z, valores, matriz.index.values)
                    st.session_state["dashboard"] = {
                        "detecciones": det,
                        "conteos": conteos,
                        "indice": indice,
                        "mlt": mlt,
                        "criticos": criticos,
                        "bbox": parsed_bbox,
                        "inicio": inicio,
                        "fin": fin,
                    }
                    st.session_state.pop("dashboard_error", None)
            except ValueError as exc:
                st.session_state["dashboard_error"] = str(exc)
            except SystemExit as exc:
                st.session_state["dashboard_error"] = str(exc)
            except Exception as exc:
                st.session_state["dashboard_error"] = f"No se pudieron procesar los datos: {exc}"

st.markdown(
    """
    <div class='hero'>
        <div class='eyebrow'>Observatorio de actividad · FIRMS</div>
        <div class='hero-title'>Fuego visto desde<br>órbita.</div>
        <div class='hero-meta'>MODIS + VIIRS &nbsp; / &nbsp; análisis satelital de alto valor para monitoreo ambiental</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "dashboard_error" in st.session_state:
    st.error(st.session_state["dashboard_error"])

with st.expander("Generador visual de BBOX dentro de la página", expanded=True):
    if folium is None or st_folium is None:
        st.info("Para habilitar el mapa interactivo, instala folium y streamlit-folium en el entorno.")
    else:
        mapa = folium.Map(location=[-30, -60], zoom_start=3, tiles="CartoDB positron")
        draw = Draw(
            export=False,
            draw_options={
                "polyline": False,
                "polygon": False,
                "circle": False,
                "marker": False,
                "circlemarker": False,
                "rectangle": True,
                "poly": False,
            },
            edit_options={"edit": True, "remove": True},
        )
        draw.add_to(mapa)

        current_bbox = st.session_state.get("bbox_text", "")
        try:
            west, south, east, north = validar_bbox(current_bbox)
            folium.Rectangle(bounds=[[south, west], [north, east]], color="#df633e", fill=True, fill_opacity=0.2).add_to(mapa)
            mapa.fit_bounds([[south, west], [north, east]])
        except ValueError:
            pass

        map_output = st_folium(mapa, width=900, height=430, key="bbox_map")
        if map_output and map_output.get("last_active_drawing"):
            drawing = map_output["last_active_drawing"]
            if drawing.get("type") == "rectangle":
                bounds = drawing.get("bounds")
                if bounds:
                    all_lats = [float(bounds[0][0]), float(bounds[1][0])]
                    all_lngs = [float(bounds[0][1]), float(bounds[1][1])]
                    west = min(all_lngs)
                    east = max(all_lngs)
                    south = min(all_lats)
                    north = max(all_lats)
                    st.session_state["bbox_text"] = f"{west},{south},{east},{north}"
                    st.rerun()

        if st.button("Limpiar selección", use_container_width=True):
            st.session_state["bbox_text"] = ""
            st.rerun()

if "dashboard" in st.session_state:
    data = st.session_state["dashboard"]
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
    mapa_tab, actividad_tab, sensores_tab, eventos_tab = st.tabs(["Mapa", "Actividad", "Satélites", "Periodos críticos"])

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
else:
    st.markdown(
        "<div class='empty-state'><span class='micro'>Sin consulta activa</span></div>",
        unsafe_allow_html=True,
    )

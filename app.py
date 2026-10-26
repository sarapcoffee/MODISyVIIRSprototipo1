from datetime import date
from io import BytesIO
import json

import altair as alt
import numpy as np
import pandas as pd
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


def bbox_desde_dibujo(drawing):
    if not isinstance(drawing, dict):
        return None

    if drawing.get("type") == "Feature":
        drawing = drawing.get("geometry") or {}

    if drawing.get("type") == "Polygon":
        coordinates = drawing.get("coordinates", [])
        puntos = []

        def recolectar_puntos(coordenadas):
            if (
                isinstance(coordenadas, (list, tuple))
                and len(coordenadas) >= 2
                and all(isinstance(value, (int, float)) for value in coordenadas[:2])
            ):
                puntos.append(coordenadas)
            elif isinstance(coordenadas, (list, tuple)):
                for elemento in coordenadas:
                    recolectar_puntos(elemento)

        recolectar_puntos(coordinates)
        if puntos:
            longitudes = [float(punto[0]) for punto in puntos]
            latitudes = [float(punto[1]) for punto in puntos]
            return (
                min(longitudes),
                min(latitudes),
                max(longitudes),
                max(latitudes),
            )

    bounds = drawing.get("bounds")
    if isinstance(bounds, dict):
        southwest = bounds.get("_southWest")
        northeast = bounds.get("_northEast")
        if isinstance(southwest, dict) and isinstance(northeast, dict):
            try:
                return (
                    min(float(southwest["lng"]), float(northeast["lng"])),
                    min(float(southwest["lat"]), float(northeast["lat"])),
                    max(float(southwest["lng"]), float(northeast["lng"])),
                    max(float(southwest["lat"]), float(northeast["lat"])),
                )
            except (KeyError, TypeError, ValueError):
                return None

    if isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
        try:
            latitudes = [float(bounds[0][0]), float(bounds[1][0])]
            longitudes = [float(bounds[0][1]), float(bounds[1][1])]
        except (IndexError, TypeError, ValueError):
            return None
        return (
            min(longitudes),
            min(latitudes),
            max(longitudes),
            max(latitudes),
        )

    return None


def aplicar_region(region_name):
    if region_name in (None, "Personalizada"):
        return st.session_state.get("bbox_text", "")
    bbox = REGIONES_PREDEFINIDAS[region_name]
    if bbox is None:
        return st.session_state.get("bbox_text", "")
    return format_bbox(bbox)


def preparar_dashboard(det, bbox, inicio, fin, origen, verdad=None):
    mlt = core.mlt_empirico(det)
    conteos = core.conteos_diarios(det)
    indice, fuente, factores, referencia, ajustado = core.armonizar(conteos, mlt)
    matriz = core.matriz_anual(indice.rolling(7, min_periods=4, center=True).mean())
    z, valores_log = core.anomalias(matriz)
    criticos = core.periodos_criticos(z, valores_log, matriz.index.values)
    return {
        "detecciones": det,
        "conteos": conteos,
        "indice": indice,
        "anomalias": serie_anomalias(indice),
        "fuente": fuente,
        "factores": factores,
        "referencia_viirs": referencia,
        "modis_ajustado": ajustado,
        "mlt": mlt,
        "criticos": criticos,
        "bbox": bbox,
        "inicio": pd.Timestamp(inicio),
        "fin": pd.Timestamp(fin),
        "origen": origen,
        "verdad": verdad,
    }


def serie_anomalias(indice):
    serie = indice.rolling(7, min_periods=4, center=True).mean()
    matriz = core.matriz_anual(serie)
    z, _ = core.anomalias(matriz)
    resultado = pd.Series(np.nan, index=serie.index, dtype=float)
    for fila, año in enumerate(matriz.index):
        fechas = pd.date_range(f"{int(año)}-01-01", periods=366 if pd.Timestamp(f"{int(año)}-12-31").dayofyear == 366 else 365)
        for fecha in fechas:
            día = fecha.dayofyear - 1
            if día < z.shape[1] and fecha in resultado.index:
                resultado.loc[fecha] = z[fila, día]
    return resultado


def matriz_exportacion(calendario, titulo, divergente=False):
    matriz = calendario.pivot_table(
        index="año", columns="periodo", values="valor", aggfunc="mean"
    )
    figura, ax = core.plt.subplots(figsize=(12, max(3, len(matriz) * 0.35)))
    cmap = "RdBu_r" if divergente else "YlOrRd"
    imagen = ax.imshow(matriz.to_numpy(), aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_title(titulo)
    ax.set_ylabel("Año")
    ax.set_xlabel("Día del año / período")
    ax.set_yticks(range(len(matriz.index)))
    ax.set_yticklabels(matriz.index)
    figura.colorbar(imagen, ax=ax, label="Anomalía z" if divergente else "Valor diario / agregado")
    figura.tight_layout()
    salida = BytesIO()
    figura.savefig(salida, format="png", dpi=160)
    core.plt.close(figura)
    salida.seek(0)
    return salida.getvalue()


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
    [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"],
    [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] *,
    [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"],
    [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] *,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stMain"] [data-testid="stExpander"] summary,
    [data-testid="stMain"] [data-testid="stExpander"] summary * {
        color: #35443d !important;
    }
    [data-testid="stMain"] [data-testid="stExpander"] summary {
        background: #e7ece6;
        border: 1px solid #cbd5cc;
        border-radius: 4px;
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
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stSidebar"] details summary,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
        color: #eff3e9 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="input"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-testid="stDateInput"] input {
        background: #24483e;
        border-color: #48675d;
        color: #fff;
    }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
        color: #fff !important;
        font-size: .78rem;
        font-weight: 700;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stAlert"] {
        background: #24483e;
        border: 1px solid #48675d;
    }
    [data-testid="stSidebar"] [data-testid="stAlert"] p {
        color: #eff3e9 !important;
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
    [data-testid="stAlert"] p,
    [role="alert"] p {
        color: #3b3210 !important;
    }
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
    .hero .eyebrow, .hero .hero-meta { color: #d7e2da !important; }
    .side-mark { color: #e1eae3; font-size: .68rem; margin: 4px 0 23px; }
    .section-heading {
        color: var(--forest);
        font-size: 1.06rem;
        font-weight: 800;
        margin: 4px 0 2px;
    }
    .section-rule { border-top: 1px solid var(--line); margin: 12px 0 18px; }
    [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 7px; }
    [data-testid="stTabs"] button[role="tab"] {
        color: #35443d !important;
        font-weight: 700;
        border-radius: 4px 4px 0 0;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--forest);
        border-bottom-color: var(--signal);
    }
    div[data-testid="stButton"] > button:not([kind="primary"]),
    div[data-testid="stDownloadButton"] > button {
        background: var(--forest);
        border: 1px solid var(--forest);
        color: #fff !important;
        border-radius: 4px;
        font-weight: 700;
    }
    div[data-testid="stButton"] > button:not([kind="primary"]) *,
    div[data-testid="stDownloadButton"] > button * {
        color: #fff !important;
        -webkit-text-fill-color: #fff !important;
        opacity: 1 !important;
    }
    div[data-testid="stButton"] > button:not([kind="primary"]):hover,
    div[data-testid="stDownloadButton"] > button:hover {
        background: #24483e;
        border-color: #24483e;
        color: #fff !important;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        background: #24483e;
        border-color: #24483e;
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
st.session_state.setdefault("bbox_input", st.session_state["bbox_text"])
# El mapa entrega el BBOX en otro rerun, guardo el cambio pendiente y lo aplico.
if "bbox_pending" in st.session_state:
    st.session_state["bbox_text"] = st.session_state["bbox_pending"]
    st.session_state["bbox_input"] = st.session_state["bbox_pending"]
    st.session_state["selected_region"] = "Personalizada"
    st.session_state["region_select"] = "Personalizada"
    del st.session_state["bbox_pending"]


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
    if region_choice != st.session_state.get("selected_region"):
        if region_choice != "Personalizada":
            st.session_state["bbox_text"] = aplicar_region(region_choice)
            st.session_state["bbox_input"] = st.session_state["bbox_text"]
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
    if st.button("Explorar demostración sintética", use_container_width=True):
        # Ojo: es un ejemplo simulado para explorar la interfaz, NO son datos reales de NASA.
        demo_bbox = validar_bbox(bbox_text)
        demo_inicio = pd.Timestamp("2018-01-01")
        demo_fin = pd.Timestamp("2025-12-31")
        demo_det, demo_verdad = core.simular(demo_inicio, demo_fin, demo_bbox)
        st.session_state["dashboard"] = preparar_dashboard(
            demo_det, demo_bbox, demo_inicio, demo_fin, "Demostración sintética", demo_verdad
        )
        st.session_state.pop("dashboard_error", None)

    if inicio <= fin:
        tramos_estimados = ((fin - inicio).days // 5 + 1) * 3
        st.caption(
            f"FIRMS limita las consultas a 5 días: este intervalo requiere hasta "
            f"{tramos_estimados} tramos entre sensores. Las consultas completadas se guardan en caché."
        )

    ejecutar = st.button("Consultar FIRMS", type="primary", use_container_width=True)
    if ejecutar:
        st.session_state.pop("dashboard_error", None)
        # Así no dejo a la vista el resultado anterior mientras llega la consulta nueva del usuario que lo use.
        st.session_state.pop("dashboard", None)
        try:
            parsed_bbox = validar_bbox(bbox_text)
            if not key:
                st.session_state["dashboard_error"] = "Ingresa tu MAP_KEY de FIRMS."
            elif inicio > fin:
                st.session_state["dashboard_error"] = "La fecha inicial debe ser anterior a la final."
            else:
                with st.spinner("Consultando detecciones satelitales…"):
                    progreso = st.progress(0, text="Preparando consulta a FIRMS…")

                    def actualizar_progreso(completados, total, fuente, desde_cache):
                        origen = "caché" if desde_cache else "FIRMS"
                        progreso.progress(
                            completados / total,
                            text=f"{fuente}: tramo {completados}/{total} completado ({origen})",
                        )

                    try:
                        det = core.descargar_firms(
                            key,
                            parsed_bbox,
                            inicio.isoformat(),
                            fin.isoformat(),
                            progreso=actualizar_progreso,
                        )
                        progreso.progress(1, text="Procesando detecciones…")
                        dashboard = preparar_dashboard(
                            det, parsed_bbox, inicio, fin, "NASA FIRMS"
                        )
                    finally:
                        progreso.empty()
                st.session_state["dashboard"] = dashboard
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
        mapa = folium.Map(
            location=[-30, -60],
            zoom_start=3,
            tiles="OpenStreetMap",
            attr="&copy; OpenStreetMap contributors",
        )
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
            edit_options={"edit": True, "remove": False},
        )
        draw.add_to(mapa)

        current_bbox = st.session_state.get("bbox_text", "")
        try:
            west, south, east, north = validar_bbox(current_bbox)
            folium.Rectangle(bounds=[[south, west], [north, east]], color="#df633e", fill=True, fill_opacity=0.2).add_to(mapa)
            mapa.fit_bounds([[south, west], [north, east]])
        except ValueError:
            pass

        # Mantener esta key estable para que Streamlit no reinicie el estado del mapa en cada rerun!
        map_output = st_folium(mapa, width=900, height=430, key="bbox_map")
        drawing = map_output.get("last_active_drawing") if map_output else None
        if not drawing and map_output and map_output.get("all_drawings"):
            drawing = map_output["all_drawings"][-1]
        bounds = bbox_desde_dibujo(drawing)
        if bounds:
            bbox = ",".join(f"{value:.6f}" for value in bounds)
            if bbox != st.session_state.get("bbox_drawing_applied"):
                st.session_state["bbox_drawing_applied"] = bbox
                if bbox != st.session_state.get("bbox_input", st.session_state["bbox_text"]):
                    st.session_state["bbox_pending"] = bbox
                    st.rerun()

        if st.button("Limpiar selección", use_container_width=True):
            # La caja de texto toma este valor al comienzo del siguiente rerun.
            st.session_state["bbox_pending"] = ""
            st.rerun()

if "dashboard" in st.session_state:
    data = st.session_state["dashboard"]
    detecciones = data["detecciones"]
    conteos = data["conteos"]
    indice = data["indice"]
    criticos = data["criticos"]
    bbox_label = ", ".join(f"{value:.4f}" for value in data["bbox"])
    es_demo = data["origen"] == "Demostración sintética"

    st.markdown(
        f"<div class='hero'><div class='eyebrow'>Observatorio de actividad · FIRMS</div>"
        f"<div class='hero-title'>Observatorio de actividad térmica satelital</div>"
        f"<div class='hero-meta'>{data['inicio']:%d %b %Y} — {data['fin']:%d %b %Y}"
        f" &nbsp; / &nbsp; BBOX {bbox_label} &nbsp; / &nbsp; {data['origen']}</div></div>",
        unsafe_allow_html=True,
    )
    if es_demo:
        st.warning(
            "Demostración sintética: las detecciones y la actividad simulada no son observaciones NASA "
            "y no deben interpretarse como resultados reales."
        )

    detecciones_validas = int(detecciones["conf_ok"].sum())
    dias_con_detecciones = int(conteos.gt(0).any(axis=1).sum())
    metricas = st.columns(4)
    metricas[0].metric("Detecciones", f"{len(detecciones):,}")
    metricas[1].metric("Confianza aceptada", f"{detecciones_validas:,}")
    metricas[2].metric("Días con detecciones", f"{dias_con_detecciones:,}")
    metricas[3].metric("Períodos críticos exploratorios", f"{len(criticos):,}")
    st.caption(
        "Los días sin detecciones no equivalen a días confirmados sin fuego: FIRMS no aporta aquí "
        "metadatos suficientes de cobertura efectiva. Hotspots no equivalen a incendios independientes "
        "ni a superficie quemada."
    )

    calendario_tab, mapa_tab, evolucion_tab, validacion_tab, resumen_tab = st.tabs(
        ["Calendario de incendios", "Mapa satelital", "Evolución histórica", "Validación científica", "Resumen y metodología"]
    )

    with calendario_tab:
        modo = st.radio(
            "Modo de calendario",
            ["Histórico", "Anomalías", "Año detallado"],
            horizontal=True,
            key="calendario_modo",
        )
        resolucion = st.selectbox(
            "Agregación temporal",
            ["Diaria", "Semanal", "Mensual"],
            key="calendario_resolucion",
        )

        columnas_modis = [s for s in conteos.columns if core.SENSORES[s]["familia"] == "MODIS"]
        columnas_viirs = [s for s in conteos.columns if core.SENSORES[s]["familia"] == "VIIRS"]
        variables = ["Índice equivalente a VIIRS", "Detecciones VIIRS", "Detecciones MODIS"]
        if es_demo:
            variables.append("Actividad simulada (verdad)")
        variable = st.selectbox("Variable", variables, key="calendario_variable")

        if variable == "Índice equivalente a VIIRS":
            valores = indice
            unidad = "Detecciones equivalentes/día (experimental)"
        elif variable == "Detecciones VIIRS":
            valores = conteos[columnas_viirs].sum(axis=1, min_count=1)
            unidad = "Detecciones FIRMS VIIRS (sin armonizar)"
        elif variable == "Detecciones MODIS":
            valores = conteos[columnas_modis].sum(axis=1, min_count=1)
            unidad = "Detecciones FIRMS MODIS (sin armonizar)"
        else:
            valores = data["verdad"]
            unidad = "Actividad simulada (unidades arbitrarias)"

        base = pd.DataFrame({"fecha": valores.index, "valor": valores.to_numpy()})
        if modo == "Anomalías":
            # La referencia usa el mismo intervalo consultado, es exploratoria, no una climatología independiente.
            base["valor"] = data["anomalias"].reindex(base["fecha"]).to_numpy()
            unidad = "Anomalía z exploratoria, calculada sobre el período consultado"
            divergente = True
        else:
            divergente = False

        if resolucion == "Semanal":
            base = base.set_index("fecha").resample("W-SUN").agg({"valor": "mean" if modo == "Anomalías" else "sum"}).dropna(how="all")
            base.index.name = "fecha"
            base = base.reset_index()
        elif resolucion == "Mensual":
            base = base.set_index("fecha").resample("MS").agg({"valor": "mean" if modo == "Anomalías" else "sum"}).dropna(how="all")
            base.index.name = "fecha"
            base = base.reset_index()

        base["año"] = base["fecha"].dt.year
        if resolucion == "Diaria":
            base["periodo"] = base["fecha"].dt.dayofyear.astype(str).str.zfill(3)
            base["etiqueta_periodo"] = base["fecha"].dt.strftime("%d %b")
        elif resolucion == "Semanal":
            base["periodo"] = base["fecha"].dt.isocalendar().week.astype(str).str.zfill(2)
            base["etiqueta_periodo"] = "Semana " + base["periodo"]
        else:
            base["periodo"] = base["fecha"].dt.month.astype(str).str.zfill(2)
            base["etiqueta_periodo"] = base["fecha"].dt.strftime("%b")

        if modo == "Año detallado":
            años_disponibles = sorted(base["año"].unique())
            año = st.selectbox("Año", años_disponibles, index=len(años_disponibles) - 1)
            grafico_df = base.loc[base["año"] == año].copy()
            seleccion = alt.selection_point(name="periodo_seleccionado", fields=["fecha"], on="click")
            grafico = (
                alt.Chart(grafico_df)
                .mark_bar(color="#df633e")
                .encode(
                    x=alt.X("etiqueta_periodo:N", sort=None, title="Período"),
                    y=alt.Y("valor:Q", title=unidad),
                    tooltip=[
                        alt.Tooltip("fecha:T", title="Fecha"),
                        alt.Tooltip("valor:Q", title=unidad, format=".3f"),
                    ],
                )
                .add_params(seleccion)
                .properties(height=300)
            )
        else:
            grafico_df = base
            if modo == "Anomalías":
                grafico_df = grafico_df.loc[grafico_df["valor"].notna()]
            seleccion = alt.selection_point(name="periodo_seleccionado", fields=["fecha"], on="click")
            color = alt.Color(
                "valor:Q",
                title=unidad,
                scale=alt.Scale(scheme="redblue" if divergente else "orangered"),
            )
            if divergente:
                color = alt.Color("valor:Q", title=unidad, scale=alt.Scale(scheme="redblue", domainMid=0))
            grafico = (
                alt.Chart(grafico_df)
                .mark_rect()
                .encode(
                    x=alt.X("periodo:O", sort=None, title="Día del año / semana / mes"),
                    y=alt.Y("año:O", sort="ascending", title="Año"),
                    color=color,
                    tooltip=[
                        alt.Tooltip("fecha:T", title="Período"),
                        alt.Tooltip("año:O", title="Año"),
                        alt.Tooltip("etiqueta_periodo:N", title="Intervalo"),
                        alt.Tooltip("valor:Q", title=unidad, format=".3f"),
                    ],
                )
                .add_params(seleccion)
                .properties(height=min(600, max(180, len(grafico_df["año"].unique()) * 28)))
            )

        evento_calendario = st.altair_chart(
            grafico,
            use_container_width=True,
            on_select="rerun",
            selection_mode="periodo_seleccionado",
        )
        estado = getattr(evento_calendario, "selection", {})
        puntos = []
        if isinstance(estado, dict):
            for conjunto in estado.values():
                if isinstance(conjunto, list):
                    puntos.extend(p for p in conjunto if isinstance(p, dict))
        if puntos and puntos[-1].get("fecha"):
            fecha_evento = pd.Timestamp(puntos[-1]["fecha"]).normalize()
            st.session_state["periodo_calendario"] = fecha_evento
            st.session_state["fecha_inspeccion"] = fecha_evento.date()

        if len(base):
            seleccion_fecha = st.date_input(
                "Inspeccionar una fecha",
                value=(
                    st.session_state.get("periodo_calendario", base["fecha"].iloc[-1])
                    if base["fecha"].min() <= pd.Timestamp(
                        st.session_state.get("periodo_calendario", base["fecha"].iloc[-1])
                    ) <= base["fecha"].max()
                    else base["fecha"].iloc[-1]
                ).date(),
                min_value=base["fecha"].min().date(),
                max_value=base["fecha"].max().date(),
                key="fecha_inspeccion",
            )
            fecha_seleccionada = pd.Timestamp(seleccion_fecha)
            st.session_state["periodo_calendario"] = fecha_seleccionada
            if resolucion == "Semanal":
                inicio_periodo = fecha_seleccionada - pd.Timedelta(days=fecha_seleccionada.weekday())
                fin_periodo = inicio_periodo + pd.Timedelta(days=6)
            elif resolucion == "Mensual":
                inicio_periodo = fecha_seleccionada.replace(day=1)
                fin_periodo = inicio_periodo + pd.offsets.MonthEnd(0)
            else:
                inicio_periodo = fin_periodo = fecha_seleccionada

            filas_periodo = detecciones.loc[
                detecciones["fecha"].between(inicio_periodo, fin_periodo)
            ]
            valor_periodo = base.loc[
                (base["fecha"] >= inicio_periodo) & (base["fecha"] <= fin_periodo), "valor"
            ]
            st.markdown(f"**Período seleccionado:** {inicio_periodo:%d %b %Y} — {fin_periodo:%d %b %Y}")
            a, b, c = st.columns(3)
            a.metric("Detecciones registradas", f"{len(filas_periodo):,}")
            b.metric("Sensores con registros", f"{filas_periodo['sensor'].nunique():,}")
            c.metric("Valor de calendario", f"{valor_periodo.sum(min_count=1):,.3f}" if len(valor_periodo) else "Sin dato")
            if filas_periodo.empty:
                st.info("No hay detecciones registradas en este intervalo. La cobertura satelital efectiva es desconocida.")
            else:
                st.dataframe(
                    filas_periodo[[col for col in ("fecha", "hora_utc", "sensor", "lat", "lon", "frp", "diurno", "conf_ok") if col in filas_periodo]],
                    hide_index=True,
                    use_container_width=True,
                )

        st.caption(
            f"Unidad: {unidad}. En este producto los días sin hotspots no permiten inferir cobertura válida "
            "ni ausencia de fuego. Las anomalías son exploratorias; la referencia no es independiente del intervalo consultado."
        )

        csv_calendario = base.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar matriz del calendario (CSV)", csv_calendario, "calendario_actividad.csv", "text/csv")
        if modo == "Anomalías":
            anomalías_csv = pd.DataFrame(
                {"fecha": data["anomalias"].index, "anomalia_z_exploratoria": data["anomalias"].values}
            ).to_csv(index=False).encode("utf-8")
            st.download_button("Descargar anomalías (CSV)", anomalías_csv, "anomalias_exploratorias.csv", "text/csv")
        png = matriz_exportacion(grafico_df, f"{modo}: {variable}", divergente=divergente)
        st.download_button("Descargar calendario como PNG", png, "calendario_actividad.png", "image/png")
        parametros = {
            "fuente": data["origen"],
            "bbox_oeste_sur_este_norte": data["bbox"],
            "inicio": data["inicio"].date().isoformat(),
            "fin": data["fin"].date().isoformat(),
            "modo": modo,
            "resolucion": resolucion,
            "variable": variable,
            "unidad": unidad,
            "metodo_indice": "promedio VIIRS diario; MODIS escalado proporcional con estabilidad horaria aproximada",
            "cobertura": "desconocida en días sin detecciones; FIRMS hotspots no es un producto de cobertura",
        }
        st.download_button(
            "Descargar parámetros (JSON)",
            json.dumps(parametros, ensure_ascii=False, indent=2).encode("utf-8"),
            "parametros_analisis.json",
            "application/json",
        )

    with mapa_tab:
        st.markdown("<div class='section-heading'>Distribución de detecciones registradas</div>", unsafe_allow_html=True)
        sensores_disponibles = sorted(detecciones["sensor"].dropna().unique())
        sensores_mapa = st.multiselect("Sensores", sensores_disponibles, default=sensores_disponibles)
        fecha_mapa = pd.Timestamp(
            st.session_state.get("periodo_calendario", detecciones["fecha"].max())
        )
        resolucion_mapa = st.session_state.get("calendario_resolucion", "Diaria")
        if resolucion_mapa == "Semanal":
            inicio_mapa = fecha_mapa - pd.Timedelta(days=fecha_mapa.weekday())
            fin_mapa = inicio_mapa + pd.Timedelta(days=6)
        elif resolucion_mapa == "Mensual":
            inicio_mapa = fecha_mapa.replace(day=1)
            fin_mapa = inicio_mapa + pd.offsets.MonthEnd(0)
        else:
            inicio_mapa = fin_mapa = fecha_mapa
        st.caption(f"Período vinculado al calendario: {inicio_mapa:%d %b %Y} — {fin_mapa:%d %b %Y}")
        puntos = detecciones.loc[
            detecciones["sensor"].isin(sensores_mapa)
            & detecciones["fecha"].between(inicio_mapa, fin_mapa),
            ["lat", "lon", "sensor"],
        ].dropna()
        if len(puntos) > 5000:
            puntos = puntos.sample(5000, random_state=42)
        if not puntos.empty:
            st.map(puntos, latitude="lat", longitude="lon", zoom=7, size=12, color="#df633e")
        else:
            st.info("No hay detecciones para los sensores seleccionados.")
        if len(detecciones.loc[detecciones["sensor"].isin(sensores_mapa)]) > len(puntos):
            st.caption(f"Se muestra una muestra de {len(puntos):,} detecciones; las métricas usan todos los registros.")
        st.caption("Los puntos representan detecciones térmicas; una detección no confirma por sí sola un incendio.")

    with evolucion_tab:
        frecuencia = st.selectbox("Resolución de la serie", ["Diaria", "Semanal", "Mensual", "Anual"], key="evolucion_freq")
        columnas = {"Diaria": "D", "Semanal": "W-SUN", "Mensual": "MS", "Anual": "YS"}
        serie_cruda = conteos.resample(columnas[frecuencia]).sum(min_count=1)
        indice_agregado = indice.resample(columnas[frecuencia]).mean()
        st.markdown("**Detecciones FIRMS por sensor (sin armonizar)**")
        st.line_chart(serie_cruda, use_container_width=True)
        st.markdown("**Índice equivalente a VIIRS (experimental)**")
        st.line_chart(indice_agregado.rename("Detecciones equivalentes/día"), use_container_width=True)
        st.markdown("**Distribución mensual de detecciones observadas**")
        mensual = conteos.groupby(conteos.index.month).sum(min_count=1)
        mensual.index = [pd.Timestamp(2000, int(mes), 1).strftime("%b") for mes in mensual.index]
        st.bar_chart(mensual, use_container_width=True)
        st.caption(
            "Las series describen detecciones registradas, no incendios confirmados. La agregación no corrige "
            "por nubosidad ni por días con cobertura desconocida."
        )

    with validacion_tab:
        st.warning("Armonización experimental: evidencia de validación independiente insuficiente.")
        filas_factores = []
        for sensor, factor in data["factores"].items():
            filas_factores.append(
                {
                    "sensor": sensor,
                    "factor proporcional MODIS→VIIRS": factor,
                    "días con solapamiento usados": int(
                        (conteos[sensor].notna() & data["referencia_viirs"].notna()).sum()
                    ),
                }
            )
        st.dataframe(pd.DataFrame(filas_factores), hide_index=True, use_container_width=True)
        comparacion = pd.concat(
            [
                data["referencia_viirs"].rename("Referencia VIIRS diaria"),
                data["modis_ajustado"].mean(axis=1, skipna=True).rename("MODIS ajustado experimental"),
            ],
            axis=1,
        )
        st.line_chart(comparacion, use_container_width=True)
        st.markdown(
            """
            El factor proporcional actual se estima con los días disponibles del intervalo consultado; no hay
            separación temporal entre calibración y validación ni intervalos de incertidumbre. La hora local
            calculada a partir de detecciones es una aproximación condicionada a que haya focos detectados, no
            una medición independiente de la órbita. Por ello, continuidad gráfica no implica comparabilidad validada.
            """
        )
        st.markdown("**Factores estimados**")
        st.json({sensor: (float(factor) if pd.notna(factor) else None) for sensor, factor in data["factores"].items()})

    with resumen_tab:
        meses = conteos.sum(axis=1, min_count=1).groupby(conteos.index.month).sum(min_count=1).dropna()
        if meses.empty or meses.sum() == 0:
            st.info("No hay detecciones suficientes para resumir la estacionalidad de este período.")
        else:
            mes_max = pd.Timestamp(2000, int(meses.idxmax()), 1).strftime("%B")
            mes_min = pd.Timestamp(2000, int(meses.idxmin()), 1).strftime("%B")
            st.markdown(
                f"En los datos consultados se registraron **{len(detecciones):,} detecciones**. "
                f"El mes con más detecciones fue **{mes_max}** y el de menos detecciones fue **{mes_min}**. "
                f"El índice experimental tiene un promedio de **{indice.mean():.2f} detecciones equivalentes por día** "
                "en los días donde puede calcularse."
            )
        st.markdown("### Cómo funciona el observatorio")
        st.markdown(
            """
            MODIS y VIIRS son instrumentos satelitales con diferentes resoluciones y sensibilidades. NASA FIRMS
            distribuye sus detecciones de anomalías térmicas. Un BBOX define el rectángulo oeste, sur, este, norte
            que delimita la consulta. El FRP es la potencia radiativa del fuego estimada por el producto; no es
            superficie quemada. El índice experimental aplica una escala estadística y no representa incendios
            independientes ni elimina las diferencias instrumentales.
            """
        )
        with st.expander("Limitaciones científicas y de cobertura"):
            st.markdown(
                """
                - Una detección térmica no equivale necesariamente a un incendio independiente.
                - Hotspots no miden directamente el área quemada; nubes, horarios y sensibilidad afectan el registro.
                - Sin metadatos de pasada o cobertura, los días sin detecciones tienen cobertura desconocida.
                - El índice y las anomalías se calculan sobre el intervalo disponible y no constituyen una validación externa.
                - La aplicación es de apoyo al análisis; no es un sistema oficial de emergencia o pronóstico de riesgo.
                """
            )
        with st.expander("Sensores, órbita estimada y fuentes"):
            st.dataframe(conteos.sum(min_count=1).rename("Detecciones registradas").to_frame(), use_container_width=True)
            st.line_chart(data["mlt"], use_container_width=True)
            st.markdown(
                "[NASA FIRMS Area API](https://firms.modaps.eosdis.nasa.gov/api/area) · "
                "[Disponibilidad de datos](https://firms.modaps.eosdis.nasa.gov/api/data_availability/)"
            )
        if criticos.empty:
            st.info("No se detectaron rachas críticas con el umbral exploratorio actual.")
        else:
            st.markdown("### Rachas de anomalía exploratorias")
            st.dataframe(criticos, hide_index=True, use_container_width=True)
else:
    st.markdown(
        """
        <div class='empty-state'>
            <span class='micro'>Sin análisis activo</span>
            <p>Seleccioná una región y un intervalo para consultar NASA FIRMS, o cargá la demostración sintética.</p>
            <p>Los hotspots son observaciones térmicas y no equivalen necesariamente a incendios confirmados.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

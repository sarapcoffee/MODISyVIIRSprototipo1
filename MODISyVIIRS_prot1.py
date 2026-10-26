"""
PILOTO: Calendario armonizado de actividad de incendios (MODIS + VIIRS)

Idea: los focos de incendio de MODIS (Terra/Aqua) y VIIRS (S-NPP, NOAA-20, NOAA-21)
no se pueden sumar directamente: (1) resoluciones distintas (1 km vs 375 m), y
(2) Terra y Aqua derivan en hora local de paso, así que muestrean otra parte del
ciclo diario del fuego y generan cambios "falsos" en las series.

Este programa piloto hace estas cosas:
  1. Estima la deriva de hora local de cada satélite DESDE LOS PROPIOS DATOS
     (hora de adquisición + longitud -> hora solar local).
  2. Calibra a los sensores MODIS contra VIIRS en el periodo en que ambos estaban en órbita
     estable y arma un índice "equivalente a VIIRS" (MODIS solo se usa mientras
     su hora de paso está cerca de la de diseño).
  3. Genera un calendario año x día, anomalías y periodos críticos.

USO:
  # 1) Demo con datos SINTÉTICOS (no necesita internet ni claves):
  la dir url o nombre del archivo.py con demo

  # 2) Datos reales de FIRMS (necesita MAP_KEY gratuita de NASA FIRMS):
  export FIRMS_MAP_KEY=api_clave
  python url o puedo poner tambien el nombre.py live --bbox=-62,-20,-55,-12 --inicio 2024-01-01 --fin 2024-12-31

  # 3) Órbitas: hora local actual de cada satélite desde sus TLE (CelesTrak)
  lo mismo peero con orbitas
  #    o la serie histórica desde un archivo de TLE (p. ej. de Space-Track):
  lo mismo o un .py orbitas --tle-file aqua_historico.tle --sat Aqua

NOTA SARA (modo live): para calibrar MODIS contra VIIRS el rango de fechas debe incluir
días en que ambos coexistían con MODIS en órbita estable (aprox. 2012-2020). Con una
ventana reciente (p. ej. solo 2024) el índice sale solo de VIIRS. Para rangos largos
conviene la descarga masiva del archivo de FIRMS en vez de la API por tramos.

DEPENDENCIASSSS IMPORTANTES: numpy, pandas, matplotlib  (pip install numpy pandas matplotlib)

El modo `demo` es el que está verificado de punta a punta. Los modos
`live` y `orbitas` necesitan internet: revisar los nombres de fuentes/parámetros de la API de
FIRMS en su documentación antes de usarlos en serio.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
import urllib.request
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# CONFIGURACIÓN
# mlt_diseño = hora local media del paso DIURNO según el diseño de la misión.
SENSORES = {
    "Terra":  {"familia": "MODIS", "inicio": "2000-11-01", "mlt_diseño": 10.5},
    "Aqua":   {"familia": "MODIS", "inicio": "2002-07-03", "mlt_diseño": 13.5},
    "SNPP":   {"familia": "VIIRS", "inicio": "2012-01-20", "mlt_diseño": 13.5},
    "NOAA20": {"familia": "VIIRS", "inicio": "2018-04-01", "mlt_diseño": 13.5},
    "NOAA21": {"familia": "VIIRS", "inicio": "2024-01-17", "mlt_diseño": 13.5},
}
FIN_SNPP = pd.Timestamp("2026-10-31")  # fin anunciado de la distribución de S-NPP

UMBRAL_MLT = 0.25    # horas: tolerancia para considerar "estable" a un sensor MODIS
Z_CRITICO = 1.5      # anomalía (z) a partir de la cual un día es "crítico"
MIN_DIAS_CRITICO = 5 # días consecutivos para declarar un periodo crítico

FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
# (fuente FIRMS sin sufijo, sensor fijo o None si la fuente mezcla Terra+Aqua)
FIRMS_FUENTES = [("MODIS", None), ("VIIRS_SNPP", "SNPP"),
                 ("VIIRS_NOAA20", "NOAA20"), ("VIIRS_NOAA21", "NOAA21")]
NORAD = {"Terra": 25994, "Aqua": 27424, "SNPP": 37849, "NOAA20": 43013, "NOAA21": 54234}


# DATOS SINTÉTICOS (para el modo demo)
# SUPUESTOS del demo (no es data real):
#  - Ciclo diurno del fuego con pico ~15:30 hora local y ancho ~2.5 h.
#  - VIIRS (375 m) detecta 3x más focos que MODIS (1 km).
#  - Curvas de hora local ancladas en fechas publicadas por NASA: Aqua deriva
#    desde 2021 (~13:45 en feb-2023, ~15:50 hacia ago-2026); Terra ~10:30 hasta
#    feb-2020 y ~9:00 proyectado para 2026. Interpolación lineal entre anclas.
_MLT_PUNTOS = {
    "Terra": ([2000.8, 2020.15, 2026.1], [10.5, 10.5, 9.0]),
    "Aqua":  ([2002.5, 2021.5, 2023.1, 2026.6], [13.5, 13.5, 13.75, 15.83]),
}


def perfil_diurno(h):
    """Actividad relativa del fuego según la hora local (supuesto del demo)."""
    d = np.abs(((np.asarray(h, dtype=np.float64) - 15.5 + 12) % 24) - 12)
    return 0.06 + np.exp(-0.5 * (d / 2.5) ** 2)


def año_decimal(fechas):
    f = pd.DatetimeIndex(fechas)
    return np.asarray(f.year + (f.dayofyear - 1) / 365.25, float)


def mlt_simulado(sensor, años):
    if sensor in _MLT_PUNTOS:
        xs, ys = _MLT_PUNTOS[sensor]
        return np.interp(años, xs, ys)
    return np.full(len(años), SENSORES[sensor]["mlt_diseño"], float)


def simular(inicio, fin, bbox, semilla=42):
    """Devuelve (detecciones sintéticas, serie 'verdad' de actividad real)."""
    rng = np.random.default_rng(semilla)
    fechas = pd.date_range(inicio, fin, freq="D")
    doy = fechas.dayofyear.values
    dist = np.abs(doy - 245)
    dist = np.minimum(dist, 365 - dist)
    estacional = 0.03 + np.exp(-0.5 * (dist / 35.0) ** 2)      # pico ~ septiembre
    anual = {int(a): rng.lognormal(0, 0.35) for a in fechas.year.unique()}
    for a, f in {2010: 2.5, 2019: 3.0, 2024: 3.5}.items():       # años extremos
        if a in anual:
            anual[a] *= f
    f_anual = np.array([anual[int(a)] for a in fechas.year])
    ruido = pd.Series(rng.lognormal(0, 0.7, len(fechas))).rolling(5, min_periods=1).mean().values
    verdad = pd.Series(30.0 * estacional * f_anual * ruido, index=fechas, name="verdad")

    W, S, E, N = bbox
    filas = []
    for sensor, cfg in SENSORES.items():
        msk = fechas >= pd.Timestamp(cfg["inicio"])
        if sensor == "SNPP":
            msk &= fechas <= FIN_SNPP
        idx = np.where(msk)[0]
        if len(idx) == 0:
            continue
        mlt = mlt_simulado(sensor, año_decimal(fechas[idx]))
        sens = 1.0 if cfg["familia"] == "MODIS" else 3.0
        for noche, off in ((False, 0.0), (True, 12.0)):          # paso diurno y nocturno
            lam = verdad.values[idx] * sens * perfil_diurno((mlt + off) % 24)
            k = rng.poisson(lam)
            if k.sum() == 0:
                continue
            pos = np.repeat(np.arange(len(idx)), k)
            hl = (mlt[pos] + off + rng.normal(0, 0.4, len(pos))) % 24   # hora local
            lon = rng.uniform(W, E, len(pos))
            lat = rng.uniform(S, N, len(pos))
            filas.append(pd.DataFrame({
                "fecha": fechas[idx][pos], "hora_utc": (hl - lon / 15.0) % 24,
                "lat": lat, "lon": lon, "sensor": sensor, "familia": cfg["familia"],
                "conf_ok": rng.random(len(pos)) < 0.85,
                "frp": rng.gamma(2.0, 8.0, len(pos)), "diurno": not noche}))
    return pd.concat(filas, ignore_index=True), verdad


# 2. DATOS REALES (FIRMS)

def _get(url, reintentos=2):
    for i in range(reintentos):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:
            if i + 1 < reintentos:
                time.sleep(2 * (i + 1))
    return None  # (no se imprime la URL para no exponer la MAP_KEY)


def normalizar_firms(raw, sensor_fijo):
    """Convierte un CSV de FIRMS al esquema común del piloto."""
    t = pd.to_numeric(raw["acq_time"], errors="coerce").fillna(0).astype(int)
    if sensor_fijo is None:  # MODIS: Terra y Aqua vienen mezclados
        sat = raw["satellite"].astype(str).str.lower().str[0]
        sensor = np.where(sat == "t", "Terra", "Aqua")
    else:
        sensor = np.full(len(raw), sensor_fijo, dtype=object)
    conf = raw["confidence"]
    num = pd.to_numeric(conf, errors="coerce")
    # MODIS: 0-100 (>=30 nominal/alta). VIIRS: l/n/h (se conservan n y h).
    ok = np.where(num.notna(), num >= 30,
                  conf.astype(str).str.lower().isin(["n", "h", "nominal", "high"]))
    if "type" in raw:                                  # solo vegetación (type == 0)
        ok = ok & (pd.to_numeric(raw["type"], errors="coerce") == 0).values
    out = pd.DataFrame({
        "fecha": pd.to_datetime(raw["acq_date"]).values,
        "hora_utc": (t // 100 + (t % 100) / 60.0).values,
        "lat": raw["latitude"].values, "lon": raw["longitude"].values,
        "sensor": sensor,
        "conf_ok": ok,
        "frp": raw["frp"].values if "frp" in raw else np.nan,
        "diurno": raw["daynight"].astype(str).str.upper().eq("D").values})
    out["familia"] = out["sensor"].map(lambda s: SENSORES[s]["familia"])
    return out


def descargar_firms(map_key, bbox, inicio, fin, cache="cache_firms", progreso=None):
    """Descarga por tramos de 5 días y guarda en caché (la API limita el rango)."""
    # Un año entero puede superar las 200 peticiones; probá con pocos días si solo estás verificando algo.
    W, S, E, N = bbox
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    hoy = pd.Timestamp.today().normalize()
    inicio = pd.Timestamp(inicio)
    fin = pd.Timestamp(fin)
    frames = []
    tramos = {}
    for fuente, sensor_fijo in FIRMS_FUENTES:
        inicio_sensor = pd.Timestamp(SENSORES[sensor_fijo or "Terra"]["inicio"])
        tramos[fuente] = [
            d0 for d0 in pd.date_range(inicio, fin, freq="5D")
            if d0 + pd.Timedelta(days=min(5, (fin - d0).days + 1) - 1) >= inicio_sensor
            and not (fuente == "VIIRS_SNPP" and d0 > FIN_SNPP)
        ]
    total_tramos = sum(map(len, tramos.values()))
    tramos_completados = 0
    tramos_sin_respuesta = 0

    for fuente, sensor_fijo in FIRMS_FUENTES:
        for d0 in tramos[fuente]:
            dias = int(min(5, (fin - d0).days + 1))
            d1 = d0 + pd.Timedelta(days=dias - 1)
            f_cache = cache / f"{fuente}_{d0:%Y%m%d}_{dias}d_{W}_{S}_{E}_{N}.csv"
            raw = None
            desde_cache = False
            if f_cache.exists():
                try:
                    raw = pd.read_csv(f_cache)
                    desde_cache = True
                except pd.errors.EmptyDataError:
                    # Si quedó un archivo vacío/corrupto, seguimos y volvemos a consultar FIRMS.
                    pass
            if raw is None:
                orden = ["NRT", "SP"] if d0 > hoy - pd.Timedelta(days=90) else ["SP", "NRT"]
                raw = pd.DataFrame()
                for suf in orden:
                    url = f"{FIRMS_BASE}/{map_key}/{fuente}_{suf}/{W},{S},{E},{N}/{dias}/{d0:%Y-%m-%d}"
                    txt = _get(url)
                    if txt and txt.lstrip().startswith("latitude"):
                        raw = pd.read_csv(io.StringIO(txt))
                        break
                    elif txt:
                        print(f"  ! respuesta inesperada de {fuente}_{suf}: {txt[:80]!r}", file=sys.stderr)
                if d1 < hoy - pd.Timedelta(days=3) and len(raw.columns):
                    raw.to_csv(f_cache, index=False)
            if not len(raw.columns):
                # Sin columnas: no hubo respuesta usable. Con cabecera y cero filas: FIRMS sí respondió sin focos.
                tramos_sin_respuesta += 1
            if len(raw):
                frames.append(normalizar_firms(raw, sensor_fijo))
            tramos_completados += 1
            if progreso is not None:
                progreso(tramos_completados, total_tramos, fuente, desde_cache)
        print(f"  {fuente}: listo")
    if not frames:
        if tramos_sin_respuesta:
            sys.exit(
                f"La consulta quedó incompleta: FIRMS no respondió en "
                f"{tramos_sin_respuesta} de {total_tramos} tramos. Revisa la conexión y vuelve a intentar."
            )
        sys.exit("FIRMS respondió, pero no encontró focos para el BBOX y las fechas seleccionadas.")
    det = pd.concat(frames, ignore_index=True)
    # MODIS trae Terra+Aqua en la misma fuente: sensores que no existan aún se descartan
    return det



# 3. ANÁLISIS
def mlt_empirico(det, min_n=30):
    """Hora solar local mediana de las detecciones DIURNAS, por trimestre y sensor.
    (Aproximación: depende algo de la latitud y del borde del barrido.)"""
    d = det[det["diurno"]].copy()
    d["hl"] = (d["hora_utc"] + d["lon"] / 15.0) % 24
    d["trim"] = d["fecha"].dt.to_period("Q").dt.start_time
    g = d.groupby(["trim", "sensor"])["hl"]
    med = g.median().where(g.size() >= min_n).unstack("sensor")
    return med.rolling(3, center=True, min_periods=1).mean()


def conteos_diarios(det):
    d = det[det["conf_ok"]]
    rango = pd.date_range(det["fecha"].min(), det["fecha"].max(), freq="D")
    c = d.groupby(["fecha", "sensor"]).size().unstack("sensor").reindex(rango).fillna(0.0)
    for s in c.columns:
        c.loc[c.index < pd.Timestamp(SENSORES[s]["inicio"]), s] = np.nan
    if "SNPP" in c:
        c.loc[c.index > FIN_SNPP, "SNPP"] = np.nan
    return c


def armonizar(c, mlt_emp, tol=UMBRAL_MLT):
    """Índice 'equivalente a VIIRS'.
    - Referencia: promedio de los VIIRS disponibles cada día.
    - MODIS: se escala a VIIRS con un factor k calculado en los días en que el
      sensor estaba estable (|MLT - MLT de diseño| <= tol) y VIIRS existía.
    - Antes de VIIRS se usa MODIS estable x k; después, VIIRS."""
    # Recordatorio: k se estima dentro de los datos consultados; falta validarlo en otro período independiente.
    viirs = [s for s in c.columns if SENSORES[s]["familia"] == "VIIRS"]
    modis = [s for s in c.columns if SENSORES[s]["familia"] == "MODIS"]
    ref = c[viirs].mean(axis=1, skipna=True) if viirs else pd.Series(np.nan, index=c.index)

    estable, k = {}, {}
    for s in modis:
        if s in mlt_emp:
            m = mlt_emp[s].dropna()
            m.index = pd.DatetimeIndex(m.index) + pd.Timedelta(days=45)   # centro del trimestre
            md = (m.reindex(m.index.union(c.index)).interpolate("time", limit_direction="both")
                  .reindex(c.index))
            estable[s] = (md - SENSORES[s]["mlt_diseño"]).abs() <= tol
        else:
            estable[s] = pd.Series(False, index=c.index)
        m_cal = estable[s] & ref.notna() & c[s].notna()
        den = c.loc[m_cal, s].sum()
        k[s] = ref[m_cal].sum() / den if den > 0 else np.nan

    eq = pd.DataFrame({s: c[s] * k[s] for s in modis}, index=c.index)       # sin máscara
    eq_est = pd.DataFrame({s: eq[s].where(estable[s]) for s in modis}, index=c.index)
    modis_eq = eq_est.mean(axis=1, skipna=True) if modis else pd.Series(np.nan, index=c.index)
    indice = ref.where(ref.notna(), modis_eq)
    fuente = np.where(ref.notna(), "VIIRS", np.where(modis_eq.notna(), "MODIS-estable", "sin dato"))
    return indice, fuente, k, ref, eq


def matriz_anual(serie):
    df = pd.DataFrame({"a": serie.index.year, "d": serie.index.dayofyear, "v": serie.values})
    return df.pivot(index="a", columns="d", values="v").reindex(columns=range(1, 367))


def _suave_circular(v, w=15):
    h = w // 2
    ext = np.concatenate([v[-h:], v, v[:h]])
    return pd.Series(ext).rolling(w, center=True, min_periods=1).mean().values[h:-h]


def anomalias(M):
    """z-score del log(1+índice) respecto a la climatología por día del año."""
    X = np.log1p(M.values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = _suave_circular(pd.Series(np.nanmean(X, axis=0)).bfill().ffill().values)
        sd = np.maximum(_suave_circular(pd.Series(np.nanstd(X, axis=0)).bfill().ffill().values), 0.25)
    return (X - mu) / sd, X


def periodos_criticos(z, X, años):
    """Rachas de >= MIN_DIAS_CRITICO días con z >= Z_CRITICO y actividad sobre la mediana."""
    piso = np.nanmedian(X)
    filas = []
    for i, a in enumerate(años):
        crit = (np.nan_to_num(z[i], nan=-9) >= Z_CRITICO) & (np.nan_to_num(X[i], nan=-9) >= piso)
        j = 0
        while j < len(crit):
            if crit[j]:
                k = j
                while k + 1 < len(crit) and crit[k + 1]:
                    k += 1
                if k - j + 1 >= MIN_DIAS_CRITICO:
                    ini = pd.Timestamp(f"{int(a)}-01-01") + pd.Timedelta(days=j)
                    filas.append({"año": int(a), "inicio": ini.date(),
                                  "fin": (ini + pd.Timedelta(days=k - j)).date(),
                                  "dias": k - j + 1, "z_max": round(float(np.nanmax(z[i, j:k + 1])), 2)})
                j = k + 1
            else:
                j += 1
    return pd.DataFrame(filas)


# 4. GRÁFICOS
COLORES = {"Terra": "#2a9d8f", "Aqua": "#e76f51", "SNPP": "#264653", "NOAA20": "#6a4c93", "NOAA21": "#e9c46a"}


def graf_mlt(mlt_emp, ruta):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for s in mlt_emp.columns:
        ax.plot(mlt_emp.index, mlt_emp[s], label=s, color=COLORES.get(s), lw=1.8)
        ax.axhline(SENSORES[s]["mlt_diseño"], color=COLORES.get(s), ls=":", lw=0.8)
    ax.set_ylabel("Hora solar local del paso diurno (h)")
    ax.set_title("Deriva de hora local estimada desde las detecciones (puntos = hora de diseño)")
    ax.grid(alpha=0.3)
    ax.legend(ncol=5, loc="upper center", frameon=False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    plt.close(fig)


def graf_sesgo(eq, ref, ruta):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for s in eq.columns:
        r = eq[s].rolling(90, min_periods=60).sum() / ref.rolling(90, min_periods=60).sum()
        ax.plot(r.index, r, label=f"{s} (x k) / VIIRS", color=COLORES.get(s), lw=1.8)
    ax.axhline(1.0, color="k", lw=1)
    ax.set_ylabel("Cociente (ventana 90 días)")
    ax.set_title("Sesgo de MODIS respecto a VIIRS: 1.0 = comparable")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    plt.close(fig)


def graf_calendario(M, z, X, ruta):
    años = M.index.values
    meses = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    nombres = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    fig, axs = plt.subplots(2, 1, figsize=(11, 0.34 * len(años) * 2 + 3), sharex=True)
    cm1 = plt.get_cmap("YlOrRd").copy(); cm1.set_bad("#e6e6e6")
    cm2 = plt.get_cmap("RdBu_r").copy(); cm2.set_bad("#e6e6e6")
    im1 = axs[0].imshow(np.ma.masked_invalid(X), aspect="auto", cmap=cm1, interpolation="nearest")
    im2 = axs[1].imshow(np.ma.masked_invalid(z), aspect="auto", cmap=cm2, vmin=-3, vmax=3, interpolation="nearest")
    crit = np.nan_to_num(z, nan=-9) >= Z_CRITICO
    if crit.any():
        axs[1].contour(crit.astype(float), levels=[0.5], colors="k", linewidths=0.5)
    axs[0].set_title("Índice armonizado (equivalente a VIIRS), log(1 + focos/día, suavizado 7 d)")
    axs[1].set_title(f"Anomalía respecto a la climatología (contorno: z ≥ {Z_CRITICO})")
    for ax, im in zip(axs, (im1, im2)):
        ax.set_yticks(range(0, len(años), 2)); ax.set_yticklabels(años[::2])
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    axs[1].set_xticks([m - 1 for m in meses]); axs[1].set_xticklabels(nombres)
    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    plt.close(fig)


# 5. ÓRBITAS (TLE -> hora local del nodo)
def parsear_tles(texto):
    lineas = [l.rstrip() for l in texto.splitlines() if l.strip()]
    return [(lineas[i], lineas[i + 1]) for i in range(len(lineas) - 1)
            if lineas[i].startswith("1 ") and lineas[i + 1].startswith("2 ")]


def hora_local_nodo(l1, l2):
    """Hora solar media local del nodo ascendente (LTAN) y del paso diurno.
    LTAN = 12 h + (RAAN - longitud media del Sol)/15. Precisión ~ minutos."""
    yy, dia = int(l1[18:20]), float(l1[20:32])
    epoca = datetime(2000 + yy if yy < 57 else 1900 + yy, 1, 1) + timedelta(days=dia - 1)
    d = (epoca - datetime(2000, 1, 1, 12)).total_seconds() / 86400.0   # días desde J2000
    raan = float(l2[17:25])
    L_sol = (280.460 + 0.9856474 * d) % 360.0                          # Sol medio
    ltan = (12.0 + (raan - L_sol) / 15.0) % 24.0
    diurno = ltan if 6 <= ltan < 18 else (ltan + 12.0) % 24.0
    return epoca, ltan, diurno


def modo_orbitas(args):
    salida = Path(args.salida); salida.mkdir(parents=True, exist_ok=True)
    if args.tle_file:
        pares = parsear_tles(Path(args.tle_file).read_text())
        filas = [dict(epoca=e, ltan_h=round(a, 3), paso_diurno_h=round(b, 3))
                 for e, a, b in (hora_local_nodo(*p) for p in pares)]
        df = pd.DataFrame(filas).sort_values("epoca")
        df.to_csv(salida / f"mlt_tle_{args.sat}.csv", index=False)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["epoca"], df["paso_diurno_h"], color=COLORES.get(args.sat, "C0"))
        if args.sat in SENSORES:
            ax.axhline(SENSORES[args.sat]["mlt_diseño"], color="k", ls=":")
        ax.set_ylabel("Hora local del paso diurno (h)"); ax.grid(alpha=0.3)
        ax.set_title(f"{args.sat}: hora local desde TLE históricos")
        fig.tight_layout(); fig.savefig(salida / f"04_mlt_tle_{args.sat}.png", dpi=140)
        print(df.tail(10).to_string(index=False))
        return
    filas = []
    for nombre, cat in NORAD.items():
        txt = _get(f"https://celestrak.org/NORAD/elements/gp.php?CATNR={cat}&FORMAT=tle")
        pares = parsear_tles(txt) if txt else []
        if not pares:
            print(f"  ! sin TLE para {nombre}", file=sys.stderr)
            continue
        e, ltan, dia = hora_local_nodo(*pares[0])
        filas.append(dict(satelite=nombre, epoca=e.strftime("%Y-%m-%d"),
                          ltan_h=round(ltan, 2), paso_diurno_h=round(dia, 2),
                          desviacion_vs_diseño_h=round(dia - SENSORES[nombre]["mlt_diseño"], 2)))
    df = pd.DataFrame(filas)
    df.to_csv(salida / "orbitas_actuales.csv", index=False)
    print(df.to_string(index=False))


# 6. PIPELINE
def pipeline(det, salida, verdad=None):
    salida = Path(salida); salida.mkdir(parents=True, exist_ok=True)
    print(f"Detecciones: {len(det):,} | sensores: {sorted(det['sensor'].unique())}")

    mlt_e = mlt_empirico(det)
    graf_mlt(mlt_e, salida / "01_mlt_empirico.png")
    ultimo = mlt_e.ffill().iloc[-1].round(2).to_dict()
    print("Hora local (último trimestre con datos):", ultimo)

    c = conteos_diarios(det)
    indice, fuente, k, ref, eq = armonizar(c, mlt_e)
    print("Factores de escala MODIS -> VIIRS (k):", {s: (round(float(v), 3) if v == v else None) for s, v in k.items()})
    sin_k = [s for s, v in k.items() if v != v]
    if sin_k:
        print(f"  ! Sin calibración para {sin_k}: no hay días con MODIS estable y VIIRS a la vez en esta "
              "ventana. MODIS queda fuera del índice; para calibrar incluye en el rango un periodo "
              "~2012-2020 (órbitas estables).")
    graf_sesgo(eq, ref, salida / "02_sesgo_modis_vs_viirs.png")

    S = indice.rolling(7, min_periods=4, center=True).mean()
    M = matriz_anual(S)
    z, X = anomalias(M)
    graf_calendario(M, z, X, salida / "03_calendario_armonizado.png")

    crit = periodos_criticos(z, X, M.index.values)
    crit.to_csv(salida / "periodos_criticos.csv", index=False)
    out = c.copy()
    out["indice_armonizado"] = indice
    out["fuente"] = fuente
    out.to_csv(salida / "indice_armonizado.csv", index_label="fecha")
    print(f"Periodos críticos detectados: {len(crit)} (ver periodos_criticos.csv)")
    if len(crit):
        print(crit.sort_values("z_max", ascending=False).head(8).to_string(index=False))

    if verdad is not None:   # validación solo posible en el demo
        print("\n--- VALIDACIÓN CONTRA LA VERDAD SIMULADA ---")
        m30 = lambda s: s.rolling(30, min_periods=20).mean()
        alin = pd.DataFrame({"idx": m30(indice), "ver": m30(verdad)}).dropna()
        for nombre, sub in (("solo MODIS (antes de 2012)", alin[alin.index < "2012-01-20"]),
                            ("VIIRS (2012 en adelante)", alin[alin.index >= "2012-01-20"])):
            if len(sub) > 30:
                print(f"  correlación índice vs. verdad, {nombre}: r = {sub['idx'].corr(sub['ver']):.3f}")
        ult = pd.Timestamp(det["fecha"].max()) - pd.Timedelta(days=365)
        for s in eq.columns:
            num = eq[s][eq.index > ult].sum(); den = ref[ref.index > ult].sum()
            print(f"  último año, {s}(x k)/VIIRS = {num / den:.2f}  (1.00 = sin sesgo)")
        errs = []
        for s in mlt_e.columns:
            m = mlt_e[s].dropna()
            sim = mlt_simulado(s, año_decimal(m.index + pd.Timedelta(days=45)))
            errs.append(np.abs(m.values - sim).mean())
        print(f"  error medio de la hora local empírica vs. la simulada: {np.mean(errs):.2f} h")
    print(f"\nSalidas en: {salida.resolve()}")


def parse_bbox(txt):
    try:
        p = [float(x.strip()) for x in txt.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bbox debe contener cuatro números: O,S,E,N") from exc
    if len(p) != 4:
        raise argparse.ArgumentTypeError("bbox debe contener cuatro valores: O,S,E,N")
    west, south, east, north = p
    if not all(np.isfinite(p)):
        raise argparse.ArgumentTypeError("bbox debe contener coordenadas finitas")
    if not (-180 <= west <= 180 and -180 <= east <= 180):
        raise argparse.ArgumentTypeError("las longitudes deben estar entre -180 y 180")
    if not (-90 <= south <= 90 and -90 <= north <= 90):
        raise argparse.ArgumentTypeError("las latitudes deben estar entre -90 y 90")
    if west >= east or south >= north:
        raise argparse.ArgumentTypeError("bbox requiere O < E y S < N")
    return tuple(p)


def main():
    comun = argparse.ArgumentParser(add_help=False)
    comun.add_argument("--salida", default="salida")
    ap = argparse.ArgumentParser(description="Piloto: calendario armonizado de incendios MODIS+VIIRS")
    sub = ap.add_subparsers(dest="modo", required=True)

    d = sub.add_parser("demo", parents=[comun], help="datos sintéticos, sin internet")
    d.add_argument("--bbox", type=parse_bbox, default=parse_bbox("-62,-20,-55,-12"))
    d.add_argument("--inicio", default="2000-11-01")
    d.add_argument("--fin", default="2026-09-30")
    d.add_argument("--semilla", type=int, default=42)

    l = sub.add_parser("live", parents=[comun], help="datos reales de NASA FIRMS")
    l.add_argument("--map-key", default=os.environ.get("FIRMS_MAP_KEY"))
    l.add_argument("--bbox", type=parse_bbox, required=True, help="W,S,E,N (usa --bbox=-62,... si empieza con '-')")
    l.add_argument("--inicio", required=True)
    l.add_argument("--fin", required=True)
    l.add_argument("--cache", default="cache_firms")

    o = sub.add_parser("orbitas", parents=[comun], help="hora local de los satélites desde TLE")
    o.add_argument("--tle-file")
    o.add_argument("--sat", default="Aqua", choices=list(NORAD))

    a = ap.parse_args()
    if a.modo == "demo":
        det, verdad = simular(a.inicio, a.fin, a.bbox, a.semilla)
        pipeline(det, a.salida, verdad)
    elif a.modo == "live":
        if not a.map_key:
            sys.exit("Falta la MAP_KEY (--map-key o variable FIRMS_MAP_KEY).")
        det = descargar_firms(a.map_key, a.bbox, a.inicio, a.fin, a.cache)
        pipeline(det, a.salida)
    else:
        modo_orbitas(a)


if __name__ == "__main__":
    main()
   

"""
HidroGeo Lab — Aplicación de Modelación y Enseñanza en Hidrogeología
Ejecutar: streamlit run app.py
Dependencias: streamlit, plotly, numpy, pandas
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HidroGeo Lab",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# ESTILOS CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0f4c81 0%, #1a73e8 50%, #0d9276 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(15,76,129,0.25);
    }
    .main-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
    .main-header p  { font-size: 1.05rem; margin: 0.4rem 0 0; opacity: 0.88; }

    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        text-align: center;
    }
    .metric-card .label { font-size: 0.78rem; color: #64748b; font-weight: 500; text-transform: uppercase; letter-spacing: 0.6px; }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #0f4c81; margin: 0.3rem 0; }
    .metric-card .unit  { font-size: 0.82rem; color: #94a3b8; }

    .info-box {
        background: #f0f9ff;
        border-left: 4px solid #1a73e8;
        border-radius: 0 10px 10px 0;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        font-size: 0.93rem;
        color: #1e3a5f;
    }
    .warn-box {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        border-radius: 0 10px 10px 0;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        font-size: 0.93rem;
        color: #78350f;
    }
    .ok-box {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        border-radius: 0 10px 10px 0;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        font-size: 0.93rem;
        color: #14532d;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #0f4c81;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem;
    }
    .facies-badge {
        display: inline-block;
        background: linear-gradient(135deg, #0f4c81, #1a73e8);
        color: white;
        padding: 0.4rem 1.2rem;
        border-radius: 999px;
        font-size: 1rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #f1f5f9;
        border-radius: 8px 8px 0 0;
        padding: 0.6rem 1.4rem;
        font-weight: 500;
        color: #475569;
    }
    .stTabs [aria-selected="true"] {
        background: #0f4c81 !important;
        color: white !important;
    }
    .geo-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    }
    .process-item {
        display: flex;
        align-items: flex-start;
        gap: 0.7rem;
        padding: 0.5rem 0;
        font-size: 0.93rem;
        color: #334155;
    }
    .process-icon { font-size: 1.1rem; margin-top: 1px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTES GEOQUÍMICAS
# ─────────────────────────────────────────────
# Pesos equivalentes (meq/mg) = 1 / (peso_molecular / valencia)
EQ_WEIGHTS = {
    "Ca":   0.04990,   # 40.08/2
    "Mg":   0.08226,   # 24.31/2
    "Na":   0.04350,   # 22.99/1
    "K":    0.02558,   # 39.10/1
    "Cl":   0.02821,   # 35.45/1
    "SO4":  0.02082,   # 96.06/2
    "HCO3": 0.01639,   # 61.02/1
    "CO3":  0.03333,   # 60.01/2
}

# ─────────────────────────────────────────────
# FUNCIONES GEOQUÍMICAS
# ─────────────────────────────────────────────
@dataclass
class WaterSample:
    Ca: float; Mg: float; Na: float; K: float
    Cl: float; SO4: float; HCO3: float; CO3: float

    def to_meq(self) -> dict:
        return {ion: getattr(self, ion) * EQ_WEIGHTS[ion]
                for ion in EQ_WEIGHTS}

    def cation_sum(self) -> float:
        m = self.to_meq()
        return m["Ca"] + m["Mg"] + m["Na"] + m["K"]

    def anion_sum(self) -> float:
        m = self.to_meq()
        return m["Cl"] + m["SO4"] + m["HCO3"] + m["CO3"]

    def ionic_balance_error(self) -> float:
        c, a = self.cation_sum(), self.anion_sum()
        if (c + a) == 0:
            return 0.0
        return 100 * (c - a) / (c + a)

    def facies(self) -> str:
        m = self.to_meq()
        c_tot = m["Ca"] + m["Mg"] + m["Na"] + m["K"]
        a_tot = m["Cl"] + m["SO4"] + m["HCO3"] + m["CO3"]
        if c_tot == 0 or a_tot == 0:
            return "Indeterminada"

        c_pct = {ion: 100 * m[ion] / c_tot for ion in ["Ca", "Mg", "Na", "K"]}
        a_pct = {ion: 100 * m[ion] / a_tot for ion in ["Cl", "SO4", "HCO3", "CO3"]}

        cation_label = _dominant_label(c_pct, {"Ca": "Cálcica", "Mg": "Magnésica",
                                                "Na": "Sódica", "K": "Potásica"})
        anion_label  = _dominant_label(a_pct, {"Cl": "Clorurada", "SO4": "Sulfatada",
                                                "HCO3": "Bicarbonatada", "CO3": "Carbonatada"})
        return f"{anion_label} {cation_label}"


def _dominant_label(pct: dict, names: dict) -> str:
    """
    Clasificación según criterio estándar:
    - >50%: ion dominante simple
    - 50-33.33%: co-dominante (top 2)
    - <33.33%: no se considera dominante
    """
    best = max(pct, key=pct.get)
    if pct[best] > 50:
        return names[best]
    # Dos dominantes: top 2 que sumen > 50% o ambos > 25%
    sorted_items = sorted(pct.items(), key=lambda x: x[1], reverse=True)
    top2 = sorted_items[:2]
    if top2[0][1] >= 25 and top2[1][1] >= 25:
        return f"{names[top2[0][0]]}-{names[top2[1][0]]}"
    # Fallback: ion con mayor porcentaje
    return names[best]


# ─────────────────────────────────────────────
# DIAGRAMA DE PIPER — coordenadas ternarias
# ─────────────────────────────────────────────
def ternary_to_cart(a: float, b: float, c: float):
    """Convierte coordenadas ternarias (a,b,c sumando 100) a cartesianas."""
    total = a + b + c
    if total == 0:
        a = b = c = 33.33
    else:
        a, b, c = 100*a/total, 100*b/total, 100*c/total
    x = b + c / 2
    y = c * np.sqrt(3) / 2
    return x, y


def project_to_diamond(cat_x, cat_y, an_x, an_y):
    """
    Proyecta los puntos catiónicos y aniónicos al rombo central del diagrama de Piper.

    Geometría del diagrama de Piper:
    - Triángulo cationes: base [0,0] a [100,0], vértice superior [50, 86.6]
    - Triángulo aniones: base [120,0] a [220,0], vértice superior [170, 86.6]
    - Rombo central: centrado en x=110, con vértices en [60,0], [110,86.6], [160,0], [110,-86.6]

    La proyección estándar mapea:
    - cat_x ∈ [0, 100] → componente horizontal en rombo
    - an_x ∈ [0, 100] (relativo a triángulo anión) → componente horizontal complementario
    - cat_y y an_y → componente vertical combinado

    Fórmula corregida: el rombo tiene ancho 100 (de 60 a 160) y alto 173.2 (de -86.6 a 86.6)
    """
    # Normalizar a escala del rombo
    # cat_x: 0-100 (Ca=0, Mg=100 en base; Na+K=50 en vértice)
    # an_x: 0-100 (Cl=0, SO4=100 en base; HCO3+CO3=50 en vértice)

    # Proyección estándar de Piper:
    # dx = 60 + (cat_x + an_x) / 2  # centro del rombo en x=110, ancho 100
    # dy = (cat_y + an_y) / 2 - 86.6 + 86.6  # ajuste vertical

    # Simplificación: mapeo directo al rombo
    dx = 60 + (cat_x + an_x) / 2
    dy = (cat_y + an_y) / 2 - 86.6 + 86.6  # esto es (cat_y + an_y)/2

    # Ajuste fino: el rombo va de y=-86.6 a y=86.6
    # cat_y va de 0 a 86.6, an_y va de 0 a 86.6
    # El punto central del rombo está en y=0 cuando ambos y son iguales
    dy = (cat_y + an_y) / 2 - 86.6 + 86.6

    # Corrección final: el rombo en el Piper está desplazado
    # Vértices del rombo: (60,0), (110,86.6), (160,0), (110,-86.6)
    # Centro: (110, 0)
    # Mapeo: cat_x=50, an_x=50 → centro del rombo (110, 0)
    #        cat_x=0, an_x=0 → (60, -86.6) esquina inferior izq
    #        cat_x=100, an_x=100 → (160, 86.6) esquina superior der

    dx = 60 + (cat_x + an_x) / 2
    dy = (cat_y - an_y)  # Diferencia vertical para mostrar evolución

    # Ajuste para que el rango sea simétrico alrededor de y=0
    # cat_y máx = 86.6, an_y máx = 86.6
    # dy debería ir de -86.6 a 86.6
    dy = (cat_y - an_y) / 2

    # Verificación: si cat_y = 86.6 (Na+K puro) y an_y = 0 (Cl puro)
    # dy = (86.6 - 0)/2 = 43.3 → arriba del centro, correcto
    # si cat_y = 0 (Ca puro) y an_y = 86.6 (HCO3 puro)
    # dy = (0 - 86.6)/2 = -43.3 → abajo del centro, correcto

    return dx, dy


def build_piper_background() -> go.Figure:
    """Construye el fondo del diagrama de Piper con ejes, etiquetas y triángulos."""
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="#fafbfd",
        plot_bgcolor="#fafbfd",
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False, range=[-20, 220]),
        yaxis=dict(visible=False, range=[-110, 110], scaleanchor="x", scaleratio=1),
        showlegend=True,
        legend=dict(x=0.78, y=0.98, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#e2e8f0", borderwidth=1, font=dict(size=11)),
        font=dict(family="Inter, sans-serif"),
    )

    def tri_coords(pts):
        xs, ys = zip(*pts)
        return list(xs) + [xs[0]], list(ys) + [ys[0]]
    
    y_offset=-100

    # ── Triángulo cationes (izquierda) ──
    cat_tri = [(0, 0), (100, 0), (50, 86.6)]
    cat_tri = [(0, y_offset), (100, y_offset), (50, 86.6 + y_offset)]
    xs, ys = tri_coords(cat_tri)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                             line=dict(color="#94a3b8", width=1.5),
                             showlegend=False, hoverinfo="skip"))

    # ── Triángulo aniones (derecha, offset +120) ──
    an_tri = [(120, 0), (220, 0), (170, 86.6)]
    an_tri = [(120, y_offset), (220, y_offset), (170, 86.6 + y_offset)]
    xs, ys = tri_coords(an_tri)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                             line=dict(color="#94a3b8", width=1.5),
                             showlegend=False, hoverinfo="skip"))

    # ── Rombo central ──
    # Vértices correctos del rombo de Piper:
    # Inferior-izq: (60, 0) — proyección de Ca puro + Cl puro
    # Superior: (110, 86.6) — proyección de Na+K puro + HCO3+CO3 puro  
    # Inferior-der: (160, 0) — proyección de Mg puro + SO4 puro
    # Inferior: (110, -86.6) — proyección opuesta
    diamond_pts = [(60, 0), (110, 86.6), (160, 0), (110, -86.6)]
    xs, ys = tri_coords(diamond_pts)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                             line=dict(color="#94a3b8", width=1.5),
                             showlegend=False, hoverinfo="skip"))

    # ── Etiquetas de vértices ──
    labels = [
        # cationes
        dict(x=-8,  y=-5 + y_offset,   text="Ca²⁺"),
        dict(x=100, y=-5 + y_offset,   text="Mg²⁺"),
        dict(x=48,  y=92 + y_offset,   text="Na⁺+K⁺"),
        # aniones
        dict(x=117, y=-5 + y_offset,   text="Cl⁻"),
        dict(x=222, y=-5 + y_offset,   text="SO₄²⁻"),
        dict(x=167, y=92 + y_offset,   text="HCO₃⁻+CO₃²⁻"),
        # rombo
        dict(x=110, y=96,   text="Na+K / HCO3+CO3"),
        dict(x=50,  y=-10,  text="Ca / Cl"),
        dict(x=170, y=-10,  text="Mg / SO4"),
        dict(x=110, y=-96,  text="Ca+Mg / Cl+SO4"),
    ]
    for lb in labels:
        fig.add_annotation(x=lb["x"], y=lb["y"], text=lb["text"],
                           showarrow=False, font=dict(size=10, color="#334155"),
                           xanchor="center")

    # ── Líneas de cuadrícula internas (25%, 50%, 75%) ──

    y_offset_tri = -100
    Anion_X_offset = 120
    
    # Triángulo cationes
    for pct in [25, 50, 75]:
        p = pct / 100
        # Líneas horizontales (paralelas a base)
        y_h = p * 86.6
        
        x_left = p * 50
        x_right = 100 - p * 50
        fig.add_shape(type="line", x0=x_left, y0=y_h, x1=x_right, y1=y_h,
                      line=dict(color="#cbd5e1", width=0.8, dash="dot"))

        # Líneas paralelas a Ca-Mg (pendiente -√3)
        # Punto en lado Ca-Na+K: x = p*50, y = p*86.6
        fig.add_shape(type="line", x0=p*50, y0=p*86.6, x1=100-p*25, y1=0,
                      line=dict(color="#cbd5e1", width=0.8, dash="dot"))

        # Líneas paralelas a Mg-Na+K (pendiente +√3)
        fig.add_shape(type="line", x0=100-p*50, y0=p*86.6, x1=p*25, y1=0,
                      line=dict(color="#cbd5e1", width=0.8, dash="dot"))

    # Triángulo aniones (offset +120)
    for pct in [25, 50, 75]:
        p = pct / 100
        y_h = p * 86.6
        x_left = 120 + p * 50
        x_right = 220 - p * 50
        fig.add_shape(type="line", x0=x_left, y0=y_h, x1=x_right, y1=y_h,
                      line=dict(color="#cbd5e1", width=0.8, dash="dot"))

        fig.add_shape(type="line", x0=120+p*50, y0=p*86.6, x1=220-p*25, y1=0,
                      line=dict(color="#cbd5e1", width=0.8, dash="dot"))

        fig.add_shape(type="line", x0=220-p*50, y0=p*86.6, x1=120+p*25, y1=0,
                      line=dict(color="#cbd5e1", width=0.8, dash="dot"))

    return fig


def add_piper_point(fig: go.Figure, sample: WaterSample,
                    name: str = "Muestra", color: str = "#ef4444", size: int = 14):
    m = sample.to_meq()
    c_tot = m["Ca"] + m["Mg"] + m["Na"] + m["K"]
    a_tot = m["Cl"] + m["SO4"] + m["HCO3"] + m["CO3"]
    if c_tot == 0 or a_tot == 0:
        return fig

    # cationes: Ca=a, Mg=b, Na+K=c
    ca_p = 100 * m["Ca"] / c_tot
    mg_p = 100 * m["Mg"] / c_tot
    nak_p = 100 * (m["Na"] + m["K"]) / c_tot

    # aniones: Cl=a, SO4=b, HCO3+CO3=c (offset 120)
    cl_p   = 100 * m["Cl"] / a_tot
    so4_p  = 100 * m["SO4"] / a_tot
    hco_p  = 100 * (m["HCO3"] + m["CO3"]) / a_tot

    # puntos en triángulos
    cx, cy = ternary_to_cart(ca_p, mg_p, nak_p)
    ax_raw, ay = ternary_to_cart(cl_p, so4_p, hco_p)
    ax = ax_raw + 120

    # punto en rombo — proyección corregida
    dx, dy = project_to_diamond(cx, cy, ax_raw, ay)

    tooltip = (f"<b>{name}</b><br>"
               f"Ca: {ca_p:.1f}%  Mg: {mg_p:.1f}%  Na+K: {nak_p:.1f}%<br>"
               f"Cl: {cl_p:.1f}%  SO₄: {so4_p:.1f}%  HCO₃: {hco_p:.1f}%")

    kw = dict(mode="markers", name=name, showlegend=True, hovertemplate=tooltip,
              marker=dict(color=color, size=size, line=dict(color="white", width=2),
                          symbol="circle"))

    fig.add_trace(go.Scatter(x=[cx], y=[cy], **kw))
    fig.add_trace(go.Scatter(x=[ax], y=[ay], mode="markers", name=f"{name} (an.)",
                             showlegend=False, hovertemplate=tooltip,
                             marker=dict(color=color, size=size,
                                         line=dict(color="white", width=2))))
    fig.add_trace(go.Scatter(x=[dx], y=[dy], mode="markers",
                             name=f"{name} (rombo)", showlegend=False,
                             hovertemplate=tooltip,
                             marker=dict(color=color, size=size+2, symbol="diamond",
                                         line=dict(color="white", width=2))))
    return fig


def add_reference_zone(fig: go.Figure, zone: dict):
    """Añade polígono de zona de referencia en los tres subcampos del Piper."""
    for region in ["cat", "an", "diamond"]:
        pts = zone.get(region, [])
        if not pts:
            continue
        xs = [p[0] for p in pts] + [pts[0][0]]
        ys = [p[1] for p in pts] + [pts[0][1]]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, fill="toself", mode="lines",
            fillcolor=zone["color"], line=dict(color=zone["line_color"], width=1.5, dash="dash"),
            name=zone["name"], opacity=0.35, hoverinfo="skip"
        ))
    return fig


# ─────────────────────────────────────────────
# DIAGRAMA DE STIFF
# ─────────────────────────────────────────────
def build_stiff_diagram(sample: WaterSample, title: str = "Diagrama de Stiff",
                        color: str = "#1a73e8") -> go.Figure:
    m = sample.to_meq()
    # Filas: Na+K | Ca | Mg  /  Cl | SO4 | HCO3
    rows = [
        ("Na⁺+K⁺",  m["Na"] + m["K"],  "Cl⁻",    m["Cl"]),
        ("Ca²⁺",    m["Ca"],            "SO₄²⁻",  m["SO4"]),
        ("Mg²⁺",    m["Mg"],            "HCO₃⁻",  m["HCO3"] + m["CO3"]),
    ]
    y_pos = [2, 1, 0]

    cat_x = [-r[1] for r in rows]
    an_x  = [ r[3] for r in rows]
    ys    = y_pos

    # polígono cerrado
    poly_x = cat_x + list(reversed(an_x)) + [cat_x[0]]
    poly_y = ys    + list(reversed(ys))    + [ys[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=poly_x, y=poly_y,
        fill="toself", mode="lines+markers",
        fillcolor=color, opacity=0.35,
        line=dict(color=color, width=2.5),
        marker=dict(size=8, color=color),
        name="Firma Química"
    ))

    # línea central
    max_val = max([abs(v) for r in rows for v in [r[1], r[3]]] + [0.01])
    fig.add_shape(type="line", x0=0, y0=-0.3, x1=0, y1=2.3,
                  line=dict(color="#475569", width=1.5, dash="dot"))

    # etiquetas
    for i, (cl, cv, al, av) in enumerate(rows):
        y = y_pos[i]
        fig.add_annotation(x=-cv - 0.05, y=y, text=f"<b>{cl}</b> {cv:.2f}",
                           showarrow=False, xanchor="right",
                           font=dict(size=10, color="#1e3a5f"))
        fig.add_annotation(x=av + 0.05, y=y, text=f"{av:.2f} <b>{al}</b>",
                           showarrow=False, xanchor="left",
                           font=dict(size=10, color="#1e3a5f"))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#0f4c81"), x=0.5),
        xaxis=dict(title="meq/L", zeroline=True, zerolinecolor="#475569",
                   range=[-max_val * 1.5, max_val * 1.5], gridcolor="#e2e8f0"),
        yaxis=dict(showticklabels=False, range=[-0.5, 2.7], gridcolor="#e2e8f0"),
        paper_bgcolor="#fafbfd", plot_bgcolor="#fafbfd",
        showlegend=False,
        margin=dict(l=130, r=130, t=50, b=50),
        font=dict(family="Inter, sans-serif"),
        height=320,
    )
    # etiquetas de cationes / aniones
    fig.add_annotation(x=-max_val * 0.7, y=2.55, text="◀ CATIONES",
                       showarrow=False, font=dict(size=11, color="#0f4c81"))
    fig.add_annotation(x=max_val * 0.7, y=2.55, text="ANIONES ▶",
                       showarrow=False, font=dict(size=11, color="#0d9276"))
    return fig


# ─────────────────────────────────────────────
# DATOS DE AMBIENTES GEOLÓGICOS
# ─────────────────────────────────────────────
GEO_ENVS = {
    "🪨 Rocas Carbonatadas (Calizas)": {
        "icon": "🪨",
        "color": "#1a73e8",
        "line": "#0f4c81",
        "description": """
Las **rocas carbonatadas** (calizas, dolomitas) son los acuíferos más productivos del mundo.
El agua meteórica, enriquecida en CO₂ del suelo, disuelve la calcita y dolomita mediante los procesos:

- **Disolución de calcita:** CaCO₃ + CO₂ + H₂O → Ca²⁺ + 2HCO₃⁻
- **Disolución de dolomita:** CaMg(CO₃)₂ + 2CO₂ + 2H₂O → Ca²⁺ + Mg²⁺ + 4HCO₃⁻
- **Intercambio iónico:** En zonas profundas, Ca²⁺ puede intercambiarse por Na⁺ en arcillas
- **Karstificación:** Apertura de conductos que acelera la circulación y dilución

El resultado es una facies típicamente **Bicarbonatada Cálcica-Magnésica**, con bajo contenido
en cloruros y sulfatos, y pH entre 7 y 8.5.
        """,
        "stiff_sample": WaterSample(Ca=80, Mg=20, Na=5, K=1, Cl=10, SO4=15, HCO3=280, CO3=0),
        "stiff_label": "Stiff típico: ancho en HCO₃ y Ca²⁺",
        "piper_zones": {
            "cat": [(10, 5), (45, 75), (80, 5)],  # zona rica en Ca (base izq.)
            "an":  [(130, 5), (165, 75), (195, 5)],  # zona rica en HCO3 (base der.)
            "diamond": [(60, 60), (80, 85), (100, 65), (85, 40)],
        },
    },
    "🔥 Rocas Ígneas (Silicatadas)": {
        "icon": "🔥",
        "color": "#f59e0b",
        "line": "#b45309",
        "description": """
Los acuíferos en **granitos, basaltos y gneis** dependen de fracturas y meteorización.
Los procesos dominantes son la **hidrólisis de silicatos**:

- **Feldespato potásico:** 2KAlSi₃O₈ + 2H⁺ + H₂O → Al₂Si₂O₅(OH)₄ + 4SiO₂ + 2K⁺
- **Plagioclasa (anortita):** CaAl₂Si₂O₈ + 2CO₂ + 3H₂O → Al₂Si₂O₅(OH)₄ + Ca²⁺ + 2HCO₃⁻
- **Olivino/Piroxeno:** Liberan Mg²⁺ y Fe²⁺ (basaltos)
- **Mineralización lenta:** Aguas jóvenes con baja conductividad eléctrica

La facies es tipicamente **Bicarbonatada Sódico-Cálcica** o **Bicarbonatada Cálcica**,
con concentraciones bajas (TDS < 300 mg/L) y SiO₂ elevado (10–50 mg/L).
        """,
        "stiff_sample": WaterSample(Ca=20, Mg=8, Na=25, K=5, Cl=8, SO4=5, HCO3=120, CO3=0),
        "stiff_label": "Stiff típico: forma estrecha, iones bajos",
        "piper_zones": {
            "cat": [(20, 5), (55, 60), (35, 5)],
            "an":  [(130, 5), (165, 65), (150, 5)],
            "diamond": [(65, 45), (85, 75), (105, 55), (88, 30)],
        },
    },
    "🧂 Evaporitas": {
        "icon": "🧂",
        "color": "#8b5cf6",
        "line": "#6d28d9",
        "description": """
Las **evaporitas** (yeso, halita, anhidrita) producen aguas con mineralización muy elevada.
Los procesos principales son:

- **Disolución de halita (NaCl):** NaCl → Na⁺ + Cl⁻ (muy soluble, TDS puede superar 35 g/L)
- **Disolución de yeso:** CaSO₄·2H₂O → Ca²⁺ + SO₄²⁻ + 2H₂O
- **Disolución de anhidrita:** CaSO₄ → Ca²⁺ + SO₄²⁻
- **Efecto común-ion:** La presencia de Ca²⁺ del yeso puede precipitar CaCO₃
- **Dedolomitización:** SO₄²⁻ del yeso promueve conversión de dolomita a calcita

La facies varía de **Sulfatada Cálcica** (yesos) a **Clorurada Sódica** (halita),
con TDS elevado, dureza extrema y riesgo de contaminación en acuíferos adyacentes.
        """,
        "stiff_sample": WaterSample(Ca=300, Mg=60, Na=800, K=20, Cl=1200, SO4=500, HCO3=150, CO3=0),
        "stiff_label": "Stiff típico: muy ancho (alargado), alto Na y Cl",
        "piper_zones": {
            "cat": [(60, 5), (90, 5), (70, 25)],  # zona Na-K dominante
            "an":  [(135, 5), (180, 5), (160, 35)],  # zona Cl-SO4
            "diamond": [(110, 10), (140, 50), (155, 20), (125, -20)],
        },
    },
    "🌊 Intrusión Marina": {
        "icon": "🌊",
        "color": "#0d9276",
        "line": "#065f46",
        "description": """
La **intrusión salina** ocurre en acuíferos costeros sobreexplotados donde el agua de mar
desplaza al agua dulce. El agua de mar tiene una composición característica:

- **Ion dominante:** Cl⁻ (~19,800 mg/L), Na⁺ (~10,800 mg/L)
- **Mezcla agua dulce-salada:** Proporciones variables según grado de intrusión
- **Intercambio catiónico:** Na⁺ marino desplaza Ca²⁺ y Mg²⁺ adsorbidos en arcillas
- **Reacción de mezcla:** Puede provocar precipitación de CaCO₃ o disolución de dolomita
- **Indicadores diagnósticos:** Ratio Cl⁻/HCO₃⁻ > 0.5, ratio Na⁺/Cl⁻ cercano a 0.86

La facies evoluciona de **Bicarbonatada** (agua dulce) → **Clorurada Mixta** →
**Clorurada Sódica** (agua marina pura), con CE > 3000 µS/cm como señal de alerta.
        """,
        "stiff_sample": WaterSample(Ca=150, Mg=200, Na=2000, K=50, Cl=3500, SO4=400, HCO3=200, CO3=0),
        "stiff_label": "Stiff típico: dominado por Na⁺ y Cl⁻",
        "piper_zones": {
            "cat": [(50, 5), (95, 5), (72, 40)],
            "an":  [(120, 5), (165, 5), (143, 45)],
            "diamond": [(108, 5), (140, 40), (158, 12), (128, -25)],
        },
    },
}

# ─────────────────────────────────────────────
# CABECERA PRINCIPAL
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>💧 HidroGeo Lab</h1>
  <p>Plataforma Interactiva de Modelación y Enseñanza en Hidrogeología</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS PRINCIPALES
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs([
    "🔬 Herramienta de Modelación",
    "📚 Biblioteca de Ambientes Geológicos"
])

# ══════════════════════════════════════════════
# TAB 1: CALCULADORA / MODELACIÓN
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">📥 Ingreso de Concentraciones Iónicas</div>',
                unsafe_allow_html=True)

    with st.container():
        col_cat, col_an = st.columns(2)

        with col_cat:
            st.markdown("**⊕ Cationes (mg/L)**")
            c1, c2 = st.columns(2)
            ca_val  = c1.number_input("Ca²⁺",  min_value=0.0, value=85.0,  step=0.1, format="%.2f")
            mg_val  = c2.number_input("Mg²⁺",  min_value=0.0, value=18.0,  step=0.1, format="%.2f")
            na_val  = c1.number_input("Na⁺",   min_value=0.0, value=12.0,  step=0.1, format="%.2f")
            k_val   = c2.number_input("K⁺",    min_value=0.0, value=2.5,   step=0.1, format="%.2f")

        with col_an:
            st.markdown("**⊖ Aniones (mg/L)**")
            a1, a2 = st.columns(2)
            cl_val   = a1.number_input("Cl⁻",    min_value=0.0, value=22.0,  step=0.1, format="%.2f")
            so4_val  = a2.number_input("SO₄²⁻",  min_value=0.0, value=28.0,  step=0.1, format="%.2f")
            hco3_val = a1.number_input("HCO₃⁻",  min_value=0.0, value=240.0, step=0.1, format="%.2f")
            co3_val  = a2.number_input("CO₃²⁻",  min_value=0.0, value=0.0,   step=0.1, format="%.2f")

    sample = WaterSample(Ca=ca_val, Mg=mg_val, Na=na_val, K=k_val,
                         Cl=cl_val, SO4=so4_val, HCO3=hco3_val, CO3=co3_val)
    meq = sample.to_meq()
    ebi = sample.ionic_balance_error()
    facies_str = sample.facies()

    # ── Métricas principales ──
    st.markdown('<div class="section-title">📊 Resultados del Procesamiento</div>',
                unsafe_allow_html=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">Suma Cationes</div>
          <div class="value">{sample.cation_sum():.3f}</div>
          <div class="unit">meq/L</div>
        </div>""", unsafe_allow_html=True)
    with mc2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">Suma Aniones</div>
          <div class="value">{sample.anion_sum():.3f}</div>
          <div class="unit">meq/L</div>
        </div>""", unsafe_allow_html=True)
    with mc3:
        ebi_color = "#22c55e" if abs(ebi) <= 5 else ("#f59e0b" if abs(ebi) <= 10 else "#ef4444")
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">Error Balance Iónico</div>
          <div class="value" style="color:{ebi_color}">{ebi:+.2f}</div>
          <div class="unit">%</div>
        </div>""", unsafe_allow_html=True)
    with mc4:
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">Facies Hidroquímica</div>
          <div class="value" style="font-size:1rem; padding-top:0.3rem">{facies_str}</div>
          <div class="unit">&nbsp;</div>
        </div>""", unsafe_allow_html=True)

    # EBI interpretación
    if abs(ebi) <= 5:
        st.markdown(f'<div class="ok-box">✅ <b>Balance Iónico Aceptable</b> — EBI = {ebi:+.2f}%. Análisis dentro del rango de precisión analítica estándar (±5%).</div>', unsafe_allow_html=True)
    elif abs(ebi) <= 10:
        st.markdown(f'<div class="warn-box">⚠️ <b>Balance Iónico Marginal</b> — EBI = {ebi:+.2f}%. Posibles errores analíticos o iones no medidos (NH₄⁺, NO₃⁻, Fe²⁺). Revisar metodología.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="warn-box">❌ <b>Balance Iónico Inadmisible</b> — EBI = {ebi:+.2f}%. El análisis debe repetirse. Probables errores de laboratorio o datos incompletos.</div>', unsafe_allow_html=True)

    # ── Tabla meq/L ──
    st.markdown('<div class="section-title">🧪 Conversión mg/L → meq/L</div>',
                unsafe_allow_html=True)

    ions_data = []
    for ion, label, type_ in [
        ("Ca","Ca²⁺","Catión"),("Mg","Mg²⁺","Catión"),("Na","Na⁺","Catión"),("K","K⁺","Catión"),
        ("Cl","Cl⁻","Anión"),("SO4","SO₄²⁻","Anión"),("HCO3","HCO₃⁻","Anión"),("CO3","CO₃²⁻","Anión"),
    ]:
        mg_l = getattr(sample, ion)
        meq_l = meq[ion]
        ions_data.append({"Ion": label, "Tipo": type_,
                          "mg/L": f"{mg_l:.2f}",
                          "Factor eq.": f"{EQ_WEIGHTS[ion]:.5f}",
                          "meq/L": f"{meq_l:.4f}"})

    df = pd.DataFrame(ions_data)
    st.dataframe(df, use_container_width=True, hide_index=True,
                 column_config={
                     "Ion": st.column_config.TextColumn("Ion"),
                     "Tipo": st.column_config.TextColumn("Tipo"),
                     "mg/L": st.column_config.TextColumn("mg/L"),
                     "Factor eq.": st.column_config.TextColumn("Factor eq. (meq/mg)"),
                     "meq/L": st.column_config.TextColumn("meq/L"),
                 })

    # ── Gráficos ──
    st.markdown('<div class="section-title">📈 Diagramas Hidrogeoquímicos</div>',
                unsafe_allow_html=True)

    gcol1, gcol2 = st.columns([3, 2])

    with gcol1:
        st.markdown("**Diagrama de Piper**")
        fig_piper = build_piper_background()
        fig_piper = add_piper_point(fig_piper, sample, "Muestra actual", "#ef4444", 16)
        fig_piper.update_layout(
            title=dict(text=f"Piper — Facies: {facies_str}", font=dict(size=13, color="#0f4c81"), x=0.5),
            height=480,
        )
        st.plotly_chart(fig_piper, use_container_width=True, config={"displayModeBar": False})

    with gcol2:
        st.markdown("**Diagrama de Stiff**")
        fig_stiff = build_stiff_diagram(sample, "Firma Hidroquímica", "#1a73e8")
        st.plotly_chart(fig_stiff, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="info-box">ℹ️ El diagrama de Stiff grafica los cationes (izquierda) y aniones (derecha) en meq/L. La forma y extensión del polígono revela la "firma" química del acuífero.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2: BIBLIOTECA DE AMBIENTES GEOLÓGICOS
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">🌍 Selecciona el Ambiente Geológico</div>',
                unsafe_allow_html=True)

    env_choice = st.selectbox(
        "Ambiente hidrogeológico:",
        options=list(GEO_ENVS.keys()),
        index=0,
        label_visibility="collapsed",
    )

    env = GEO_ENVS[env_choice]

    st.markdown(f"""
    <div class="geo-card">
      <h3 style="color:{env['color']}; margin:0 0 0.8rem">{env_choice}</h3>
      {env['description'].replace(chr(10), '<br>')}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ── Gráficos del ambiente ──
    ecol1, ecol2 = st.columns([3, 2])

    with ecol1:
        st.markdown("**Diagrama de Piper de Referencia**")
        fig_ref = build_piper_background()

        # zona sombreada
        zone_data = {
            "name": f"Zona típica: {env_choice}",
            "color": env["color"],
            "line_color": env["line"],
            "cat": env["piper_zones"].get("cat", []),
            "an":  env["piper_zones"].get("an",  []),
            "diamond": env["piper_zones"].get("diamond", []),
        }
        fig_ref = add_reference_zone(fig_ref, zone_data)

        # punto de muestra de ejemplo
        fig_ref = add_piper_point(fig_ref, env["stiff_sample"],
                                  "Muestra típica", env["color"], 14)
        fig_ref.update_layout(
            title=dict(text=f"Zona típica — {env_choice.split(' ', 1)[-1]}",
                       font=dict(size=13, color="#0f4c81"), x=0.5),
            height=480,
        )
        st.plotly_chart(fig_ref, use_container_width=True, config={"displayModeBar": False})

        st.markdown(f"""
        <div class="info-box">
          🔵 La <b>zona sombreada</b> indica el campo típico donde caen las muestras
          de este ambiente. El punto representa una muestra representativa.
          <br><br>📌 <i>{env["stiff_label"]}</i>
        </div>
        """, unsafe_allow_html=True)

    with ecol2:
        st.markdown("**Diagrama de Stiff Representativo**")
        fig_env_stiff = build_stiff_diagram(
            env["stiff_sample"],
            f"Stiff — {env_choice.split(' ', 1)[-1]}",
            env["color"]
        )
        st.plotly_chart(fig_env_stiff, use_container_width=True,
                        config={"displayModeBar": False})

        # tabla de composición del ejemplo
        m_env = env["stiff_sample"].to_meq()
        ebi_env = env["stiff_sample"].ionic_balance_error()
        facies_env = env["stiff_sample"].facies()

        st.markdown(f"""
        <div class="metric-card" style="margin-top:1rem">
          <div class="label">Facies de Referencia</div>
          <div class="value" style="font-size:0.95rem; padding:0.3rem 0">{facies_env}</div>
          <div class="unit">EBI: {ebi_env:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        comp_data = {
            "Ion": ["Ca²⁺", "Mg²⁺", "Na⁺", "K⁺", "Cl⁻", "SO₄²⁻", "HCO₃⁻"],
            "mg/L": [env["stiff_sample"].Ca, env["stiff_sample"].Mg,
                     env["stiff_sample"].Na, env["stiff_sample"].K,
                     env["stiff_sample"].Cl, env["stiff_sample"].SO4,
                     env["stiff_sample"].HCO3],
            "meq/L": [round(m_env["Ca"],3), round(m_env["Mg"],3),
                      round(m_env["Na"],3), round(m_env["K"],3),
                      round(m_env["Cl"],3), round(m_env["SO4"],3),
                      round(m_env["HCO3"],3)],
        }
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True,
                     hide_index=True)

    # ── Comparativa visual de todos los ambientes ──
    st.markdown('<div class="section-title">🔄 Comparativa de Firmas de Stiff</div>',
                unsafe_allow_html=True)
    st.markdown("Comparación de los diagramas de Stiff típicos para los cuatro ambientes geológicos.")

    all_cols = st.columns(4)
    env_list = list(GEO_ENVS.items())
    for idx, (ename, edata) in enumerate(env_list):
        with all_cols[idx]:
            short_name = ename.split(" ", 1)[-1].split("(")[0].strip()
            fig_c = build_stiff_diagram(edata["stiff_sample"], short_name, edata["color"])
            fig_c.update_layout(height=260, margin=dict(l=100, r=100, t=40, b=30))
            st.plotly_chart(fig_c, use_container_width=True,
                            config={"displayModeBar": False})
            st.markdown(f"<center style='font-size:0.78rem; color:#475569'>{edata['stiff_label']}</center>",
                        unsafe_allow_html=True)

# ── Pie de página ──
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#94a3b8; font-size:0.82rem; padding:0.5rem 0 1rem">
  HidroGeo Lab · Modelación Hidrogeoquímica · 
  Basado en métodos estándar APHA/AWWA · 
  <em>Piper (1944) & Stiff (1951)</em>
</div>
""", unsafe_allow_html=True)

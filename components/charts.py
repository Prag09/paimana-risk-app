"""Themed Plotly chart builders for the PAIMANA app."""

import hashlib
import json
import random

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from components.styles import PALETTE, style_plotly

INDIA_GEOJSON_PATH = "data/india_states_lgd2024.geojson"
NON_STATE_LABELS = {"Offshore", "PAN India"}
# Source: Bharatlas (bharatlas.com/view/lgd_states), LGD 2024 states/UTs layer,
# built from the Local Government Directory - the authoritative Indian
# government source for administrative boundaries. CC0-1.0 / CC-BY-4.0.
# Ring winding reversed to clockwise-exterior (Plotly's geo renderer wants the
# opposite of the GeoJSON RFC 7946 default) and geometry simplified
# (shapely .simplify(0.005), verified against known reference points - see
# scripts/check_map.py) from the ~25MB source download to ~0.9MB. State names
# live in the STNAME_SH property, title-cased here to match our project data
# exactly (source file has them upper-case, e.g. "JAMMU & KASHMIR").
GEOJSON_STATE_KEY = "STNAME_SH"

INDIA_MAP_DISCLAIMER = ("Map boundaries are indicative, sourced from open GIS data, and do not "
                         "necessarily represent authentic international boundaries.")


def _largest_ring(geom):
    rings = geom["coordinates"] if geom["type"] == "Polygon" else [p[0] for p in geom["coordinates"]]
    return max(rings, key=len)


def _point_in_ring(lon, lat, ring):
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > lat) != (yj > lat):
            x_at_lat = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < x_at_lat:
                inside = not inside
        j = i
    return inside


def load_india_geojson():
    with open(INDIA_GEOJSON_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def all_india_state_names(_geojson=None):
    geojson = _geojson or load_india_geojson()
    return [feat["properties"][GEOJSON_STATE_KEY] for feat in geojson["features"]]


@st.cache_data
def state_shapes(_geojson=None):
    """Per-state: a simplified ring for point-containment tests, its bounding box,
    and a centroid fallback - all derived from the geojson geometry."""
    geojson = _geojson or load_india_geojson()
    shapes = {}
    for feat in geojson["features"]:
        name = feat["properties"][GEOJSON_STATE_KEY]
        ring = _largest_ring(feat["geometry"])
        simplified = ring[::max(1, len(ring) // 150)]
        lons = [pt[0] for pt in ring]
        lats = [pt[1] for pt in ring]
        shapes[name] = {
            "ring": simplified,
            "bbox": (min(lons), max(lons), min(lats), max(lats)),
            "centroid": (sum(lats) / len(lats), sum(lons) / len(lons)),
        }
    return shapes


def _place_point(shape, seed_key):
    minlon, maxlon, minlat, maxlat = shape["bbox"]
    seed = int(hashlib.md5(seed_key.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    for _ in range(12):
        lon = rng.uniform(minlon, maxlon)
        lat = rng.uniform(minlat, maxlat)
        if _point_in_ring(lon, lat, shape["ring"]):
            return lat, lon
    return shape["centroid"]


@st.cache_data
def india_risk_map(df, cost_threshold_pct=10):
    """Choropleth of avg cost overrun by state, with individual ongoing projects
    scattered within their actual state boundary (point-in-polygon sampled) so
    judges can see *where* risk sits, not just the aggregate."""
    geojson = load_india_geojson()
    shapes = state_shapes(geojson)
    all_states = all_india_state_names(geojson)

    scoped = df[~df["state"].isin(NON_STATE_LABELS) & df["state"].isin(shapes.keys())]
    state_summary = scoped.groupby("state", as_index=False).agg(
        projects=("project_name", "count"),
        avg_overrun=("cost_overrun_pct", "mean"),
    ).set_index("state").reindex(all_states)

    fig = go.Figure()
    fig.add_trace(go.Choropleth(
        geojson=geojson,
        locations=state_summary.index,
        z=state_summary["avg_overrun"],
        featureidkey=f"properties.{GEOJSON_STATE_KEY}",
        colorscale=[[0, PALETTE["success"]], [0.5, PALETTE["warning"]], [1, PALETTE["danger"]]],
        marker_line_color="rgba(255,255,255,0.35)",
        marker_line_width=0.8,
        colorbar=dict(
            title=dict(text="Avg overrun %", font=dict(color=PALETTE["muted"], size=11)),
            tickfont=dict(color=PALETTE["muted"], size=10),
            len=0.65, thickness=14,
        ),
        customdata=state_summary["projects"].fillna(0),
        hovertemplate="<b>%{location}</b><br>Avg cost overrun: %{z:.1f}%<br>Projects: %{customdata}<extra></extra>",
    ))

    lats, lons, colors, hover = [], [], [], []
    for _, row in scoped.iterrows():
        lat, lon = _place_point(
            shapes[row["state"]],
            f"{row['state']}|{row.get('project_name', '')}|{row.get('report_month', '')}",
        )
        lats.append(lat)
        lons.append(lon)
        overrun = row["cost_overrun_pct"]
        colors.append(PALETTE["danger"] if overrun > cost_threshold_pct else
                       (PALETTE["warning"] if overrun > 5 else PALETTE["success"]))
        hover.append(f"{row['project_name']}<br>{row['state']} · {overrun:.1f}% overrun")

    fig.add_trace(go.Scattergeo(
        lat=lats, lon=lons, mode="markers",
        marker=dict(size=5, color=colors, opacity=0.85, line=dict(width=0.5, color="rgba(0,0,0,0.4)")),
        text=hover, hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))

    fig.update_geos(
        scope="asia", fitbounds="locations", visible=False,
        bgcolor="rgba(0,0,0,0)",
        # Plotly's built-in basemap (land/country/coastline/frame layers) draws
        # its own de facto international borders underneath our choropleth -
        # explicitly disabling every one of them so only our own GeoJSON
        # shapes (which carry the boundary we've vetted) are ever visible.
        showland=False,
        showcountries=False,
        showcoastlines=False,
        showframe=False,
        showsubunits=False,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0), height=560,
        font=dict(color=PALETTE["text"]),
    )
    return fig


def risk_distribution_donut(low, moderate, high):
    fig = go.Figure(go.Pie(
        labels=["Low", "Moderate", "High"],
        values=[low, moderate, high],
        hole=0.68,
        marker=dict(colors=[PALETTE["success"], PALETTE["warning"], PALETTE["danger"]]),
        textinfo="percent",
        textfont=dict(color=PALETTE["text"], size=12),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", showlegend=True,
        legend=dict(orientation="h", y=-0.1, font=dict(color=PALETTE["muted"], size=11)),
        margin=dict(l=10, r=10, t=10, b=10), height=260,
        annotations=[dict(text=f"{high}%<br><span style='font-size:11px;color:{PALETTE['muted']}'>High risk</span>",
                           x=0.5, y=0.5, font=dict(size=20, color=PALETTE["danger"]), showarrow=False)],
    )
    return fig


def cost_overrun_scatter(df):
    fig = px.scatter(df, x="physical_progress_pct", y="cost_overrun_pct", color="ministry",
                      labels={"physical_progress_pct": "Physical progress %", "cost_overrun_pct": "Cost overrun %"})
    fig.update_traces(marker=dict(size=7, opacity=0.75, line=dict(width=0)))
    return style_plotly(fig)


def time_overrun_scatter(df):
    fig = px.scatter(df, x="physical_progress_pct", y="time_overrun_months", color="ministry",
                      labels={"physical_progress_pct": "Physical progress %", "time_overrun_months": "Schedule slip (months)"})
    fig.update_traces(marker=dict(size=7, opacity=0.75, line=dict(width=0)))
    return style_plotly(fig)


def ministry_box(df):
    fig = px.box(df, x="ministry", y="cost_overrun_pct", color_discrete_sequence=[PALETTE["accent"]])
    fig.update_xaxes(tickangle=45)
    return style_plotly(fig, height=380)


def region_bar(region_summary, value_col, label):
    fig = px.bar(region_summary, x="state", y=value_col, labels={"state": "State", value_col: label},
                 color_discrete_sequence=[PALETTE["accent"]])
    fig.update_xaxes(tickangle=45)
    return style_plotly(fig, height=360)


def seasonal_bar(monthly, value_col, label):
    fig = px.bar(x=[f"{m:02d}" for m in monthly.index], y=monthly[value_col].values,
                 labels={"x": "Approval month", "y": label}, color_discrete_sequence=[PALETTE["accent"]])
    return style_plotly(fig)


def histogram(df, col, label, color=None):
    fig = px.histogram(df, x=col, nbins=20, labels={col: label},
                        color_discrete_sequence=[color or PALETTE["accent"]])
    return style_plotly(fig)

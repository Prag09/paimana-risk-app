"""Themed Plotly chart builders for the PAIMANA app."""

import plotly.express as px
import plotly.graph_objects as go
from components.styles import PALETTE, style_plotly


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

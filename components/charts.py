"""Themed Plotly chart builders for the PAIMANA app."""

import plotly.express as px
import plotly.graph_objects as go
from components.styles import PALETTE, style_plotly

NON_STATE_LABELS = {"Offshore", "PAN India"}


def state_ranking_bar(df, top_n=15):
    summary = (
        df[~df["state"].isin(NON_STATE_LABELS)]
        .groupby("state", as_index=False)
        .agg(projects=("project_name", "count"), avg_overrun=("cost_overrun_pct", "mean"))
        .nlargest(top_n, "avg_overrun")
        .iloc[::-1]
    )
    labels = [f"  {v:.1f}%  ·  {n} project{'s' if n != 1 else ''}"
              for v, n in zip(summary["avg_overrun"], summary["projects"])]

    fig = go.Figure(go.Bar(
        x=summary["avg_overrun"], y=summary["state"], orientation="h",
        marker=dict(color=PALETTE["accent"], cornerradius=4),
        customdata=summary["projects"],
        hovertemplate="<b>%{y}</b><br>Avg cost overrun: %{x:.1f}%<br>Projects: %{customdata}<extra></extra>",
    ))
    # Separate left-anchored text layer: Bar's "outside" text overlaps the bar end here.
    fig.add_trace(go.Scatter(
        x=summary["avg_overrun"].clip(lower=0), y=summary["state"], mode="text",
        text=labels, textposition="middle right", cliponaxis=False, hoverinfo="skip",
        textfont=dict(family="IBM Plex Mono, monospace", color=PALETTE["muted"], size=11),
    ))
    fig.update_layout(showlegend=False)
    style_plotly(fig, height=max(320, 30 * len(summary) + 60))
    fig.update_layout(bargap=0.3, margin=dict(l=20, r=150, t=10, b=30))
    fig.update_xaxes(title_text="Avg cost overrun %", ticksuffix="%", showgrid=True,
                     gridcolor="rgba(255,255,255,0.06)", color=PALETTE["muted"])
    fig.update_yaxes(showgrid=False, color=PALETTE["text"])
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

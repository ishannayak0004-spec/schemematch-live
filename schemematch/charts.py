"""Interactive figures created in Python. Plotly.js is served from the installed package."""
import numpy as np
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go

COLORS = ["#008d86", "#f28155", "#6a67b4", "#437dc3", "#a57a17", "#bb5886"]


def style(fig, x, y):
    fig.update_layout(template="plotly_white", margin=dict(l=50,r=24,b=50,t=18),
                      height=350, font=dict(family="Arial",color="#274144"),
                      xaxis_title=x, yaxis_title=y, legend=dict(orientation="h",y=1.15),
                      hovermode="closest", paper_bgcolor="rgba(0,0,0,0)")
    return fig.to_html(include_plotlyjs=False, full_html=False,
                       config={"responsive": True, "displaylogo": False,
                               "modeBarButtonsToRemove": ["select2d", "lasso2d"]})


def figures(report):
    points = report["points"]
    labels = report["point_groups"]
    pca = go.Figure()
    for group in sorted(set(labels)):
        selected = [xy for xy,g in zip(points,labels) if g == group]
        pca.add_scatter(x=[p[0] for p in selected], y=[p[1] for p in selected],
                        mode="markers", name=f"Group {group}",
                        marker=dict(color=COLORS[(group-1) % len(COLORS)], size=8, opacity=.8))
    scores = go.Figure()
    for name, color in zip(("K-means", "Gaussian mixture"), COLORS):
        trials = sorted((t for t in report["trials"] if t["algorithm"] == name), key=lambda t:t["k"])
        scores.add_scatter(x=[t["k"] for t in trials], y=[t["silhouette"] for t in trials],
                           name=name, mode="lines+markers", line=dict(color=color,width=3))
    scores.update_xaxes(dtick=1)
    variance = report["variance"]
    # Smooth a two-dimensional histogram for a rotatable density illustration.
    coordinates = np.asarray(points)
    counts, x_edges, y_edges = np.histogram2d(*coordinates.T, bins=32)
    landscape = gaussian_filter(counts, sigma=1.25)
    surface = go.Figure(go.Surface(
        x=(x_edges[:-1] + x_edges[1:]) / 2,
        y=(y_edges[:-1] + y_edges[1:]) / 2,
        z=landscape.T,
        colorscale=[[0, "#123540"], [.25, "#137787"], [.6, "#55c8ae"], [1, "#dfef8d"]],
        showscale=False,
        hovertemplate="PC1 %{x:.2f}<br>PC2 %{y:.2f}<br>Smoothed count %{z:.2f}<extra></extra>",
        contours_z=dict(show=True, usecolormap=True, project_z=True)))
    surface.update_layout(height=390, margin=dict(l=0,r=0,t=0,b=0),
                          scene=dict(xaxis_title="PC1", yaxis_title="PC2",
                                     zaxis_title="Smoothed count",
                                     aspectratio=dict(x=1.5,y=1,z=.55)))
    terrain = surface.to_html(include_plotlyjs=False, full_html=False,
                              config={"responsive":True, "displaylogo":False})
    return (style(pca, f"PC1 · {variance[0]:.1%} variance", f"PC2 · {variance[1]:.1%} variance"),
            style(scores, "Number of clusters", "Validation silhouette"), terrain)

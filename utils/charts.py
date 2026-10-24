from __future__ import annotations

import plotly.express as px


def line(data, x, y, title, color=None, labels=None):
    return px.line(data, x=x, y=y, color=color, title=title, labels=labels)


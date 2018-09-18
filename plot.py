import numpy as np
import altair as alt
import matplotlib.pyplot as plt


def plot_function(f):
    x = np.linspace(-3, 3, 100)
    plt.plot(x, f(x))
    plt.title(f"{f.__name__}")


def format_kwargs(sep, pattern, **kwargs):
    out = sep.join([
        pattern % (key, val) for key, val in kwargs.items()
    ])
    return out


def as_query(query_format, **kwargs):
    formats = {
        "pandas": (" & ", "%s=='%s'"),
        "vega": (" & ", "(datum.%s=='%s')"),
        "title": (" ", "%s=%s")
    }
    if query_format in formats:
        sep, pattern = formats[query_format]
    else:
        raise ValueError("query_format should be one of %s" % formats.keys())
    query = format_kwargs(sep, pattern, **kwargs)
    return query


def vega_plot(data, x, y,
              color=None, column=None, row=None,
              xscale="log", xdomain=None,
              yscale="linear", yzero=False, ydomain=None,
              mark="-",
              **filters):
    title = as_query("title", **filters)
    vega_filter = as_query("vega", **filters)
    xdomain = xdomain if xdomain else alt.Undefined
    ydomain = ydomain if ydomain else alt.Undefined
    x_scale = alt.Scale(type=xscale, domain=xdomain)
    y_scale = alt.Scale(type=yscale, domain=ydomain, zero=yzero)
    chart = alt.Chart(data, width=300, height=200, title=title)
    if (mark=="-"):
        chart = chart.mark_line()
    elif (mark=="."):
        chart = chart.mark_point()
    elif (mark=="o"):
        chart = chart.mark_circle()
    else:
        raise NotImplementedError(f"Uknown marker {mark}")
    chart = chart.encode(
        x=alt.X(x + ":Q", scale=x_scale),
        y=alt.Y(y + ":Q", scale=y_scale),
        color=color + ":N"
    )
    if vega_filter:
        chart = chart.transform_filter(vega_filter)
    if column:
        chart = chart.encode(column=column + ":O")
    if row:
        chart = chart.e
    return chart

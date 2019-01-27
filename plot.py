import itertools
import numpy as np
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt


def plot_function(f, xlim=(-3,3)):
    x = np.linspace(xlim[0], xlim[1], 100)
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


def get_plot_instructions(data, aes_fields):
    def get_values(field):
        return sorted(data[field].unique())
    aes_palette = dict(
        color=sns.color_palette(n_colors=10),
        linestyle=['-', '--', '-.', ':'],
        marker=['.', 'x', 'o', 'v', '^', '<', '>', 's', 'D']
    )
    field_choices = {
        field: get_values(field)
        for field in aes_fields.values() if field
    }
    field_records = [
        {key: value for key, value in zip(field_choices.keys(), record_values)}
        for record_values in itertools.product(*field_choices.values())
    ]
    instructions = []
    for field_record in field_records:
        query = as_query("pandas", **field_record)
        pos = dict(row=0, column=0)
        options = {}
        title = label = ""
        for aes in ["row", "column", "color", "marker", "linestyle"]:
            field = aes_fields[aes]
            if field:
                value = field_record[field]
                field_values = get_values(field)
                idx_value = field_values.index(value)
                if aes in ["row", "column"]:
                    title += f"{field}={value} "
                    pos[aes] = idx_value
                if aes in ["color", "marker", "linestyle"]:
                    label += f"{field}={value} "
                    palette = aes_palette[aes]
                    options[aes] = palette[idx_value]
        instructions.append(dict(
            query=query, row=pos['row'], column=pos['column'],
            options=options, title=title, label=label
        ))
    return instructions


def qplot(data, x, y,
          color=None, column=None, row=None, marker=None, linestyle=None,
          xlog=False, ylog=False, xlim=None, ylim=None,
          y_markers="-", sharex=True, sharey=True, figsize=4,
          y_legend=False,
          rename=None, font_size=12, usetex=False
          ):
    plt.rc('text', usetex=usetex)
    plt.rc('font', family='serif', size=font_size)
    aes_fields = dict(
        color=color, column=column, row=row, marker=marker, linestyle=linestyle
    )
    instructions = get_plot_instructions(data, aes_fields)
    nrows = max(instruction["row"] for instruction in instructions) + 1
    ncols = max(instruction["column"] for instruction in instructions) + 1
    if isinstance(figsize, float) or isinstance(figsize, int):
        figsize = (figsize * ncols, figsize * nrows)

    def replace(label, mapping):
        if mapping:
            for old, new in mapping.items():
                label = label.replace(old, new)
        return label
    fig, axs = plt.subplots(
        nrows, ncols, squeeze=False, figsize=figsize,
        sharex=sharex, sharey=sharey
    )
    if xlog:
        plt.xscale("log")
    if ylog:
        plt.yscale("log")
    if not isinstance(y, list):
        y = [y]
    if not isinstance(y_markers, list):
        y_markers = [y_markers]
    if len(y)!=len(y_markers):
        raise ValueError("y and y_markers must have the same length")
    for instruction in instructions:
        if instruction["query"]:
            df = data.query(instruction["query"])
        else:
            df = data
        ax = axs[instruction['row'], instruction['column']]
        title = replace(instruction['title'], rename)
        for i, (y_var, y_marker) in enumerate(zip(y, y_markers)):
            if y_legend:
                label = instruction['label'] + " " + y_var
            else:
                label = instruction['label'] if i ==0 else ""
            label = replace(label, rename)
            if y_marker=="-":
                ax.plot(df[x], df[y_var], **instruction['options'], label=label)
            else:
                ax.scatter(
                    df[x], df[y_var], marker=y_marker,
                    **instruction['options'], label=label
                )
        ax.set(title=title)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    ylabel= "" if y_legend else ", ".join(y)
    xlabel = replace(x, rename)
    ylabel = replace(ylabel, rename)
    any_label = any(instruction['label'] for instruction in instructions)
    for ax in axs.ravel():
        ax.set(xlabel=xlabel, ylabel=ylabel)
        if y_legend or any_label:
            ax.legend()
    fig.tight_layout()


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
    if (mark == "-"):
        chart = chart.mark_line()
    elif (mark == "."):
        chart = chart.mark_point()
    elif (mark == "o"):
        chart = chart.mark_circle()
    else:
        raise NotImplementedError(f"Uknown marker {mark}")
    chart = chart.encode(
        x=alt.X(x + ":Q", scale=x_scale),
        y=alt.Y(y + ":Q", scale=y_scale),
    )
    if vega_filter:
        chart = chart.transform_filter(vega_filter)
    if color:
        chart = chart.encode(color=color + ":N")
    if column:
        chart = chart.encode(column=column + ":O")
    if row:
        chart = chart.encode(row=row + ":O")
    return chart

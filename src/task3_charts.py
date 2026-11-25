"""Render the Task 3 probability and convergence figures with Vega-Altair."""

import altair as alt
import vl_convert as vlc


INK = "#17332B"
MUTED = "#506660"
GRID = "#DDE5E1"
TEAL = "#19725E"
CLAY = "#AC4D37"
BLUE = "#48639A"
PURPLE = "#79518B"
PLAYERS = ["Novak Djokovic", "Rohan Bopanna", "Alexander Zverev", "Martin Damm"]


def _axis(labels, horizontal=False, title=None):
    scale_options = {"values": labels}
    if not horizontal:
        labels_scientific = {
            0: "0", 1: "1", 1e-2: "10⁻²", 1e-4: "10⁻⁴", 1e-6: "10⁻⁶",
            1e-8: "10⁻⁸", 1e-10: "10⁻¹⁰",
        }
        expr = " : ".join(
            f"datum.value === {tick} ? '{labels_scientific.get(tick, f'{tick:g}')}'" for tick in labels
        ) + " : datum.label"
        scale_options["labelExpr"] = expr
    return alt.Axis(title=title, grid=not horizontal, **scale_options)


def _style(chart):
    return (
        chart
        .configure(background="white", font="Arial", padding={"left": 24, "top": 18, "right": 25, "bottom": 18})
        .configure_view(stroke=None)
        .configure_axis(
            domainColor="#9CB0A8",
            domainWidth=1,
            tickColor="#9CB0A8",
            tickSize=4,
            gridColor=GRID,
            gridOpacity=1,
            labelColor=MUTED,
            labelFontSize=16,
            labelPadding=9,
            titleColor=INK,
            titleFontSize=18,
            titleFontWeight="normal",
            titlePadding=14,
        )
        .configure_title(
            anchor="start", color=INK, fontSize=31, fontWeight="bold",
            subtitleColor=MUTED, subtitleFontSize=17, subtitlePadding=7,
            offset=19,
        )
        .configure_legend(
            labelFontSize=16, labelColor=INK, symbolStrokeWidth=3.3,
            symbolSize=200, symbolType="stroke", labelLimit=250,
        )
    )


def plot_convergence(runs, tolerance):
    """Plot the total change at each update and mark each successful stopping point."""
    starts = list(runs)
    max_steps = max(result.iterations for result in runs.values())
    positive_changes = [change for result in runs.values() for change in result.changes if change > 0]
    smallest_positive = min(positive_changes, default=tolerance)
    threshold_label = "10⁻¹⁰" if tolerance == 1e-10 else f"{tolerance:.0e}"
    walk_values = []
    endpoints = []
    for name, result in runs.items():
        walk_values.extend(
            {"start": name, "step": step, "change": float(change)}
            for step, change in enumerate(result.changes, 1)
            if change > 0
        )
        if result.converged and result.changes[-1] > 0:
            endpoints.append({
                "start": name,
                "step": result.iterations,
                "change": float(result.changes[-1]),
                "label": f"{result.iterations} steps",
            })

    run_color = alt.Color(
        "start:N", scale=alt.Scale(domain=starts, range=[TEAL, CLAY]),
        legend=alt.Legend(title=None, orient="top", direction="horizontal", columns=2, offset=14),
    )
    run_dash = alt.StrokeDash(
        "start:N", scale=alt.Scale(domain=starts, range=[[1, 0], [8, 5]]),
        legend=alt.Legend(title=None, orient="top", direction="horizontal", columns=2, offset=14),
    )
    step_x = alt.X(
        "step:Q", scale=alt.Scale(domain=[0, max_steps + 1], nice=False),
        axis=_axis([0, 5, 10, 15, 20, 25, 30], horizontal=True, title="Update number"),
    )
    log_y = alt.Y(
        "change:Q", scale=alt.Scale(type="log", domain=[min(tolerance / 10, smallest_positive / 2), 3], nice=False),
        axis=_axis([1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10], title="Total absolute change (L1, log scale)"),
    )
    lines = (
        alt.Chart(alt.Data(values=walk_values)).mark_line(strokeWidth=3.3, clip=True)
        .encode(x=step_x, y=log_y, color=run_color, strokeDash=run_dash)
    )
    last_points = alt.Chart(alt.Data(values=endpoints)).encode(x=step_x, y=log_y, color=run_color)
    threshold = alt.Chart(alt.Data(values=[{"change": tolerance, "step": 0.7}]))
    threshold_rule = threshold.mark_rule(color="#758783", strokeDash=[4, 5], strokeWidth=1.6).encode(y=log_y)
    threshold_text = (
        threshold.mark_text(text=f"Stopping threshold  {threshold_label}", align="left", dy=-12, color=MUTED, fontSize=16)
        .encode(x=step_x, y=log_y)
    )
    return _style(
        alt.layer(
            threshold_rule,
            lines,
            last_points.mark_point(filled=True, size=91, stroke="white", strokeWidth=1.4),
            last_points.mark_text(dx=-12, dy=19, align="right", fontSize=17, fontWeight="bold").encode(text="label:N"),
            threshold_text,
        ).properties(
            title=alt.Title(
                "Convergence of the original walk",
                subtitle="Each dot marks the first update below the stopping threshold.",
            ),
            width=894, height=418,
        )
    )


def plot_player_probabilities(runs, player_ids):
    """Plot selected player trajectories for both starting distributions."""
    max_steps = max(result.iterations for result in runs.values())
    probability_values = []
    for start, result in runs.items():
        for name in PLAYERS:
            values = result.history[:, player_ids[name]]
            probability_values.extend(
                {
                    "start": f"Start at {start}", "player": name,
                    "step": step, "probability": float(value),
                }
                for step, value in enumerate(values)
            )

    player_legend = alt.Legend(title=None, orient="bottom", direction="horizontal", columns=4, offset=22)
    player_color = alt.Color(
        "player:N",
        scale=alt.Scale(domain=PLAYERS, range=[TEAL, CLAY, BLUE, PURPLE]),
        legend=player_legend,
    )
    player_dash = alt.StrokeDash(
        "player:N",
        scale=alt.Scale(domain=PLAYERS, range=[[1, 0], [8, 4], [9, 3, 2, 3], [2, 4]]),
        legend=player_legend,
    )
    base = alt.Chart(alt.Data(values=probability_values)).encode(
        x=alt.X(
            "step:Q", scale=alt.Scale(domain=[0, max_steps + 1], nice=False),
            axis=_axis([0, 5, 10, 15, 20, 25, 30], horizontal=True, title="Step"),
        ),
        y=alt.Y(
            "probability:Q", scale=alt.Scale(type="symlog", constant=1e-7, domain=[-1.2e-8, 1.25], nice=False),
            axis=_axis([0, 1e-6, 1e-4, 1e-2, 1], title="Player probability (symlog scale)"),
        ),
        color=player_color,
    )
    paths = base.mark_line(strokeWidth=2.8).encode(strokeDash=player_dash)
    initial_points = base.transform_filter("datum.step === 0 && datum.probability === 1").mark_point(
        size=77, filled=True, stroke="white", strokeWidth=1.4,
    )
    return _style(
        alt.layer(paths, initial_points).properties(width=435, height=360).facet(
            column=alt.Column(
                "start:N", sort=[f"Start at {name}" for name in runs],
                header=alt.Header(title=None, labelFontSize=18, labelFontWeight="bold", labelColor=INK, labelPadding=13),
            ),
            spacing=25,
        ).properties(
            title=alt.Title(
                "Player probabilities from two starting points",
                subtitle="Both panels use the same scale and show exact zeros.",
            )
        )
    )


def chart_png(chart, scale=3):
    """Render a chart to PNG bytes for display in a Jupyter notebook."""
    return vlc.vegalite_to_png(chart.to_dict(), scale=scale)


def save_figures(runs, player_ids, figures_dir, tolerance):
    """Save scalable SVGs and report-compatible PNGs from the same charts."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    charts = (
        ("task3_player_probabilities", plot_player_probabilities(runs, player_ids)),
        ("task3_convergence", plot_convergence(runs, tolerance)),
    )
    for filename, chart in charts:
        specification = chart.to_dict()
        (figures_dir / f"{filename}.svg").write_text(
            vlc.vegalite_to_svg(specification), encoding="utf-8"
        )
        (figures_dir / f"{filename}.png").write_bytes(
            vlc.vegalite_to_png(specification, scale=3)
        )

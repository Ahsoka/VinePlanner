"""PGFPlots/TikZ export for 2D courses and pressure profiles.

Why this exists rather than a ``tikzplotlib`` call
--------------------------------------------------
``tikzplotlib`` walks a finished Matplotlib figure and re-emits its artists.
An obstacle course is mostly ``matplotlib.patches.PathPatch`` objects created
by ``shapely.plotting.plot_polygon``, which is exactly the artist class that
converter does not handle, so the obstacles come out empty. Nothing about the
figure is worth recovering anyway: the geometry is already available as
polygons and vertex arrays, and going straight from those to coordinates is
both shorter and far more controllable than post-processing a rendered figure.

What is emitted
---------------
A single ``tikzpicture`` holding one ``axis``. Every visual choice is routed
through a named style (``vine/obstacle``, ``vine/pressure path``, and so on)
defined in a preamble block, so the figure can be restyled in the paper --
column width, line weights, colours, fonts -- without regenerating anything.
The block is emitted inline behind an ``\\ifdefined`` guard by default, which
makes each file safe to ``\\input`` on its own and safe to ``\\input`` several
times in one document; pass ``styles="preamble"`` to move it to the preamble
once, which is what you want for a paper with several of these figures.

Sizing
------
``axis equal image`` is used rather than a computed ``height``. A maze read at
the wrong aspect ratio is a misleading figure, and this is the one pgfplots
key that guarantees equal unit vectors regardless of what the surrounding
document does to ``\\linewidth``.

Clipping
--------
The axis clips, which reproduces Matplotlib's clipping of obstacle patches to
the axes box. The start and target markers are the deliberate exception: they
sit exactly on the corners of a typical course, so they are emitted as named
coordinates inside the axis and drawn after it, which is the TikZ equivalent
of ``clip_on=False``.
"""

import numpy as np
import matplotlib.ticker as ticker

from .constants import _LENGTH_TO_FEET
from .utils import (
    path_metrics,
    path_pressure,
    pressure_profile,
    legend_label,
    normalize_legend_fields,
)


# Matplotlib's colours, so the exported figure and the on-screen figure are
# the same picture.
_COLORS = {
    "vineobstacle": "1F77B4",   # tab:blue
    "vinedistance": "D62728",   # tab:red
    "vineangle": "FF7F0E",      # tab:orange
    "vinepressure": "FF00FF",   # magenta
    "vinestart": "008000",      # green
    "vinetarget": "800080",     # purple
}

# Guard macro. Emitting the styles inline is only safe if a document holding
# several of these figures defines them once.
_GUARD = r"\vineplannerstyles"

_ANGLE_ALIASES = {
    "degree": (180.0 / np.pi, "deg"),
    "degrees": (180.0 / np.pi, "deg"),
    "deg": (180.0 / np.pi, "deg"),
    "radian": (1.0, "rad"),
    "radians": (1.0, "rad"),
    "rad": (1.0, "rad"),
}

_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


# --------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------- #

def _escape(text: str) -> str:
    """Escape LaTeX specials in text that came from a caller."""
    return "".join(_LATEX_ESCAPES.get(character, character) for character in str(text))


def _number(value: float, precision: int) -> str:
    """Fixed-point with trailing zeros stripped.

    Coordinates dominate the file, and pgfplots parses every one of them, so
    ``12.5`` rather than ``12.500000`` is worth the two lines. Fixed point
    rather than ``%g`` because pgfplots' fpu reader is happier without
    exponents.
    """
    text = f"{float(value):.{precision}f}"

    if "." in text:
        text = text.rstrip("0").rstrip(".")

    return "0" if text in {"", "-", "-0"} else text


def _coordinates(points, precision: int, per_line: int = 6, indent: str = "  ") -> str:
    """``(x,y) (x,y) ...`` wrapped so the file stays readable and diffable."""
    points = np.asarray(points, dtype=float)

    formatted = [
        f"({_number(x, precision)},{_number(y, precision)})"
        for x, y in points[:, :2]
    ]

    if len(formatted) <= per_line:
        return " ".join(formatted)

    lines = [
        " ".join(formatted[start:start + per_line])
        for start in range(0, len(formatted), per_line)
    ]

    return "\n" + "\n".join(f"{indent}{line}" for line in lines) + "\n"


def _scale_dimension(dimension: str, factor: float) -> str:
    """Multiply a LaTeX length such as ``\\linewidth`` or ``8cm`` by ``factor``.

    Only used for the pressure plot, where the aspect ratio is a free choice
    and a height has to be produced from the requested width.
    """
    text = str(dimension).strip()

    index = 0
    while index < len(text) and (text[index].isdigit() or text[index] in ".-+"):
        index += 1

    magnitude = float(text[:index]) if index else 1.0
    unit = text[index:].strip()

    return f"{_number(magnitude * factor, 4)}{unit}"


def _resolve_angle_units(angle_units: str) -> tuple[float, str]:
    resolved = _ANGLE_ALIASES.get(str(angle_units).lower())

    if resolved is None:
        raise ValueError("angle_units must be 'degrees' or 'radians'.")

    return resolved


def _length_conversion(source_units: str, target_units: str) -> float:
    for units in (source_units, target_units):
        if str(units).lower() not in _LENGTH_TO_FEET:
            raise ValueError(
                "length units must be one of: 'ft', 'in', 'm', or 'cm'; "
                f"got {units!r}."
            )

    return (
        _LENGTH_TO_FEET[str(source_units).lower()]
        / _LENGTH_TO_FEET[str(target_units).lower()]
    )


def _tick_options(axis_name: str, limits, display_scale: float, precision: int) -> list[str]:
    """Explicit ticks that read as round numbers in display units.

    Mirrors ``obstacle_course._relabel_axis``: the tick *labels* are chosen to
    be round, which means picking them in display space and mapping the
    positions back into coordinate space. Letting pgfplots choose and only
    rewriting the labels would put round positions on the wrong scale and
    print things like ``2.08333``.
    """
    low, high = (value * display_scale for value in limits)

    locator = ticker.MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10])
    display_ticks = np.asarray(locator.tick_values(low, high), dtype=float)

    inside = (display_ticks >= low - 1e-9) & (display_ticks <= high + 1e-9)
    display_ticks = display_ticks[inside]

    positions = ",".join(
        _number(tick / display_scale, precision) for tick in display_ticks
    )
    labels = ",".join(f"{tick:g}" for tick in display_ticks)

    return [f"{axis_name}tick={{{positions}}}", f"{axis_name}ticklabels={{{labels}}}"]


# --------------------------------------------------------------------- #
# Styles
# --------------------------------------------------------------------- #

def tikz_preamble() -> str:
    """Colour, library and style definitions the exported figures rely on.

    Put this in the document preamble once and export with
    ``styles="preamble"``, or let each figure carry a guarded copy of it
    (the default). Redefining any of these styles after this block, in the
    preamble or locally, restyles the figures without regenerating them.
    """
    definitions = "\n".join(
        f"\\definecolor{{{name}}}{{HTML}}{{{value}}}"
        for name, value in _COLORS.items()
    )

    return rf"""% ---- vine_planner figure styles -------------------------------------
% Styles applied to \addplot must live in the /pgfplots/ key path; those
% applied to \node and \fill must live in /tikz/. They are therefore split
% between the two blocks below rather than gathered by subject.
\usetikzlibrary{{shapes.geometric}}
{definitions}
\pgfplotsset{{
  vine/course axis/.style={{
    axis equal image,
    axis on top=false,
    tick align=outside,
    tick pos=left,
    grid=both,
    grid style={{draw=black!15, line width=0.3pt}},
    legend cell align=left,
    legend style={{font=\scriptsize, fill=white, fill opacity=1,
                  text opacity=1, draw=black!60, rounded corners=1pt}},
  }},
  vine/pressure axis/.style={{
    axis on top=false,
    tick align=outside,
    tick pos=left,
    grid=both,
    grid style={{draw=black!15, line width=0.3pt}},
    legend cell align=left,
    legend style={{font=\scriptsize, fill=white, fill opacity=1,
                  text opacity=1, draw=black!60, rounded corners=1pt}},
  }},
  vine/obstacle/.style={{
    draw=black, line width=0.5pt,
    fill=vineobstacle, fill opacity=0.4, draw opacity=0.4,
  }},
  vine/vertex/.style={{
    only marks, mark=*,
    mark options={{fill=vineobstacle, draw=vineobstacle, scale=0.35}},
  }},
  vine/distance path/.style={{
    draw=vinedistance, line width=1.1pt, solid, line join=round, line cap=round,
  }},
  vine/angle path/.style={{
    draw=vineangle, line width=1.1pt, dashed, line join=round, line cap=round,
  }},
  vine/pressure path/.style={{
    draw=vinepressure, line width=1.1pt, dash dot, line join=round, line cap=round,
  }},
  vine/burst/.style={{
    draw=black, line width=0.9pt, densely dashed,
  }},
}}
\tikzset{{
  vine/obstacle label/.style={{font=\tiny\bfseries, inner sep=0pt}},
  vine/start/.style={{fill=vinestart, draw=none}},
  vine/target/.style={{
    star, star points=5, star point ratio=2.3,
    minimum size=7pt, inner sep=0pt, draw=none, fill=vinetarget,
  }},
}}
% The target is drawn outside the axis (see below), so its legend swatch has
% to be built by hand. Declaring the node as a plot mark keeps the two from
% drifting: there is no filled five-pointed mark in pgfplots to approximate
% it with.
\pgfdeclareplotmark{{vinetarget}}{{\node[vine/target]{{}};}}
% ---------------------------------------------------------------------"""


def _style_block(styles: str) -> str:
    """Inline the style block, guarded, or leave it to the preamble."""
    if styles in {"preamble", None, False}:
        return ""

    if styles not in {"inline", True}:
        raise ValueError("styles must be 'inline' or 'preamble'.")

    return (
        f"\\ifdefined{_GUARD}\\else\n"
        f"\\gdef{_GUARD}{{}}\n"
        f"{tikz_preamble()}\n"
        f"\\fi\n\n"
    )


def _wrap(body: str, standalone: bool) -> str:
    if not standalone:
        return body

    return (
        "\\documentclass[border=2pt]{standalone}\n"
        "\\usepackage{pgfplots}\n"
        "\\pgfplotsset{compat=1.18}\n"
        "\\begin{document}\n"
        f"{body}"
        "\\end{document}\n"
    )


def _axis(options: list[str], content: list[str]) -> str:
    rendered = "".join(f"    {option},\n" for option in options)
    body = "\n".join(content)

    return (
        "\\begin{tikzpicture}\n"
        f"  \\begin{{axis}}[\n{rendered}  ]\n"
        f"{body}\n"
        "  \\end{axis}\n"
    )


# --------------------------------------------------------------------- #
# Course export
# --------------------------------------------------------------------- #

def course_tikz(
    course,
    start=None,
    end=None,
    distance_path=None,
    angle_path=None,
    pressure_path=None,
    *,
    show_vertices: bool = True,
    show_labels: bool = False,
    show_grid: bool = True,
    show_legend: bool = False,
    legend_info=("length", "angle", "pressure"),
    legend_position: str = "south west",
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    course_units: str = "ft",
    length_units: str = None,
    angle_units: str = "degrees",
    model_parameters: dict = None,
    width: str = r"\linewidth",
    precision: int = 4,
    styles: str = "inline",
    standalone: bool = False,
    filename=None,
) -> str:
    """Return the ``tikzpicture`` source for a 2D course and its paths.

    The unit handling matches ``ObstacleCourse.plot_path`` exactly, and for
    the same reason: ``course_units`` says what one coordinate unit physically
    is and is what the pressure model is evaluated in, while ``length_units``
    only changes the numbers printed on the ticks and in the legend. The
    geometry is always emitted in coordinate units.

    ``show_legend`` defaults to ``False`` here, unlike the Matplotlib method.
    A journal figure carries its description in the caption, and the entries
    are wide enough to cover a good part of a dense maze.

    ``legend_info`` selects which quantities each entry reports and in which
    order. Accepts any of ``"length"`` (``"distance"`` is a synonym),
    ``"angle"`` and ``"pressure"``, or ``()``/``False`` for names only.

    ``filename`` writes the result as well as returning it.
    """
    if course.dimension != 2:
        raise ValueError(
            "course_tikz only handles 2D courses; a voxel course has no "
            "faithful 2D pgfplots rendering."
        )

    model_parameters = dict(model_parameters or {})
    fields = normalize_legend_fields(legend_info)

    normalized_course_units = str(course_units).lower()
    if normalized_course_units not in _LENGTH_TO_FEET:
        raise ValueError("course_units must be one of: 'ft', 'in', 'm', or 'cm'.")

    normalized_length_units = (
        normalized_course_units if length_units is None else str(length_units).lower()
    )
    display_scale = _length_conversion(normalized_course_units, normalized_length_units)
    angle_scale, angle_label = _resolve_angle_units(angle_units)

    # ---- axis options ------------------------------------------------ #
    options = [
        "vine/course axis",
        f"width={width}",
        f"xmin=0, xmax={_number(course.width, precision)}",
        f"ymin=0, ymax={_number(course.height, precision)}",
    ]

    axis_x = xlabel if xlabel is not None else f"$x$~({normalized_length_units})"
    axis_y = ylabel if ylabel is not None else f"$y$~({normalized_length_units})"

    options.append(f"xlabel={{{axis_x}}}")
    options.append(f"ylabel={{{axis_y}}}")

    if title:
        options.append(f"title={{{_escape(title)}}}")

    if not show_grid:
        options.append("grid=none")

    if display_scale != 1.0:
        options += _tick_options("x", (0.0, course.width), display_scale, precision)
        options += _tick_options("y", (0.0, course.height), display_scale, precision)

    if show_legend:
        options.append(f"legend pos={legend_position}")

    # ---- obstacles --------------------------------------------------- #
    content = []
    vertices = []

    obstacles = [
        obstacle
        for obstacle in (course.obstacles or [])
        if obstacle is not None and not obstacle.is_empty
    ]

    if obstacles:
        content.append("    % obstacles")

    for index, obstacle in enumerate(obstacles, start=1):
        rings = [obstacle.exterior, *obstacle.interiors]

        # A hole would have to be subtracted, not merely stroked. Nothing in
        # the generator produces one, so this is a guard rather than a
        # feature: emitting the ring alone would silently draw a filled
        # obstacle where a ring-shaped one was meant.
        if obstacle.interiors:
            raise NotImplementedError(
                f"obstacle {index} has an interior ring; only simply "
                "connected obstacles are supported."
            )

        coordinates = np.asarray(rings[0].coords, dtype=float)
        vertices.append(coordinates[:-1])

        content.append(
            "    \\addplot[vine/obstacle, forget plot] coordinates "
            f"{{{_coordinates(coordinates, precision, indent='      ')}}};"
        )

    # ---- vertex markers ---------------------------------------------- #
    # Every polygon corner, which is what plot_polygon(add_points=True) draws.
    # course.vertices() is the smaller planner-visible set; it is not used
    # here so the exported figure matches the Matplotlib one.
    if show_vertices and vertices:
        stacked = np.vstack(vertices)
        content.append("    % obstacle corners")
        content.append(
            "    \\addplot[vine/vertex, forget plot] coordinates "
            f"{{{_coordinates(stacked, precision, indent='      ')}}};"
        )

    if show_labels:
        content.append("    % obstacle labels")
        for index, obstacle in enumerate(obstacles, start=1):
            point = obstacle.representative_point()
            content.append(
                "    \\node[vine/obstacle label] at "
                f"(axis cs:{_number(point.x, precision)},"
                f"{_number(point.y, precision)}) {{{index}}};"
            )

    # ---- goals ------------------------------------------------------- #
    # Named inside the axis, drawn after it: the markers sit on the course
    # corners and would otherwise be cut in half by the axis clip.
    #
    # Legend entries are paired with legend images by position, so each
    # \addlegendentry is emitted immediately after the \addplot or
    # \addlegendimage it belongs to. The goals come first because that is the
    # order Matplotlib's legend puts them in.
    trailer = []

    for name, point, style, legend_name, image in (
        (
            "vinestartpoint", start, "vine/start", "start",
            "only marks, mark=*, mark options={fill=vinestart, draw=none}",
        ),
        (
            "vinetargetpoint", end, "vine/target", "target",
            "only marks, mark=vinetarget",
        ),
    ):
        if point is None:
            continue

        content.append(
            f"    \\coordinate ({name}) at "
            f"(axis cs:{_number(point[0], precision)},"
            f"{_number(point[1], precision)});"
        )

        if show_legend:
            content.append(f"    \\addlegendimage{{{image}}}")
            content.append(f"    \\addlegendentry{{{_escape(legend_name)}}}")

        if style == "vine/start":
            trailer.append(f"  \\fill[{style}] ({name}) circle[radius=2.6pt];")
        else:
            trailer.append(f"  \\node[{style}] at ({name}) {{}};")

    # ---- paths ------------------------------------------------------- #
    # Worst-first, so the minimum-pressure path lands on top where they
    # overlap -- the same ordering the Matplotlib method uses.
    drawn = (
        (distance_path, "min distance", "vine/distance path"),
        (angle_path, "min angle", "vine/angle path"),
        (pressure_path, "min pressure", "vine/pressure path"),
    )

    for path, name, style in drawn:
        if path is None:
            continue

        coordinates = np.asarray(path.coords, dtype=float)
        path_length, path_angle = path_metrics(path)

        # Evaluated in course units. The display setting cannot reach the
        # model, only the number printed.
        pressure = path_pressure(
            path,
            length_units=normalized_course_units,
            **model_parameters,
        )

        content.append(f"    % {name}")
        content.append(
            f"    \\addplot[{style}{'' if show_legend else ', forget plot'}] "
            f"coordinates {{{_coordinates(coordinates, precision, indent='      ')}}};"
        )

        if show_legend:
            entry = legend_label(
                name,
                fields,
                length=path_length * display_scale,
                angle=path_angle * angle_scale,
                pressure=pressure,
                length_units=normalized_length_units,
                angle_units=angle_label,
            )
            content.append(f"    \\addlegendentry{{{_escape(entry)}}}")

    body = _style_block(styles) + _axis(options, content)

    if trailer:
        body += "\n".join(trailer) + "\n"

    body += "\\end{tikzpicture}\n"

    document = _wrap(body, standalone)

    if filename is not None:
        with open(filename, "w", encoding="utf-8") as handle:
            handle.write(document)

    return document


# --------------------------------------------------------------------- #
# Pressure-profile export
# --------------------------------------------------------------------- #

def pressure_tikz(
    planner,
    *,
    show_grid: bool = True,
    show_legend: bool = False,
    legend_info=("length", "angle", "pressure"),
    legend_position: str = "north west",
    burst_pressure: float = None,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    length_units: str = None,
    angle_units: str = "degrees",
    points_per_segment: int = 1,
    width: str = r"\linewidth",
    aspect: float = 0.62,
    precision: int = 5,
    styles: str = "inline",
    standalone: bool = False,
    filename=None,
) -> str:
    """Return the ``tikzpicture`` source for the pressure-versus-length plot.

    ``points_per_segment`` defaults to 1 rather than the Matplotlib method's
    50. Under the recursive model the tip is distal to every bend, so freshly
    deployed body adds drag at the constant rate ``f`` and the profile is
    *exactly* affine within each segment; the jump at a vertex is carried by
    the duplicated abscissa the profile already emits. One interval per
    segment is therefore not an approximation, it is the exact curve, and it
    keeps a forty-segment path at eighty coordinates instead of four thousand.

    ``burst_pressure`` adds a horizontal reference line, which is usually the
    point of showing this plot in the paper: the minimum-distance profile
    crossing it is the failure claim.
    """
    available = (planner.distance_path, planner.angle_path, planner.pressure_path)

    if all(path is None for path in available):
        raise RuntimeError(
            "No path has been computed. Call compute_distance_path(), "
            "compute_angle_path() or compute_pressure_path() before "
            "pressure_tikz()."
        )

    fields = normalize_legend_fields(legend_info)

    normalized_length_units, display_scale = planner._resolve_display_units(length_units)
    angle_scale, angle_label = _resolve_angle_units(angle_units)

    options = [
        "vine/pressure axis",
        f"width={width}",
        f"height={_scale_dimension(width, aspect)}",
        f"xlabel={{{xlabel if xlabel is not None else f'Path length~({normalized_length_units})'}}}",
        f"ylabel={{{ylabel if ylabel is not None else 'Pressure~(psi)'}}}",
        "xmin=0",
    ]

    if title:
        options.append(f"title={{{_escape(title)}}}")

    if not show_grid:
        options.append("grid=none")

    if show_legend:
        options.append(f"legend pos={legend_position}")

    styled = (
        (planner.distance_path, "min distance", "vine/distance path", None),
        (planner.angle_path, "min angle", "vine/angle path", planner.total_angle),
        (planner.pressure_path, "min pressure", "vine/pressure path", None),
    )

    content = []
    maximum_length = 0.0

    for path, name, style, reported_angle in styled:
        if path is None:
            continue

        _, measured_angle = path_metrics(path)
        angle = measured_angle if reported_angle is None else reported_angle

        # Always evaluated in the planner's course units; only the abscissa
        # is rescaled afterwards, for the axis.
        lengths, pressures = pressure_profile(
            path,
            length_units=planner.length_units,
            points_per_segment=points_per_segment,
            **planner.model_parameters,
        )
        lengths = lengths * display_scale
        maximum_length = max(maximum_length, float(lengths[-1]))

        content.append(f"    % {name}")
        content.append(
            f"    \\addplot[{style}{'' if show_legend else ', forget plot'}] "
            "coordinates "
            f"{{{_coordinates(np.column_stack([lengths, pressures]), precision, per_line=4, indent='      ')}}};"
        )

        if show_legend:
            entry = legend_label(
                name,
                fields,
                length=float(lengths[-1]),
                angle=angle * angle_scale,
                pressure=float(pressures[-1]),
                length_units=normalized_length_units,
                angle_units=angle_label,
                pressure_name="final pressure",
            )
            content.append(f"    \\addlegendentry{{{_escape(entry)}}}")

    if burst_pressure is not None:
        content.append("    % burst pressure")
        content.append(
            "    \\addplot[vine/burst, forget plot] "
            f"coordinates {{(0,{_number(burst_pressure, precision)}) "
            f"({_number(maximum_length, precision)},{_number(burst_pressure, precision)})}};"
        )
        content.append(
            "    \\node[anchor=south east, font=\\scriptsize, inner sep=1pt] at "
            f"(axis cs:{_number(maximum_length, precision)},"
            f"{_number(burst_pressure, precision)}) {{burst}};"
        )

    body = _style_block(styles) + _axis(options, content) + "\\end{tikzpicture}\n"
    document = _wrap(body, standalone)

    if filename is not None:
        with open(filename, "w", encoding="utf-8") as handle:
            handle.write(document)

    return document

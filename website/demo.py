"""Interactive demo for the project page.

Runs in the browser through Pyodide. Everything on screen is drawn by
writing SVG markup into the page from the Python below. Lengths are in
metres, with y pointing down the screen.
"""

import math

import numpy as np
from js import document, window
from pyodide.ffi import create_proxy

import planner

# --------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------- #

WIDTH, HEIGHT = 2.0, 0.84     # course size in metres
SCALE = 400.0                 # SVG units per metre (the viewBox is 800 x 336)
BURST_PRESSURE = 25.0         # hypothetical burst pressure, kPa
GRAB_PIXELS = 14              # how close a click must be to grab something
MIN_SIZE = 0.02               # smallest rectangle side, metres
MIN_AREA = 1e-4               # smallest polygon area, square metres
LIVE_BUDGET_MS = 40           # re-solve during a drag only if solving is this fast

PLOT_W, PLOT_H = 800.0, 230.0
PLOT_LEFT, PLOT_RIGHT, PLOT_TOP, PLOT_BOTTOM = 56.0, 16.0, 16.0, 44.0

# slider name -> (paper value, decimals shown)
SLIDERS = {
    "mu": (planner.MU, 3),
    "tension": (planner.TENSION, 2),
    "friction": (planner.FRICTION, 3),
    "yield": (planner.YIELD_PRESSURE, 2),
}

PATH_NAMES = {"pressure": "Minimum pressure", "distance": "Minimum distance"}


def element(name):
    return document.getElementById("demo-" + name)


course_svg = element("course")
plot_svg = element("plot")

state = {
    "obstacles": [],
    "start": None,
    "goal": None,
    "selected": None,        # index of the selected obstacle
    "drag": None,            # what the pointer is dragging right now
    "draw": None,            # corners clicked so far in draw-polygon mode
    "graph": None,           # visibility graph for the current course
    "graph_current": False,  # False after the course changes
    "result": None,          # latest planner output
    "dirty": True,           # the paths need recomputing
    "stale": False,          # the paths on screen are out of date
    "solve_ms": 0.0,
    "frame_pending": False,
}


# --------------------------------------------------------------------- #
# Course
# --------------------------------------------------------------------- #

def rectangle(x0, y0, x1, y1):
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
    return {"kind": "rect", "pts": corners}


def load_default_course():
    """A tight slalom, with a longer but much straighter lane over the top."""
    half = 0.0125
    obstacles = [rectangle(0.20, 0.15, 1.80, 0.175)]

    for k, x in enumerate(np.linspace(0.30, 1.70, 12)):
        if k % 2 == 0:
            # Hangs from the long wall. It overlaps the wall slightly on
            # purpose: two obstacles that only touch leave a seam a path
            # can slip through.
            obstacles.append(rectangle(x - half, 0.16, x + half, 0.53))
        else:
            # Stands on the bottom edge of the course.
            obstacles.append(rectangle(x - half, 0.47, x + half, HEIGHT))

    state["obstacles"] = obstacles
    state["start"] = np.array([0.12, 0.50])
    state["goal"] = np.array([1.88, 0.50])
    state["selected"] = None
    state["drag"] = None
    set_draw(None)
    course_changed()


def course_changed():
    state["graph_current"] = False
    state["dirty"] = True
    request_frame()


def clamp(point, margin=0.0):
    return np.clip(point, [margin, margin], [WIDTH - margin, HEIGHT - margin])


def resized_rectangle(anchor, point):
    low = np.minimum(anchor, point)
    high = np.maximum(anchor, point)
    for axis, limit in ((0, WIDTH), (1, HEIGHT)):
        if high[axis] - low[axis] < MIN_SIZE:
            high[axis] = min(low[axis] + MIN_SIZE, limit)
            low[axis] = high[axis] - MIN_SIZE
    return rectangle(low[0], low[1], high[0], high[1])["pts"]


# --------------------------------------------------------------------- #
# Solving
# --------------------------------------------------------------------- #

def slider(name):
    return float(element(name).value)


def solve():
    began = window.performance.now()

    if not state["graph_current"]:
        state["graph"] = planner.build_graph(
            [obstacle["pts"] for obstacle in state["obstacles"]],
            state["start"],
            state["goal"],
            WIDTH,
            HEIGHT,
        )
        state["graph_current"] = True

    state["result"] = planner.solve(
        state["graph"],
        yield_pressure=slider("yield"),
        tension=slider("tension"),
        friction=slider("friction"),
        mu=slider("mu"),
    )
    state["solve_ms"] = window.performance.now() - began


def same_path(result):
    best, short = result.get("pressure"), result.get("distance")
    return bool(
        best and short
        and best["path"].shape == short["path"].shape
        and np.allclose(best["path"], short["path"])
    )


# --------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------- #

def svg_points(points):
    return " ".join(f"{x * SCALE:.1f},{y * SCALE:.1f}" for x, y in points)


def path_label(path, text, name):
    """A label beside the longest segment of a path."""
    segments = np.diff(path, axis=0)
    k = int(np.argmax(np.linalg.norm(segments, axis=1)))
    if np.linalg.norm(segments[k]) < 0.3:
        return ""
    x, y = 0.5 * (path[k] + path[k + 1]) * SCALE
    y = y - 12 if y > 30 else y + 24
    return f'<text class="demo-label {name}" x="{x:.1f}" y="{y:.1f}">{text}</text>'


def render_course():
    parts = []

    for i, obstacle in enumerate(state["obstacles"]):
        cls = "demo-obstacle selected" if i == state["selected"] else "demo-obstacle"
        points = svg_points(obstacle["pts"])
        parts.append(f'<polygon class="{cls}" points="{points}"/>')

    result = state["result"] or {}
    matching = same_path(result)
    stale = " stale" if state["stale"] else ""

    for name in ("distance", "pressure"):
        found = result.get(name)
        if not found or (matching and name == "distance"):
            continue
        points = svg_points(found["path"])
        parts.append(f'<polyline class="demo-path {name}{stale}" points="{points}"/>')

    if not state["stale"]:
        if matching:
            parts.append(path_label(
                result["pressure"]["path"],
                "Shortest and minimum pressure",
                "pressure",
            ))
        else:
            for name in ("distance", "pressure"):
                if result.get(name):
                    parts.append(path_label(result[name]["path"], PATH_NAMES[name], name))

    if state["selected"] is not None:
        for x, y in state["obstacles"][state["selected"]]["pts"]:
            parts.append(
                f'<rect class="demo-handle" x="{x * SCALE - 5:.1f}" '
                f'y="{y * SCALE - 5:.1f}" width="10" height="10"/>'
            )

    if state["draw"]:
        hull = planner.convex_hull(state["draw"])
        if len(hull) >= 3:
            parts.append(f'<polygon class="demo-preview" points="{svg_points(hull)}"/>')
        for x, y in state["draw"]:
            parts.append(
                f'<circle class="demo-preview-dot" cx="{x * SCALE:.1f}" '
                f'cy="{y * SCALE:.1f}" r="5"/>'
            )

    for name, text in (("start", "Start"), ("goal", "Goal")):
        x, y = state[name] * SCALE
        label_y = y + 28 if y < HEIGHT * SCALE - 40 else y - 16
        parts.append(f'<circle class="demo-{name}" cx="{x:.1f}" cy="{y:.1f}" r="9"/>')
        parts.append(f'<text class="demo-label" x="{x:.1f}" y="{label_y:.1f}">{text}</text>')

    course_svg.innerHTML = "".join(parts)


def format_pressure(value):
    return f"{value:,.0f} kPa" if value >= 1000 else f"{value:.1f} kPa"


def render_readouts():
    result = state["result"] or {}

    for name in ("pressure", "distance"):
        found = result.get(name)
        if found:
            value = format_pressure(found["pressure"])
            detail = f'{found["length"]:.2f} m long · {found["angle"]:.0f}° of turning'
        else:
            value, detail = "No path", ""
        element(name + "-value").textContent = value
        element(name + "-detail").textContent = detail

    best, short = result.get("pressure"), result.get("distance")

    if state["draw"] is not None:
        text = ("Click on the course to add corners, then choose Finish polygon. "
                "Press Esc to cancel.")
    elif not best or not short:
        text = ("No path reaches the goal. Move an obstacle out of the way, "
                "or choose Reset course.")
    elif short["pressure"] <= best["pressure"] * 1.005:
        text = "On this course the shortest path is also a minimum-pressure path."
    else:
        ratio = short["pressure"] / best["pressure"]
        extra = 100.0 * (best["length"] / short["length"] - 1.0)
        text = (f"The minimum-pressure path needs {ratio:.1f}× less pressure "
                f"and is {extra:.0f}% longer.")

    element("message").textContent = text


def nice_step(span, target=4):
    raw = span / target
    magnitude = 10 ** math.floor(math.log10(raw))
    for multiple in (1, 2, 5, 10):
        if raw <= multiple * magnitude:
            return multiple * magnitude
    return 10 * magnitude


def render_plot():
    result = state["result"] or {}
    curves = [
        (name, result[name]["profile"])
        for name in ("distance", "pressure")
        if result.get(name)
    ]

    longest = max([profile[-1][0] for _, profile in curves], default=1.0)
    highest = max([p for _, profile in curves for _, p in profile], default=0.0)
    x_max = longest * 1.03
    y_max = max(highest, BURST_PRESSURE) * 1.12

    inner_w = PLOT_W - PLOT_LEFT - PLOT_RIGHT
    inner_h = PLOT_H - PLOT_TOP - PLOT_BOTTOM
    right = PLOT_W - PLOT_RIGHT
    base = PLOT_TOP + inner_h

    def px(s):
        return PLOT_LEFT + s / x_max * inner_w

    def py(p):
        return base - p / y_max * inner_h

    parts = []

    step = nice_step(y_max)
    for k in range(int(y_max / step) + 1):
        y = py(k * step)
        parts.append(
            f'<line class="demo-plot-grid" x1="{PLOT_LEFT}" y1="{y:.1f}" '
            f'x2="{right}" y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text class="demo-plot-text" x="{PLOT_LEFT - 8}" y="{y + 4:.1f}" '
            f'text-anchor="end">{k * step:g}</text>'
        )

    step = nice_step(x_max)
    for k in range(int(x_max / step) + 1):
        parts.append(
            f'<text class="demo-plot-text" x="{px(k * step):.1f}" y="{base + 18}" '
            f'text-anchor="middle">{k * step:g}</text>'
        )

    parts.append(
        f'<line class="demo-plot-axis" x1="{PLOT_LEFT}" y1="{base}" '
        f'x2="{right}" y2="{base}"/>'
    )

    y = py(BURST_PRESSURE)
    parts.append(
        f'<line class="demo-plot-burst" x1="{PLOT_LEFT}" y1="{y:.1f}" '
        f'x2="{right}" y2="{y:.1f}"/>'
    )
    parts.append(
        f'<text class="demo-plot-text" x="{right}" y="{y - 6:.1f}" '
        f'text-anchor="end">Hypothetical burst pressure</text>'
    )

    for name, profile in curves:
        points = " ".join(f"{px(s):.1f},{py(p):.1f}" for s, p in profile)
        parts.append(f'<polyline class="demo-curve {name}" points="{points}"/>')

    middle = PLOT_TOP + inner_h / 2
    parts.append(
        f'<text class="demo-plot-text" x="{PLOT_LEFT + inner_w / 2:.1f}" '
        f'y="{PLOT_H - 6}" text-anchor="middle">Deployed length (m)</text>'
    )
    parts.append(
        f'<text class="demo-plot-text" x="14" y="{middle:.1f}" text-anchor="middle" '
        f'transform="rotate(-90 14 {middle:.1f})">Pressure (kPa)</text>'
    )

    plot_svg.innerHTML = "".join(parts)


def render():
    render_course()
    render_readouts()
    render_plot()
    element("delete").disabled = state["selected"] is None


def frame(_time=None):
    """Recompute if needed, then redraw. Runs at most once per screen refresh."""
    state["frame_pending"] = False

    if state["dirty"]:
        if state["drag"] is None or state["solve_ms"] <= LIVE_BUDGET_MS:
            solve()
            state["dirty"] = False
            state["stale"] = False
        else:
            # Solving is too slow to keep up with the drag, so the old paths
            # are faded and the new ones are computed on release.
            state["stale"] = True

    render()


frame_proxy = create_proxy(frame)


def request_frame():
    if not state["frame_pending"]:
        state["frame_pending"] = True
        window.requestAnimationFrame(frame_proxy)


# --------------------------------------------------------------------- #
# Pointer and keyboard
# --------------------------------------------------------------------- #

def pointer_position(event):
    """Pointer position in metres, and the grab distance in metres."""
    box = course_svg.getBoundingClientRect()
    point = np.array([
        (event.clientX - box.left) / box.width * WIDTH,
        (event.clientY - box.top) / box.height * HEIGHT,
    ])
    return point, GRAB_PIXELS * WIDTH / box.width


def on_pointer_down(event):
    event.preventDefault()
    point, grab = pointer_position(event)

    if state["draw"] is not None:
        state["draw"].append(clamp(point))
        request_frame()
        return

    course_svg.setPointerCapture(event.pointerId)
    obstacles = state["obstacles"]
    selected = state["selected"]

    # 1. A corner handle of the selected obstacle.
    if selected is not None:
        corners = obstacles[selected]["pts"]
        distance = np.linalg.norm(corners - point, axis=1)
        nearest = int(np.argmin(distance))
        if distance[nearest] <= grab:
            if obstacles[selected]["kind"] == "rect":
                anchor = corners[(nearest + 2) % 4].copy()
                state["drag"] = {"type": "resize", "anchor": anchor}
            else:
                state["drag"] = {"type": "corner", "index": nearest}
            return

    # 2. The start or the goal.
    for name in ("start", "goal"):
        if np.linalg.norm(state[name] - point) <= grab:
            state["selected"] = None
            state["drag"] = {"type": name}
            request_frame()
            return

    # 3. An obstacle, topmost first.
    for i in reversed(range(len(obstacles))):
        if planner.outside_distance(obstacles[i]["pts"], point) <= 0.5 * grab:
            state["selected"] = i
            state["drag"] = {"type": "move", "last": point}
            request_frame()
            return

    state["selected"] = None
    request_frame()


def on_pointer_move(event):
    drag = state["drag"]
    if drag is None:
        return

    point, _ = pointer_position(event)
    kind = drag["type"]

    if kind in ("start", "goal"):
        state[kind] = clamp(point, 0.02)
    else:
        obstacle = state["obstacles"][state["selected"]]
        corners = obstacle["pts"]

        if kind == "move":
            shift = np.clip(
                point - drag["last"],
                -corners.min(axis=0),
                np.array([WIDTH, HEIGHT]) - corners.max(axis=0),
            )
            obstacle["pts"] = corners + shift
            drag["last"] = drag["last"] + shift
        elif kind == "resize":
            obstacle["pts"] = resized_rectangle(drag["anchor"], clamp(point))
        elif kind == "corner":
            moved = corners.copy()
            moved[drag["index"]] = clamp(point)
            # Obstacles have to stay convex, so a move that would dent the
            # shape is ignored.
            if planner.is_convex(moved) and planner.polygon_area(moved) >= MIN_AREA:
                obstacle["pts"] = moved

    course_changed()


def on_pointer_up(_event):
    if state["drag"] is not None:
        state["drag"] = None
        request_frame()


def on_key_down(event):
    target = event.target
    if target.tagName == "TEXTAREA" or (
        target.tagName == "INPUT" and target.type != "range"
    ):
        return

    if event.key in ("Delete", "Backspace") and state["selected"] is not None:
        event.preventDefault()
        delete_selected()
    elif event.key == "Escape" and state["draw"] is not None:
        set_draw(None)
        request_frame()


# --------------------------------------------------------------------- #
# Buttons and sliders
# --------------------------------------------------------------------- #

def set_draw(points):
    state["draw"] = points
    element("draw").textContent = "Draw polygon" if points is None else "Finish polygon"


def finish_polygon():
    corners = state["draw"] or []
    set_draw(None)
    hull = planner.convex_hull(corners)
    if len(hull) >= 3 and planner.polygon_area(hull) >= MIN_AREA:
        state["obstacles"].append({"kind": "poly", "pts": hull})
        state["selected"] = len(state["obstacles"]) - 1
        course_changed()


def delete_selected():
    if state["selected"] is not None:
        del state["obstacles"][state["selected"]]
        state["selected"] = None
        course_changed()


def on_add(_event):
    set_draw(None)
    state["obstacles"].append(rectangle(0.16, 0.72, 0.40, 0.77))
    state["selected"] = len(state["obstacles"]) - 1
    course_changed()


def on_draw(_event):
    if state["draw"] is None:
        state["selected"] = None
        set_draw([])
    else:
        finish_polygon()
    request_frame()


def on_delete(_event):
    delete_selected()


def on_reset(_event):
    load_default_course()


def show_slider_values():
    for name, (_, decimals) in SLIDERS.items():
        element(name + "-out").textContent = f"{slider(name):.{decimals}f}"


def on_slider(_event):
    show_slider_values()
    state["dirty"] = True
    request_frame()


def on_paper_values(_event):
    for name, (value, _) in SLIDERS.items():
        element(name).value = str(value)
    on_slider(None)


# --------------------------------------------------------------------- #
# Start-up
# --------------------------------------------------------------------- #

_proxies = []


def listen(target, event_name, handler):
    proxy = create_proxy(handler)
    _proxies.append(proxy)       # keeps the handler alive
    target.addEventListener(event_name, proxy)


listen(course_svg, "pointerdown", on_pointer_down)
listen(course_svg, "pointermove", on_pointer_move)
listen(course_svg, "pointerup", on_pointer_up)
listen(course_svg, "pointercancel", on_pointer_up)
listen(document, "keydown", on_key_down)

listen(element("add"), "click", on_add)
listen(element("draw"), "click", on_draw)
listen(element("delete"), "click", on_delete)
listen(element("reset"), "click", on_reset)
listen(element("paper"), "click", on_paper_values)

for _name in SLIDERS:
    listen(element(_name), "input", on_slider)

show_slider_values()
load_default_course()

element("loading").hidden = True
element("app").hidden = False
frame()

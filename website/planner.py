"""Growth planner for the website demo.

A small, self-contained version of the planner in the paper:
visibility graph -> line graph -> modified Dijkstra. It needs only NumPy, so
the same file runs in the browser (through Pyodide) and in ordinary Python.

Units are metres and kilopascals. Every obstacle is a convex polygon, given
as an (n, 2) array of corners in order.
"""

import heapq
import math

import numpy as np

# Fitted constants from the paper.
YIELD_PRESSURE = 1.917  # Y, kPa
TENSION = 5.426         # T/A, kPa
FRICTION = 0.771        # lambda/A, kPa per metre
MU = 0.315              # Capstan friction coefficient

TOL = 1e-9


# --------------------------------------------------------------------- #
# Convex polygons
# --------------------------------------------------------------------- #

def polygon_area(polygon):
    points = np.asarray(polygon, dtype=float)
    if len(points) < 3:
        return 0.0
    following = np.roll(points, -1, axis=0)
    return 0.5 * abs(float(np.sum(
        points[:, 0] * following[:, 1] - following[:, 0] * points[:, 1]
    )))


def is_convex(polygon, tol=1e-12):
    points = np.asarray(polygon, dtype=float)
    if len(points) < 3:
        return False
    edges = np.roll(points, -1, axis=0) - points
    following = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    return not (np.any(cross > tol) and np.any(cross < -tol))


def half_spaces(polygon):
    """Outward unit normals and offsets: inside means normals @ x + offsets <= 0."""
    points = np.asarray(polygon, dtype=float)
    following = np.roll(points, -1, axis=0)
    edges = following - points
    signed_area = np.sum(
        points[:, 0] * following[:, 1] - following[:, 0] * points[:, 1]
    )
    orientation = 1.0 if signed_area > 0.0 else -1.0
    normals = orientation * np.column_stack((edges[:, 1], -edges[:, 0]))
    lengths = np.linalg.norm(normals, axis=1)
    keep = lengths > 0.0
    normals = normals[keep] / lengths[keep, None]
    offsets = -np.einsum("ij,ij->i", normals, points[keep])
    return normals, offsets


def outside_distance(polygon, point):
    """Roughly how far a point is outside a polygon; negative when inside."""
    normals, offsets = half_spaces(polygon)
    return float((normals @ np.asarray(point, dtype=float) + offsets).max())


def convex_hull(points):
    """Convex hull of some points, as an (n, 2) array of corners in order."""
    unique = sorted(set((float(x), float(y)) for x, y in points))
    if len(unique) < 3:
        return np.asarray(unique, dtype=float).reshape(-1, 2)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


# --------------------------------------------------------------------- #
# Visibility
# --------------------------------------------------------------------- #

def strictly_inside(points, spaces, tol=TOL):
    """(n,) bool: is each point in the interior of some obstacle?"""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    inside = np.zeros(len(points), dtype=bool)
    for normals, offsets in spaces:
        inside |= (points @ normals.T + offsets).max(axis=1) < -tol
    return inside


def blocked(starts, ends, spaces, tol=TOL):
    """(m,) bool: does each segment pass through the interior of an obstacle?

    Same test as the package: clip the segment against the obstacle's faces
    (Cyrus-Beck), then probe the middle of the clipped piece. A segment that
    only runs along an edge or touches a corner is not blocked.
    """
    starts = np.atleast_2d(np.asarray(starts, dtype=float))
    ends = np.atleast_2d(np.asarray(ends, dtype=float))
    directions = ends - starts
    result = np.zeros(len(starts), dtype=bool)

    for normals, offsets in spaces:
        numerator = starts @ normals.T + offsets      # face distance at t = 0
        denominator = directions @ normals.T          # its rate of change
        parallel = np.abs(denominator) <= tol
        separated = (parallel & (numerator > tol)).any(axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            crossing = -numerator / denominator

        enter = np.where(~parallel & (denominator < 0), crossing, 0.0).max(axis=1)
        leave = np.where(~parallel & (denominator > 0), crossing, 1.0).min(axis=1)

        middle = 0.5 * (enter + leave)
        probe = starts + middle[:, None] * directions
        worst = (probe @ normals.T + offsets).max(axis=1)

        result |= ~separated & (enter <= leave) & (worst < -tol)

    return result


def _key(point):
    return (round(float(point[0]), 9), round(float(point[1]), 9))


def build_graph(obstacles, start, goal, width, height):
    """Visibility graph over the obstacle corners plus the start and goal.

    Returns None when the start or goal is inside an obstacle, or when the
    two are the same point.
    """
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    obstacles = [np.asarray(obstacle, dtype=float) for obstacle in obstacles]
    spaces = [half_spaces(obstacle) for obstacle in obstacles]

    if strictly_inside(np.vstack([start, goal]), spaces).any():
        return None

    # A corner is somewhere a path may bend unless it is buried inside
    # another obstacle or sits on the edge of the course. Dropping the ones
    # on the edge is what makes an obstacle pushed against a wall seal it.
    points = []
    if obstacles:
        corners = np.vstack(obstacles)
        margin = 1e-6
        usable = (
            (corners[:, 0] > margin) & (corners[:, 0] < width - margin)
            & (corners[:, 1] > margin) & (corners[:, 1] < height - margin)
            & ~strictly_inside(corners, spaces)
        )
        points = list(corners[usable])
    points += [start, goal]

    index = {}
    for point in points:
        index.setdefault(_key(point), len(index))

    coords = np.array(list(index.keys()), dtype=float)
    start_index = index[_key(start)]
    goal_index = index[_key(goal)]

    if start_index == goal_index:
        return None

    count = len(coords)
    first, second = np.triu_indices(count, 1)
    free = ~blocked(coords[first], coords[second], spaces)

    visible = np.zeros((count, count), dtype=bool)
    visible[first[free], second[free]] = True
    visible |= visible.T

    difference = coords[None, :, :] - coords[:, None, :]    # [u, v] = v - u
    length = np.linalg.norm(difference, axis=2)
    unit = difference / np.where(length > 0.0, length, 1.0)[:, :, None]

    return {
        "coords": coords,
        "visible": visible,
        "length": length,
        "unit": unit,
        "start": start_index,
        "goal": goal_index,
    }


# --------------------------------------------------------------------- #
# Searches
# --------------------------------------------------------------------- #

def shortest_path(graph):
    """Minimum-distance path as an (n, 2) array, or None."""
    visible, length = graph["visible"], graph["length"]
    start, goal = graph["start"], graph["goal"]
    count = len(graph["coords"])

    distance = np.full(count, np.inf)
    parent = np.full(count, -1, dtype=int)
    done = np.zeros(count, dtype=bool)
    distance[start] = 0.0
    queue = [(0.0, start)]

    while queue:
        cost, node = heapq.heappop(queue)
        if done[node]:
            continue
        done[node] = True
        if node == goal:
            break

        candidate = cost + length[node]
        better = visible[node] & ~done & (candidate < distance)

        for neighbour in np.nonzero(better)[0]:
            neighbour = int(neighbour)
            distance[neighbour] = candidate[neighbour]
            parent[neighbour] = node
            heapq.heappush(queue, (float(candidate[neighbour]), neighbour))

    if not np.isfinite(distance[goal]):
        return None

    path = [goal]
    while path[-1] != start:
        path.append(int(parent[path[-1]]))
    path.reverse()

    return graph["coords"][path]


def minimum_pressure_path(graph, tension, friction, mu):
    """Minimum-pressure path as an (n, 2) array, or None.

    Dijkstra over the line graph. A state is a directed edge (u, v), and its
    label is the tension built up through that segment, before the bend at v:

        label(v, w) = label(u, v) * exp(mu * theta) + friction * length(v, w)
    """
    visible, length, unit = graph["visible"], graph["length"], graph["unit"]
    start, goal = graph["start"], graph["goal"]
    count = len(graph["coords"])
    nodes = np.arange(count)

    label = np.full((count, count), np.inf)
    parent = np.full((count, count), -1, dtype=int)
    settled = np.zeros((count, count), dtype=bool)

    seeds = np.nonzero(visible[start])[0]
    if len(seeds) == 0:
        return None

    label[start, seeds] = tension + friction * length[start, seeds]
    queue = [(float(label[start, v]), start, int(v)) for v in seeds]
    heapq.heapify(queue)

    terminal = None

    while queue:
        cost, tail, head = heapq.heappop(queue)
        if settled[tail, head]:
            continue
        settled[tail, head] = True

        if head == goal:
            terminal = (tail, head)
            break

        # Turning angle from the incoming direction to every outgoing one.
        cosines = unit[head] @ unit[tail, head]
        angles = np.arccos(np.clip(cosines, -1.0, 1.0))
        candidate = cost * np.exp(mu * angles) + friction * length[head]

        better = (
            visible[head]
            & (nodes != tail)              # no U-turns
            & ~settled[head]
            & (candidate < label[head])
        )

        for neighbour in np.nonzero(better)[0]:
            neighbour = int(neighbour)
            label[head, neighbour] = candidate[neighbour]
            parent[head, neighbour] = tail
            heapq.heappush(queue, (float(candidate[neighbour]), head, neighbour))

    if terminal is None:
        return None

    tail, head = terminal
    path = [head, tail]
    while parent[tail, head] >= 0:
        previous = int(parent[tail, head])
        path.append(previous)
        tail, head = previous, tail
    path.reverse()

    return graph["coords"][path]


# --------------------------------------------------------------------- #
# Pressure model (same recursion as vine_planner.utils)
# --------------------------------------------------------------------- #

def segment_geometry(path):
    """Segment lengths and the unsigned turning angle at each interior corner."""
    vectors = np.diff(np.asarray(path, dtype=float), axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    if len(vectors) < 2:
        return lengths, np.zeros(0)
    cosines = np.einsum("ij,ij->i", vectors[:-1], vectors[1:]) / (
        lengths[:-1] * lengths[1:]
    )
    return lengths, np.arccos(np.clip(cosines, -1.0, 1.0))


def path_pressure(path, yield_pressure, tension, friction, mu):
    lengths, angles = segment_geometry(path)
    tau = float(tension)
    for segment, angle in zip(lengths[:-1], angles):
        tau = (tau + friction * segment) * math.exp(mu * angle)
    return float(yield_pressure + tau + friction * lengths[-1])


def pressure_profile(path, yield_pressure, tension, friction, mu):
    """(deployed length, pressure) points; two points at each bend show the jump."""
    lengths, angles = segment_geometry(path)
    deployed = 0.0
    tau = float(tension)
    points = [(0.0, yield_pressure + tau)]

    for k, segment in enumerate(lengths):
        if k > 0:
            tau *= math.exp(mu * angles[k - 1])
            points.append((deployed, yield_pressure + tau))
        tau += friction * segment
        deployed += float(segment)
        points.append((deployed, yield_pressure + tau))

    return points


def describe(path, yield_pressure, tension, friction, mu):
    lengths, angles = segment_geometry(path)
    model = (yield_pressure, tension, friction, mu)
    return {
        "path": path,
        "length": float(lengths.sum()),
        "angle": float(np.degrees(angles.sum())),
        "pressure": path_pressure(path, *model),
        "profile": pressure_profile(path, *model),
    }


def solve(graph, yield_pressure=YIELD_PRESSURE, tension=TENSION,
          friction=FRICTION, mu=MU):
    """Both paths, each with its length, turning, pressure and profile."""
    if graph is None:
        return {"distance": None, "pressure": None}

    # The shortest path does not depend on the constants, so it is kept.
    if "distance_path" not in graph:
        graph["distance_path"] = shortest_path(graph)

    paths = {
        "distance": graph["distance_path"],
        "pressure": minimum_pressure_path(graph, tension, friction, mu),
    }
    model = (yield_pressure, tension, friction, mu)

    return {
        name: None if path is None else describe(path, *model)
        for name, path in paths.items()
    }

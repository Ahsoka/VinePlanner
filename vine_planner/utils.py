import numpy as np
import itertools
import shapely

from functools import lru_cache

from shapely import Polygon, LineString, Point
from .constants import (
    YIELD_PRESSURE_PSI,
    LENGTH_FRICTION_PSI_PER_FT,
    TAIL_TENSION_PSI,
    CURVATURE_FRICTION_COEFFICIENT,
    _LENGTH_TO_FEET
)

# ------------------------------------------------------------------
# Pressure model
# ------------------------------------------------------------------
#
# Recursive (nested capstan) growth pressure. For a path of n segments
# with lengths l_1 ... l_n and n - 1 interior turning angles t_1 ... t_{n-1}:
#
#     tau_0 = T
#     tau_i = (tau_{i-1} + f * l_i) * exp(mu_c * |t_i|)      i = 1 ... n-1
#     P     = Y + tau_{n-1} + f * l_n
#
# Each segment's drag is amplified only by the bends *distal* to it, i.e.
# the bends its accumulated tension must still be dragged through on the
# way to the tip. The terminal segment has nothing distal to it and is
# therefore charged at face value.
#
# This replaces the summed form P = Y + (f * L + T) * exp(mu_c * sum|t_i|),
# which charges every segment for every bend and is a strict upper bound.
# ------------------------------------------------------------------

_DEGREE_ALIASES = {"degree", "degrees", "deg"}
_RADIAN_ALIASES = {"radian", "radians", "rad"}


def _to_radians(angles, angle_units: str) -> np.ndarray:
    angle_array = np.atleast_1d(
        np.asarray(angles, dtype=float)
    ).ravel()

    normalized = angle_units.lower()

    if normalized in _DEGREE_ALIASES:
        return np.deg2rad(angle_array)

    if normalized in _RADIAN_ALIASES:
        return angle_array

    raise ValueError("angle_units must be 'degrees' or 'radians'.")


def _length_scale(length_units: str) -> float:
    normalized = length_units.lower()

    if normalized not in _LENGTH_TO_FEET:
        raise ValueError(
            "length_units must be one of: 'ft', 'in', 'm', or 'cm'."
        )

    return _LENGTH_TO_FEET[normalized]


def segment_geometry(path) -> tuple[np.ndarray, np.ndarray]:
    """Return per-segment lengths and per-vertex turning angles.

    Accepts a ``LineString`` or any ``(n + 1, d)`` array of vertices, in
    two or three dimensions. Consecutive duplicate vertices are dropped.
    Turning angles are unsigned and reported in radians; there is one
    per interior vertex, so ``len(angles) == len(lengths) - 1``.
    """
    coords = np.asarray(
        path.coords if hasattr(path, "coords") else path,
        dtype=float,
    )

    if coords.ndim != 2 or len(coords) < 2:
        return np.zeros(0), np.zeros(0)

    keep = np.concatenate((
        [True],
        np.any(np.diff(coords, axis=0) != 0.0, axis=1),
    ))
    coords = coords[keep]

    if len(coords) < 2:
        return np.zeros(0), np.zeros(0)

    vectors = np.diff(coords, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)

    if len(vectors) < 2:
        return lengths, np.zeros(0)

    incoming = vectors[:-1]
    outgoing = vectors[1:]

    cosine_angles = np.einsum(
        "ij,ij->i",
        incoming,
        outgoing,
    ) / (
        lengths[:-1] * lengths[1:]
    )

    angles = np.arccos(
        np.clip(cosine_angles, -1.0, 1.0)
    )

    return lengths, angles


def vine_robot_pressure(
    segment_lengths,
    angles,
    *,
    length_units: str = "ft",
    angle_units: str = "degrees",
    yield_pressure: float = YIELD_PRESSURE_PSI,
    length_friction: float = LENGTH_FRICTION_PSI_PER_FT,
    tail_tension: float = TAIL_TENSION_PSI,
    curvature_coefficient: float = CURVATURE_FRICTION_COEFFICIENT,
) -> float:
    """
    Compute the vine-robot driving pressure in psi (recursive capstan model).

    Parameters
    ----------
    segment_lengths:
        Per-segment lengths in ``length_units``, ordered base to tip. A
        bare float is accepted for a single straight segment.
    angles:
        Turning angles at the interior vertices, one per vertex, so
        ``len(angles) == len(segment_lengths) - 1``. Signs are ignored.
    length_units:
        Units used for ``segment_lengths``: ``"ft"``, ``"in"``, ``"m"``,
        or ``"cm"``.
    angle_units:
        Units used for ``angles``: ``"degrees"`` or ``"radians"``.
    yield_pressure, length_friction, tail_tension, curvature_coefficient:
        Model parameters, defaulting to the fitted values in
        ``constants``. ``length_friction`` is psi per foot regardless of
        ``length_units``.

    Returns
    -------
    float
        Required pressure in psi.

    Notes
    -----
    This function needs the *interleaving* of lengths and bends, not just
    the totals, so it deliberately rejects the old ``(total_length,
    total_angle)`` call signature rather than silently reinterpreting it.
    """
    lengths = np.atleast_1d(
        np.asarray(segment_lengths, dtype=float)
    ).ravel()

    turning_angles = np.abs(
        _to_radians(angles, angle_units)
    )

    if lengths.size == 0:
        raise ValueError(
            "segment_lengths must contain at least one segment."
        )

    if np.any(lengths < 0.0):
        raise ValueError("Segment lengths must be nonnegative.")

    if turning_angles.size != lengths.size - 1:
        raise ValueError(
            "Expected one turning angle per interior vertex: "
            f"len(angles) == len(segment_lengths) - 1 == "
            f"{lengths.size - 1}, got {turning_angles.size}. The "
            "recursive pressure model needs per-segment lengths, not a "
            "single total length."
        )

    if curvature_coefficient < 0.0:
        raise ValueError("curvature_coefficient must be nonnegative.")

    lengths_ft = lengths * _length_scale(length_units)
    drag = length_friction * lengths_ft
    amplification = np.exp(curvature_coefficient * turning_angles)

    # tau_i = (tau_{i-1} + f * l_i) * exp(mu_c * |t_i|), then the
    # terminal segment is added without amplification.
    tension = float(tail_tension)

    for segment_drag, factor in zip(drag[:-1], amplification):
        tension = (tension + segment_drag) * factor

    tension += drag[-1]

    return float(yield_pressure + tension)


def path_pressure(
    path,
    *,
    length_units: str = "ft",
    **model_parameters,
) -> float:
    """Growth pressure of a ``LineString`` or vertex array, in psi."""
    lengths, angles = segment_geometry(path)

    if lengths.size == 0:
        raise ValueError(
            "A pressure requires a path with at least two distinct points."
        )

    return vine_robot_pressure(
        lengths,
        angles,
        length_units=length_units,
        angle_units="radians",
        **model_parameters,
    )


# ------------------------------------------------------------------
# Legend text
# ------------------------------------------------------------------
#
# Shared by the Matplotlib plots and the TikZ export so the two cannot
# report a path differently. Every quantity is optional: a paper figure
# usually wants the pressure and nothing else, while a working figure
# wants all three.

_DEFAULT_LEGEND_FIELDS = ("length", "angle", "pressure")

_LEGEND_FIELD_ALIASES = {
    "length": "length",
    "distance": "length",
    "len": "length",
    "angle": "angle",
    "turning": "angle",
    "turn": "angle",
    "pressure": "pressure",
}


def normalize_legend_fields(fields=_DEFAULT_LEGEND_FIELDS) -> tuple[str, ...]:
    """Resolve a legend-field selection to canonical names, order preserved.

    Accepts ``True`` / ``None`` for all three, ``False`` / ``()`` for none, a
    single name, or any sequence of names. ``"distance"`` is a synonym for
    ``"length"``: the path is named for the objective it minimises, the
    quantity reported is the length of it.
    """
    if fields is None or fields is True:
        return _DEFAULT_LEGEND_FIELDS

    if fields is False:
        return ()

    if isinstance(fields, str):
        fields = (fields,)

    resolved = []

    for name in fields:
        canonical = _LEGEND_FIELD_ALIASES.get(str(name).lower())

        if canonical is None:
            raise ValueError(
                "legend fields must be drawn from 'length' (or 'distance'), "
                f"'angle' and 'pressure'; got {name!r}."
            )

        if canonical not in resolved:
            resolved.append(canonical)

    return tuple(resolved)


def legend_label(
    name: str,
    fields=_DEFAULT_LEGEND_FIELDS,
    *,
    length: float = None,
    angle: float = None,
    pressure: float = None,
    length_units: str = "ft",
    angle_units: str = "deg",
    pressure_name: str = "pressure",
) -> str:
    """``"min pressure (length=48.37 ft, pressure=3.857 psi)"``.

    A field is omitted when it was not selected *or* when its value was not
    supplied, so a caller that has no angle to report does not have to edit
    its field list to match.
    """
    values = {"length": length, "angle": angle, "pressure": pressure}
    units = {"length": length_units, "angle": angle_units, "pressure": "psi"}
    formats = {"length": ".2f", "angle": ".2f", "pressure": ".3f"}
    labels = {"length": "length", "angle": "angle", "pressure": pressure_name}

    parts = [
        f"{labels[field]}={values[field]:{formats[field]}} {units[field]}"
        for field in normalize_legend_fields(fields)
        if values[field] is not None
    ]

    return f"{name} ({', '.join(parts)})" if parts else name


# ------------------------------------------------------------------
# How many obstacles fit
# ------------------------------------------------------------------


class ObstacleCapacityWarning(UserWarning):
    """A course was asked for more obstacles than it can hold.

    Its own category so it can be silenced or promoted to an error with
    ``warnings.filterwarnings`` without touching anything else.
    """


@lru_cache(maxsize=None)
def _shape_coefficient(
    min_vertices: int,
    max_vertices: int,
    samples: int = 4096,
    seed: int = 0,
) -> float:
    """Mean obstacle area divided by mean radius squared.

    The shoelace area of a polygon that is star shaped about its center is

        A = 0.5 * sum_i r_i r_{i+1} sin(theta_{i+1} - theta_i),

    and ``annulus_obstacle`` draws the radii independently of the angles and
    of each other. Consecutive radii are therefore independent, so
    ``E[r_i r_{i+1}] = E[r]^2`` and the whole expectation factors as

        E[A] = E[r]^2 * (0.5 * E[sum_i sin(gap_i)]).

    The bracketed term is dimensionless and depends only on the vertex-count
    range, which is why this is cached: the Monte Carlo runs once per range
    per process rather than once per course.

    It is estimated rather than derived because ``sample_angles`` rejects any
    draw whose largest gap reaches pi, and that conditioning has no tidy
    closed form -- it also bites hardest at three vertices, where it throws
    away three draws in four. Calling ``sample_angles`` here keeps the
    estimate correct if the angle law is ever changed; a hand-derived formula
    would silently stop matching.
    """
    rng = np.random.default_rng(seed)
    total = 0.0

    for _ in range(samples):
        count = int(rng.integers(min_vertices, max_vertices + 1))
        angles = sample_angles(count, rng)
        radii = rng.uniform(0.0, 2.0, count)

        total += 0.5 * float(np.sum(
            radii * np.roll(radii, -1) * np.sin(np.roll(angles, -1) - angles)
        ))

    # The radii above have mean 1, so dividing by E[r]^2 = 1 is a no-op and
    # the average is already the coefficient.
    return abs(total / samples)


def expected_obstacle_area(
    min_vertices: int,
    max_vertices: int,
    min_radius: float,
    max_radius: float,
) -> float:
    """Mean area of one obstacle as ``annulus_obstacle`` would draw it.

    This is the area of an *unconditional* draw. Obstacles that survive the
    overlap test are smaller on average, because a large draw is likelier to
    be rejected: on a saturated 25 ft course the accepted obstacles averaged
    1.26 square feet against 1.63 for the draws. ``obstacle_capacity``
    accounts for that in its coefficient, so do not correct for it twice.
    """
    mean_radius = 0.5 * (float(min_radius) + float(max_radius))

    return _shape_coefficient(int(min_vertices), int(max_vertices)) * mean_radius ** 2


def obstacle_capacity(
    width: float,
    height: float,
    min_vertices: int,
    max_vertices: int,
    min_radius: float,
    max_radius: float,
    packing_coefficient: float = 0.65,
) -> int:
    """Roughly the largest number of obstacles that will fit.

    ``packing_coefficient`` is *not* an area fraction. It is an empirical
    constant absorbing two effects at once: how densely sequential random
    placement can pack before it runs out of room, and the downward size bias
    of the obstacles that survive rejection. Measured on saturated 25 ft
    courses it sat between 0.53 and 0.61 over radii from 0.45 to 2.0, a 4.4x
    range; the default is set above that so the number is a ceiling rather
    than a target. The intent is to catch a request that cannot be met at
    all, not to second-guess one that merely works the sampler hard.

    The estimate is optimistic when obstacles are large relative to the
    course, since centers are confined to an inset box while the area
    accounting is not. Treat it as an order of magnitude, and rely on the
    post-generation shortfall warning for the rest.
    """
    mean_area = expected_obstacle_area(
        min_vertices, max_vertices, min_radius, max_radius
    )

    if mean_area <= 0.0:
        return 0

    return int(packing_coefficient * float(width) * float(height) / mean_area)


def suggested_radius(
    width: float,
    height: float,
    num_obstacles: int,
    min_vertices: int,
    max_vertices: int,
    annulus_offset: float,
    packing_coefficient: float = 0.65,
    margin: float = 1.25,
) -> float:
    """The ``radius`` at which ``num_obstacles`` would comfortably fit.

    Inverts ``obstacle_capacity``. Mean area goes as the square of the mean
    radius, so the target mean radius follows directly; ``annulus_offset`` is
    ``max_radius - min_radius``, subtracted back off to return the inner
    radius the caller actually passes.

    ``margin`` sizes for more obstacles than were asked for, and is not
    optional padding. ``obstacle_capacity`` reports a ceiling, and sequential
    placement stalls well before a ceiling: sizing a 301-obstacle request
    exactly at capacity produced 277. Sizing it for 376 instead lands at a
    radius where the request is met outright.
    """
    coefficient = _shape_coefficient(int(min_vertices), int(max_vertices))

    target_count = max(float(margin) * max(num_obstacles, 1), 1.0)
    target_area = packing_coefficient * float(width) * float(height) / target_count
    target_mean_radius = np.sqrt(target_area / coefficient)

    return max(float(target_mean_radius) - 0.5 * float(annulus_offset), 0.0)


def path_metrics(path: LineString) -> tuple[float, float]:
    """Return path length and cumulative unsigned turning angle.

    Retained for reporting. The cumulative angle is no longer sufficient
    to determine pressure under the recursive model, which depends on
    where each bend sits along the path.
    """
    lengths, angles = segment_geometry(path)

    return float(lengths.sum()), float(angles.sum())


def pressure_profile(
    path: LineString,
    *,
    length_units: str = "ft",
    points_per_segment: int = 50,
    yield_pressure: float = YIELD_PRESSURE_PSI,
    length_friction: float = LENGTH_FRICTION_PSI_PER_FT,
    tail_tension: float = TAIL_TENSION_PSI,
    curvature_coefficient: float = CURVATURE_FRICTION_COEFFICIENT,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cumulative path length and required pressure samples.

    The profile is the pressure history of the deployment: the value at
    arclength ``s`` is the pressure required when the tip has reached
    ``s``, which is the recursion applied to the path truncated there.

    Under the recursive model the tip is always distal to every bend, so
    freshly deployed body adds drag at the constant rate ``f`` and the
    profile rises linearly within each segment. Rounding a vertex
    multiplies the accumulated tension by ``exp(mu_c * |t|)``, so the
    profile contains two pressure values at that same cumulative length
    to display the resulting jump. (Under the old summed model the slope
    itself grew after every turn; here it does not.)
    """
    length_to_feet = _length_scale(length_units)

    if not isinstance(points_per_segment, int) or points_per_segment < 1:
        raise ValueError("points_per_segment must be a positive integer.")

    coords = np.asarray(path.coords, dtype=float)

    if len(coords) > 1:
        keep = np.concatenate((
            [True],
            np.any(np.diff(coords, axis=0) != 0.0, axis=1),
        ))
        coords = coords[keep]

    if len(coords) < 2:
        raise ValueError(
            "A pressure profile requires a path with at least two "
            "distinct points."
        )

    segment_lengths, turning_angles = segment_geometry(coords)

    if np.any(segment_lengths == 0.0):
        raise ValueError(
            "A pressure profile cannot contain zero-length segments."
        )

    cumulative_length = 0.0
    tension = float(tail_tension)

    profile_lengths = []
    profile_pressures = []

    for segment_index, segment_length in enumerate(segment_lengths):
        if segment_index > 0:
            # Everything deployed so far is now proximal to this bend, so
            # the whole accumulated tension is amplified at once. The
            # previous segment already contributed the pre-turn pressure
            # at this cumulative length.
            tension *= np.exp(
                curvature_coefficient
                * turning_angles[segment_index - 1]
            )
            profile_lengths.append(cumulative_length)
            profile_pressures.append(yield_pressure + tension)

        local_lengths = np.linspace(
            0.0,
            segment_length,
            points_per_segment + 1,
        )

        # The segment start is already present after an interior turn.
        if segment_index > 0:
            local_lengths = local_lengths[1:]

        sampled_lengths = cumulative_length + local_lengths
        sampled_pressures = (
            yield_pressure
            + tension
            + length_friction * local_lengths * length_to_feet
        )

        profile_lengths.extend(sampled_lengths.tolist())
        profile_pressures.extend(sampled_pressures.tolist())

        tension += length_friction * segment_length * length_to_feet
        cumulative_length += segment_length

    return (
        np.asarray(profile_lengths, dtype=float),
        np.asarray(profile_pressures, dtype=float),
    )

# ------------------------------------------------------------------
# Sampling helpers
# ------------------------------------------------------------------

def sample_uniform_points(
    region,
    number_of_points: int,
    rng
) -> np.ndarray:
    triangulation = shapely.constrained_delaunay_triangles(
        region
    )

    triangles = [
        triangle
        for triangle in shapely.get_parts(triangulation)
        if (
            triangle.geom_type == "Polygon"
            and triangle.area > 0
        )
    ]

    if not triangles:
        raise RuntimeError(
            "The sampling region could not be triangulated."
        )

    areas = np.array(
        [triangle.area for triangle in triangles],
        dtype=float,
    )

    # Selecting triangles in proportion to their areas makes the
    # final points uniform over the complete sampling region.
    selected_indices = rng.choice(
        len(triangles),
        size=number_of_points,
        replace=True,
        p=areas / areas.sum(),
    )

    selected_triangles = np.stack(
        [
            np.asarray(
                triangles[index].exterior.coords,
                dtype=float,
            )[:3, :2]
            for index in selected_indices
        ]
    )

    point_a = selected_triangles[:, 0]
    point_b = selected_triangles[:, 1]
    point_c = selected_triangles[:, 2]

    # Uniform barycentric coordinates within each triangle.
    random_1 = np.sqrt(rng.random(number_of_points))
    random_2 = rng.random(number_of_points)

    return (
        (1.0 - random_1)[:, None] * point_a
        + (
            random_1 * (1.0 - random_2)
        )[:, None] * point_b
        + (
            random_1 * random_2
        )[:, None] * point_c
    )

# ------------------------------------------------------------------
# Standard obstacles
# ------------------------------------------------------------------

def sample_angles(count: int, rng: np.random.Generator) -> np.ndarray:
    """Sorted vertex angles whose largest gap stays below pi.

    Independent uniform angles are the standard construction, but
    sorting them is not on its own enough to keep the polygon
    simple: once the largest gap between consecutive angles
    reaches pi, the chord that closes it passes on the far side of
    the center and can cut across the other edges. Holding every
    gap below pi confines each edge to its own angular wedge, and
    wedges cannot cross, so the polygon is simple by construction.
    """
    for _ in range(100):
        angles = np.sort(
            rng.uniform(0.0, 2.0 * np.pi, count)
        )

        gaps = np.diff(
            np.concatenate((angles, angles[:1] + 2.0 * np.pi))
        )

        if gaps.max() < np.pi:
            return angles

    # Unreachable in practice. Evenly spaced angles at a random
    # phase leave every gap at 2 pi / count, which is below pi for
    # any polygon with three or more vertices.
    return (
        2.0 * np.pi * np.arange(count) / count
        + rng.uniform(0.0, 2.0 * np.pi / count)
    )

def annulus_obstacle(
    center: np.ndarray,
    rng: np.random.Generator,
    min_vertices: int,
    max_vertices: int,
    min_radius: float,
    max_radius: float,
    polygons: list[Polygon],
    course_region: Polygon,
    *,
    forbidden_points=(),
    spanning_axis=None,
    containment_region: Polygon = None,
    max_attempts: int = 5000,
    raise_on_failure: bool = True,

):
    """Sample one annulus obstacle entirely inside current free space.

    Angles and radii retain the original annulus construction. A draw
    is accepted only when every sampled vertex lies in the Shapely
    free-space set and the complete candidate polygon is covered by
    that same set. The candidate must also contain its nominal center,
    preserving the meaning of the supplied line-placement point as the
    obstacle center.

    ``forbidden_points`` are future line centers. Rejecting candidates
    that cover any of them prevents an earlier obstacle from consuming
    the center needed by a later obstacle.

    ``containment_region`` replaces ``course_region`` when an obstacle is
    allowed to reach past the course walls, which only the two corner
    obstacles are. It relaxes the wall constraint alone: the region still
    has the placed polygons subtracted from it, so overlap with an
    existing obstacle stays forbidden.

    ``spanning_axis`` forces two of the vertices onto the line through
    the center in that direction, one on each side, each at its own
    radius drawn from the annulus, and places one further vertex on each
    side of that line so the ray along the axis runs through the
    obstacle's interior rather than along an edge. Four vertices are
    therefore the minimum when an axis is given. The obstacle then reaches its full
    width across that direction instead of whatever width the angles
    happened to give it, which is how a normal-sized obstacle is made to
    block a specific line. Two opposite vertices also cap every angular
    gap at pi, so the sorted polygon cannot self-intersect.
    """
    center = np.asarray(center, dtype=float)
    region = course_region if containment_region is None else containment_region
    free_space = shapely.difference(region, shapely.union_all(polygons))
    center_point = Point(center)
    forbidden_geometry = (
        shapely.points(np.asarray(forbidden_points, dtype=float))
        if len(forbidden_points)
        else None
    )

    if not shapely.covers(free_space, center_point):
        if raise_on_failure:
            raise RuntimeError(
                "The requested obstacle center is not in the current "
                "Shapely free-space region."
            )

        return None

    # A spanning axis needs one free vertex on each side of it, so three
    # vertices is not enough: two of them are the forced pair and the third
    # can only sit on one side.
    lowest_vertices = max(min_vertices, 4 if spanning_axis is not None else 3)
    highest_vertices = max(max_vertices, lowest_vertices)

    for _ in range(max_attempts):
        number_of_vertices = int(
            rng.integers(lowest_vertices, highest_vertices + 1)
        )

        if spanning_axis is None:
            angles = sample_angles(number_of_vertices, rng)
        else:
            # The two forced vertices sit half a turn apart on the
            # spanning axis; the rest are ordinary uniform draws. Sorting
            # the combined set keeps the polygon star shaped about the
            # center, and no gap can exceed pi because the forced pair
            # already splits the circle in half.
            axis_angle = np.arctan2(
                float(spanning_axis[1]),
                float(spanning_axis[0]),
            )

            # One free vertex is placed on each side of the axis before the
            # rest are drawn over the whole circle. Left to chance they can
            # all land on one side, which leaves a gap of exactly pi: the
            # edge joining the forced pair then runs straight through the
            # center, so the center and every point along the axis ray lie
            # on the boundary rather than inside it. Straddling makes every
            # gap strictly less than pi, which puts the whole ray out to the
            # forced vertex in the interior.
            angles = np.sort(np.mod(
                np.concatenate((
                    [axis_angle, axis_angle + np.pi],
                    axis_angle + rng.uniform(0.0, np.pi, 2)
                    + [0.0, np.pi],
                    rng.uniform(
                        0.0,
                        2.0 * np.pi,
                        number_of_vertices - 4,
                    ),
                )),
                2.0 * np.pi,
            ))

        radii = rng.uniform(
            min_radius,
            max_radius,
            number_of_vertices,
        )
        vertices = (
            center
            + radii[:, None] * np.column_stack((
                np.cos(angles),
                np.sin(angles),
            ))
        )

        # Enforce the vertex-level requirement directly using the
        # Shapely set difference defining the current free space.
        vertex_points = shapely.points(vertices)
        if not np.all(shapely.covers(free_space, vertex_points)):
            continue

        candidate = Polygon(vertices)

        if (
            candidate.is_empty
            or not candidate.is_valid
            or candidate.area <= 0.0
            or not shapely.contains(candidate, center_point)
        ):
            continue

        # Vertices being free is not sufficient by itself: an edge can
        # cross an occupied polygon while both endpoints remain free.
        # Requiring the whole candidate to be covered by free space
        # guarantees zero positive-area overlap. Boundary contact is
        # allowed, but obstacle interiors remain disjoint.
        if not shapely.covers(free_space, candidate):
            continue

        if forbidden_geometry is not None and np.any(
            shapely.intersects(candidate, forbidden_geometry)
        ):
            continue

        return candidate

    if raise_on_failure:
        raise RuntimeError(
            "Could not sample a non-overlapping obstacle around the "
            "requested line center. Reduce num_obstacles or radius, "
            "or increase the course dimensions."
        )

    return None

# ------------------------------------------------------------------
# Clusters in the two corners the barriers do not reach
# ------------------------------------------------------------------

def clip_segment(
    start: np.ndarray,
    end: np.ndarray,
    inset_low_x: float,
    inset_low_y: float,
    inset_high_x: float,
    inset_high_y: float
):
    """Parameter range of ``start -> end`` inside the inset box.

    Liang-Barsky: walk the four slabs, tightening the parameter
    interval, and bail out as soon as it becomes empty. The shifted
    bridge segments begin near barrier vertices, so they generally
    need trimming to keep complete obstacles inside the course.
    """
    direction = end - start

    low = 0.0
    high = 1.0

    for numerator, denominator in (
        (start[0] - inset_low_x, -direction[0]),
        (inset_high_x - start[0], direction[0]),
        (start[1] - inset_low_y, -direction[1]),
        (inset_high_y - start[1], direction[1]),
    ):
        if denominator == 0.0:
            if numerator < 0.0:
                return None

            continue

        ratio = numerator / denominator

        if denominator < 0.0:
            if ratio > high:
                return None

            low = max(low, ratio)
        else:
            if ratio < low:
                return None

            high = min(high, ratio)

    if high <= low:
        return None

    return low, high

def free_parameter_intervals(
    start: np.ndarray,
    end: np.ndarray,
    span: tuple[float, float],
    polygons: list[Polygon]
) -> list[tuple[float, float]]:
    """Intervals whose shifted-line centers lie in current free space.

    The wall-clipped line is differenced from the current occupied
    polygon union with Shapely. Each remaining LineString component is
    then converted back to the original segment parameter.
    """
    low, high = span
    direction = end - start
    squared_length = float(np.dot(direction, direction))

    clipped_start = start + low * direction
    clipped_end = start + high * direction
    clipped_line = LineString((clipped_start, clipped_end))
    occupied_region = shapely.union_all(polygons)
    free_line = shapely.difference(clipped_line, occupied_region)
    intervals = []

    for part in shapely.get_parts(free_line):
        if part.geom_type != "LineString" or part.length <= 0.0:
            continue

        coordinates = np.asarray(part.coords, dtype=float)
        parameters = (
            (coordinates - start) @ direction / squared_length
        )
        interval_low = max(low, float(parameters.min()))
        interval_high = min(high, float(parameters.max()))

        if interval_high > interval_low:
            intervals.append((interval_low, interval_high))

    intervals.sort()
    return intervals

def intersect_intervals(
    first: list[tuple[float, float]],
    second: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Pairwise intersections of two sorted interval collections."""
    intersections = []
    first_index = 0
    second_index = 0

    while first_index < len(first) and second_index < len(second):
        low = max(
            first[first_index][0],
            second[second_index][0],
        )
        high = min(
            first[first_index][1],
            second[second_index][1],
        )

        if high > low:
            intersections.append((low, high))

        if first[first_index][1] < second[second_index][1]:
            first_index += 1
        else:
            second_index += 1

    return intersections

def bridge_segment(path_vertices, obstacle_a_vertices, obstacle_b_vertices):
    """Return the path's unique direct edge from A to B.

    The returned edge is always oriented from obstacle A toward
    obstacle B. Paths without exactly one such edge are ignored.
    """
    crossings = []

    for first, second in itertools.pairwise(path_vertices):
        first_on_a = first in obstacle_a_vertices
        first_on_b = first in obstacle_b_vertices
        second_on_a = second in obstacle_a_vertices
        second_on_b = second in obstacle_b_vertices

        if first_on_a and second_on_b:
            crossings.append((first, second))
        elif first_on_b and second_on_a:
            crossings.append((second, first))

    if len(crossings) != 1:
        return None

    return crossings[0]

# ------------------------------------------------------------------
# Chains of small obstacles
# ------------------------------------------------------------------

def diameter_fit_count(length: float, diameter: float) -> int:
    """How many obstacle diameters fit strictly inside ``length``.

    The count is the largest ``n`` with ``n * diameter < length``, so a
    segment exactly eleven diameters long takes ten obstacles rather than
    eleven and the chain never runs flush to the ends of its own segment.
    """
    if diameter <= 0.0:
        raise ValueError("diameter must be positive.")

    if length <= 0.0:
        return 0

    return max(int(np.ceil(length / diameter)) - 1, 0)


def chain_offsets(
    length: float,
    diameter: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Arc-length positions of an evenly spaced chain inside ``length``.

    Consecutive centers sit exactly one diameter apart. Spacing the chain
    by the diameter rather than by ``length / (count + 1)`` is what makes
    non-overlap structural: two obstacles whose vertices never exceed
    ``max_radius`` from their centers cannot meet once those centers are a
    full diameter apart, whatever the annulus draw does.

    The leftover slack, always at least one diameter by the definition of
    the count, becomes a random offset of the chain as a whole, so a
    repeated attempt is not a repeat of the same layout.
    """
    count = diameter_fit_count(length, diameter)

    if count == 0:
        return np.zeros(0, dtype=float)

    slack = length - (count - 1) * diameter

    return rng.uniform(0.0, slack) + np.arange(count) * diameter


def staggered_centers(
    start: np.ndarray,
    end: np.ndarray,
    span: tuple[float, float],
    diameter: float,
    vertical_offset: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Evenly spaced centers along a segment, alternating above and below.

    Points are laid out on the segment itself and then projected
    vertically onto two copies of it, one shifted up by
    ``vertical_offset`` and one down, taking alternate points. Staggering
    neighbours this way keeps them from sitting nose to tail and widens
    the gap between consecutive centers beyond the diameter.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    direction = end - start
    total_length = float(np.linalg.norm(direction))

    if total_length <= 0.0:
        return np.zeros((0, 2), dtype=float)

    low, high = span
    offsets = chain_offsets(total_length * (high - low), diameter, rng)

    if offsets.size == 0:
        return np.zeros((0, 2), dtype=float)

    parameters = low + offsets / total_length
    centers = start + parameters[:, None] * direction

    # Which side the chain starts on is a free choice, so it is randomised
    # rather than always beginning above the line.
    leading_sign = 1.0 if rng.random() < 0.5 else -1.0
    signs = leading_sign * np.where(
        np.arange(len(centers)) % 2 == 0,
        1.0,
        -1.0,
    )

    centers[:, 1] += signs * vertical_offset

    return centers


# Fraction of ``corner_fraction`` below which a chain segment is not drawn.
# Near the corner the wall margin eats most of the segment, so a low draw
# leaves room for no obstacle at all and the corner goes unblocked. Holding
# the draw to the outer fifth of the permitted range keeps every chain long
# enough to matter while staying inside that range.
def place_random_obstacles(
    count: int,
    rng: np.random.Generator,
    min_vertices: int,
    max_vertices: int,
    min_radius: float,
    max_radius: float,
    polygons: list[Polygon],
    course_region: Polygon,
    center_region: Polygon,
    attempts_per_obstacle: int = 64,
) -> int:
    """Add obstacles whose centers are drawn from the remaining free space.

    Sampling the center out of ``center_region`` with everything placed so
    far removed is what stops a center landing inside an existing
    obstacle; the annulus sampler then rejects any candidate whose body
    would overlap. Returns how many were actually placed, which falls
    short of ``count`` only when the course has run out of room.
    """
    placed = 0

    for _ in range(count):
        free_space = shapely.difference(
            center_region,
            shapely.union_all(polygons),
        )

        if free_space.is_empty or free_space.area <= 0.0:
            return placed

        try:
            candidates = sample_uniform_points(
                free_space,
                attempts_per_obstacle,
                rng,
            )
        except RuntimeError:
            return placed

        for center in candidates:
            obstacle = annulus_obstacle(
                center,
                rng,
                min_vertices,
                max_vertices,
                min_radius,
                max_radius,
                polygons,
                course_region,
                max_attempts=64,
                raise_on_failure=False,
            )

            if obstacle is not None:
                polygons.append(obstacle)
                placed += 1
                break
        else:
            return placed

    return placed

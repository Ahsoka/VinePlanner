import numpy as np
import itertools

from numba import njit

from functools import lru_cache

from .constants import (
    YIELD_PRESSURE_PSI,
    LENGTH_FRICTION_PSI_PER_FT,
    TAIL_TENSION_PSI,
    CURVATURE_FRICTION_COEFFICIENT,
    _LENGTH_TO_FEET,
    _PRESSURE_TO_PSI,
    _PRESSURE_LABELS,
)

# ------------------------------------------------------------------
# Pressure model
# ------------------------------------------------------------------
#
# Recursive (nested capstan) growth pressure. Writing the paper's symbols
# for a path of n segments with lengths l_1 ... l_n and n - 1 interior
# turning angles theta_1 ... theta_{n-1}:
#
#     tau_0 = T
#     tau_i = (tau_{i-1} + lambda * l_i) * exp(mu * |theta_i|)   i = 1 ... n-1
#     P     = Y + tau_{n-1} + lambda * l_n
#
# where P is the growth pressure, Y the yield pressure required to evert
# the vine, lambda the length dependent friction term between the inside
# of the body and the tail, T the tension in the tail from the tug of the
# reel, and mu the Capstan friction coefficient for a bend. Dividing the
# governing equation
#
#     P A = Y A + lambda L + T exp(mu theta)
#
# through by the cross sectional area A is what puts lambda and T in
# pressure units, which is how they are carried here; see
# `vine_robot_pressure`.
#
# Each segment's drag is amplified only by the bends *distal* to it, i.e.
# the bends its accumulated tension must still be dragged through on the
# way to the tip. The terminal segment has nothing distal to it and is
# therefore charged at face value.
#
# This replaces the summed form P = Y + (lambda L + T) exp(mu sum|theta_i|),
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

    Accepts any ``(n + 1, d)`` array of vertices, in
    two or three dimensions. Consecutive duplicate vertices are dropped.
    Turning angles are unsigned and reported in radians; there is one
    per interior vertex, so ``len(angles) == len(lengths) - 1``.
    """
    coords = np.asarray(path, dtype=float)

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


# ------------------------------------------------------------------
# Pressure units
# ------------------------------------------------------------------
#
# The model is psi throughout: the calibrated parameters are psi and psi
# per foot, and `vine_robot_pressure` returns psi. Pressure units are
# therefore a display concern only, exactly as `length_units` is for the
# axis of a plot -- converting a result cannot change the physics or the
# path that was chosen.
#
# The tables themselves live in `constants`, alongside `_LENGTH_TO_FEET`,
# since they are fixed definitions rather than behaviour.


def pressure_conversion(source_units: str, target_units: str) -> float:
    """Multiplier taking a pressure in ``source_units`` to ``target_units``."""
    for units in (source_units, target_units):
        if str(units).lower() not in _PRESSURE_TO_PSI:
            raise ValueError(
                "pressure units must be one of: 'psi', 'Pa', 'kPa', 'MPa', "
                f"or 'bar'; got {units!r}."
            )

    return (
        _PRESSURE_TO_PSI[str(source_units).lower()]
        / _PRESSURE_TO_PSI[str(target_units).lower()]
    )


def resolve_pressure_units(pressure_units: str = "psi") -> tuple[str, float]:
    """Display label and the factor converting model psi into those units.

    Returns the properly cased label, so a caller passing ``"kpa"`` still
    gets ``"kPa"`` on the axis.
    """
    if pressure_units is None:
        return "psi", 1.0

    key = str(pressure_units).lower()

    return _PRESSURE_LABELS[key], pressure_conversion("psi", key)


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
        The model parameters ``Y``, ``lambda``, ``T`` and ``mu`` of the
        governing equation

            P A = Y A + lambda L + T exp(mu theta),

        where ``P`` is the growth pressure, ``A`` the cross sectional area
        of the vine robot, ``Y`` the yield pressure required to evert the
        vine, ``lambda`` a length dependent friction term accounting for
        the friction between the inside of the body and the tail in
        straight sections, ``L`` the total length of the robot, ``T`` the
        tension in the tail caused by the tug of the reel, ``mu`` the
        Capstan friction coefficient for the friction between the tail and
        the inside of the body in a bend, and ``theta`` in ``[0, pi]`` the
        angle of the turn in radians.

        Divided through by ``A``, both ``lambda`` and ``T`` carry pressure
        units, which is how they are taken here. They default to the
        fitted values in ``constants``. ``length_friction`` is psi per foot
        regardless of
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

    # tau_i = (tau_{i-1} + lambda * l_i) * exp(mu * |theta_i|), then the
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
    """Growth pressure of an ``(n + 1, d)`` vertex array, in psi."""
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


def _significant(value: float, digits: int = 4) -> str:
    """Format to a fixed number of significant figures, in positional notation.

    ``%g`` is the obvious choice and is wrong twice over here. It strips
    trailing zeros, so 36.2042 kPa prints as ``36.2`` and silently drops to
    three significant figures while 3.857 psi keeps four; and it switches to
    scientific notation, so the same pressure in pascals becomes
    ``2.659e+04`` in the middle of a legend entry.

    Choosing the decimal count from the magnitude instead keeps the figure
    count honest across the six orders of magnitude between pascals and
    megapascals, and reproduces the old ``.3f`` psi formatting exactly, since
    a pressure of a few psi has its first significant figure in the units
    place.
    """
    if not np.isfinite(value) or value == 0.0:
        return f"{value:.{max(digits - 1, 0)}f}"

    exponent = int(np.floor(np.log10(abs(value))))

    return f"{value:.{max(digits - 1 - exponent, 0)}f}"


def legend_label(
    name: str,
    fields=_DEFAULT_LEGEND_FIELDS,
    *,
    length: float = None,
    angle: float = None,
    pressure: float = None,
    length_units: str = "ft",
    angle_units: str = "deg",
    pressure_units: str = "psi",
    pressure_name: str = "pressure",
) -> str:
    """``"min pressure (length=14.75 m, pressure=26.59 kPa)"``.

    A field is omitted when it was not selected *or* when its value was not
    supplied, so a caller that has no angle to report does not have to edit
    its field list to match.

    ``pressure`` is expected already in ``pressure_units``; this only labels
    it. Conversion belongs with the caller that also has to rescale the axis,
    so that a plotted curve and its legend cannot end up in different units.

    Pressures are shown to four significant figures rather than a fixed
    number of decimals, because the same result spans six orders of magnitude
    between pascals and megapascals. Four figures reproduces the previous
    ``.3f`` psi formatting exactly.
    """
    values = {"length": length, "angle": angle, "pressure": pressure}
    units = {
        "length": length_units,
        "angle": angle_units,
        "pressure": pressure_units,
    }
    labels = {"length": "length", "angle": "angle", "pressure": pressure_name}

    def render(field: str) -> str:
        value = values[field]

        # Lengths and angles stay on two decimals: both are read off an axis
        # whose own units the caller already chose, so their magnitude is
        # bounded. Pressure is the one that has to survive psi through pascals.
        if field == "pressure":
            return _significant(value)

        return f"{value:.2f}"

    parts = [
        f"{labels[field]}={render(field)} {units[field]}"
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
    packing_coefficient: float = 0.6,
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
    packing_coefficient: float = 0.6,
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


def path_metrics(path) -> tuple[float, float]:
    """Return path length and cumulative unsigned turning angle.

    Retained for reporting. The cumulative angle is no longer sufficient
    to determine pressure under the recursive model, which depends on
    where each bend sits along the path.
    """
    lengths, angles = segment_geometry(path)

    return float(lengths.sum()), float(angles.sum())


def pressure_profile(
    path,
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
    freshly deployed body adds drag at the constant rate ``lambda`` and
    the profile rises linearly within each segment. Rounding a vertex
    multiplies the accumulated tension by ``exp(mu * |theta|)``, so the
    profile contains two pressure values at that same cumulative length
    to display the resulting jump. (Under the old summed model the slope
    itself grew after every turn; here it does not.)
    """
    length_to_feet = _length_scale(length_units)

    if not isinstance(points_per_segment, int) or points_per_segment < 1:
        raise ValueError("points_per_segment must be a positive integer.")

    coords = np.asarray(path, dtype=float)

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

# ------------------------------------------------------------------
# Convex polygon primitives
# ------------------------------------------------------------------
#
# Every obstacle in this package is a convex polytope, which is what lets
# one visibility kernel serve both the 2D and the 3D course (see
# `VisibilityMixin`). These are the handful of predicates that replaced
# Shapely once that decision was made: for convex bodies each one is a few
# dot products, so the general-purpose library was doing far more work
# than the geometry needs.
#
# The half-space convention is fixed once here and relied on everywhere:
# rows `a` are *outward* and *unit length*, and a point is inside when
# `a . x + b <= 0`. Unit length is load bearing rather than cosmetic --
# `b -= r` is only a true distance offset when `|a| = 1`, which is what
# makes dilating an obstacle by the vine radius a single subtraction.


@njit(cache=True, nogil=True)
def _point_in_convex_kernel(point, polygon, tol):
    """Is ``point`` inside a convex polygon given by its corners?

    One signed area per edge. For a convex polygon wound either way the
    point is inside when no edge puts it strictly on the outward side, so
    the test is whether the cross products ever take both signs.
    """
    positive = False
    negative = False

    for k in range(polygon.shape[0]):
        following = k + 1 if k + 1 < polygon.shape[0] else 0

        cross = (
            (polygon[following, 0] - polygon[k, 0])
            * (point[1] - polygon[k, 1])
            - (polygon[following, 1] - polygon[k, 1])
            * (point[0] - polygon[k, 0])
        )

        if cross > tol:
            positive = True
        elif cross < -tol:
            negative = True

        if positive and negative:
            return False

    return True


@njit(cache=True, nogil=True)
def _convex_overlap_kernel(first, second, tol):
    """Separating axis test between two convex polygons.

    Compiled rather than vectorised because the polygons have four corners
    on average: at that size the per-call overhead of array operations costs
    far more than the arithmetic they save, and this runs once per candidate
    per neighbour over the whole course.
    """
    for which in range(2):
        polygon = first if which == 0 else second

        for k in range(polygon.shape[0]):
            following = k + 1 if k + 1 < polygon.shape[0] else 0

            axis_x = polygon[following, 1] - polygon[k, 1]
            axis_y = -(polygon[following, 0] - polygon[k, 0])

            length = np.sqrt(axis_x * axis_x + axis_y * axis_y)

            if length <= 0.0:
                continue

            axis_x /= length
            axis_y /= length

            low_a = 1.0e30
            high_a = -1.0e30
            for p in range(first.shape[0]):
                value = axis_x * first[p, 0] + axis_y * first[p, 1]
                if value < low_a:
                    low_a = value
                if value > high_a:
                    high_a = value

            low_b = 1.0e30
            high_b = -1.0e30
            for p in range(second.shape[0]):
                value = axis_x * second[p, 0] + axis_y * second[p, 1]
                if value < low_b:
                    low_b = value
                if value > high_b:
                    high_b = value

            # A gap on any one axis is a certificate of disjointness.
            if low_a >= high_b - tol or low_b >= high_a - tol:
                return False

    return True


def polygon_is_convex(vertices, tol: float = 1e-12) -> bool:
    """Is this closed polygon convex and consistently wound?

    The cross product of consecutive edge vectors keeps one sign around a
    convex polygon and changes sign at a reflex corner, so one pass over
    the corners decides it. Collinear corners give zero and are accepted:
    they add a redundant half-space but no concavity.
    """
    points = np.asarray(vertices, dtype=float)

    if len(points) < 3:
        return False

    edges = np.roll(points, -1, axis=0) - points
    following = np.roll(edges, -1, axis=0)
    crosses = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]

    return not (np.any(crosses > tol) and np.any(crosses < -tol))


def polygon_area(vertices) -> float:
    """Unsigned area of a simple polygon, by the shoelace formula."""
    points = np.asarray(vertices, dtype=float)

    if len(points) < 3:
        return 0.0

    rolled = np.roll(points, -1, axis=0)

    return 0.5 * abs(float(np.sum(
        points[:, 0] * rolled[:, 1] - rolled[:, 0] * points[:, 1]
    )))


def polygon_half_spaces(vertices) -> tuple[np.ndarray, np.ndarray]:
    """Outward unit normals and offsets of a convex polygon.

    Returns ``(normals, offsets)`` with one row per edge, so that the
    polygon is exactly ``{x : normals @ x + offsets <= 0}``. The winding is
    detected rather than assumed, so a clockwise polygon yields outward
    normals just as a counter-clockwise one does.
    """
    points = np.asarray(vertices, dtype=float)

    if len(points) < 3:
        return np.zeros((0, 2)), np.zeros(0)

    edges = np.roll(points, -1, axis=0) - points

    # Signed area picks the winding. The right-hand normal of an edge points
    # outwards for a counter-clockwise polygon and inwards for a clockwise
    # one, so the sign below has to flip with it.
    rolled = np.roll(points, -1, axis=0)
    signed_area = 0.5 * float(np.sum(
        points[:, 0] * rolled[:, 1] - rolled[:, 0] * points[:, 1]
    ))
    orientation = 1.0 if signed_area > 0.0 else -1.0

    normals = orientation * np.column_stack((edges[:, 1], -edges[:, 0]))
    lengths = np.linalg.norm(normals, axis=1)

    keep = lengths > 0.0
    normals = normals[keep] / lengths[keep, None]

    # b = -a . p for any point p lying on the edge.
    offsets = -np.einsum("ij,ij->i", normals, points[keep])

    return normals, offsets


def point_in_convex(point, normals, offsets, tol: float = 0.0) -> bool:
    """Is ``point`` inside the polytope ``normals @ x + offsets <= 0``?

    ``tol`` shifts every face outwards, so a positive value counts a point
    just outside as inside and a negative value excludes the boundary.
    """
    if len(offsets) == 0:
        return False

    return bool(np.all(
        normals @ np.asarray(point, dtype=float) + offsets <= tol
    ))


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


class _ObstacleGrid:
    """Uniform-bin index over the obstacles already placed, keyed by bounds.

    Rejecting a candidate obstacle means asking whether it overlaps anything
    already down. Scanning the whole placed list costs O(N) per candidate and
    O(N^2) over a course, which is what made a thousand-obstacle course slow.

    Two polygons can only meet if their bounding boxes meet, so binning each
    one into the cells its box covers and reading back only the occupants of
    the cells a candidate's box covers loses nothing: every polygon left out
    is disjoint from the candidate, so testing it could not have changed the
    answer. One cell per obstacle diameter keeps a typical read to a handful
    of polygons however many are down, which is what turns the quadratic into
    a linear pass.

    Bounds rather than centers because the two corner obstacles reach past the
    course wall, so a center plus a radius does not enclose them.
    """

    __slots__ = ("cell", "bins")

    def __init__(self, cell: float):
        if cell <= 0:
            raise ValueError("cell must be positive.")

        self.cell = float(cell)
        self.bins = {}

    def _keys(self, bounds):
        """Every cell the axis-aligned box ``bounds`` touches."""
        low_x, low_y, high_x, high_y = bounds
        cell = self.cell

        # Float floor division, so negative coordinates bin downwards rather
        # than towards zero and the lattice stays uniform across the origin.
        for index_x in range(int(low_x // cell), int(high_x // cell) + 1):
            for index_y in range(int(low_y // cell), int(high_y // cell) + 1):
                yield index_x, index_y

    def add(self, vertices) -> None:
        """Bin one polygon into every cell its bounding box touches."""
        points = np.asarray(vertices, dtype=float)
        bounds = (
            points[:, 0].min(),
            points[:, 1].min(),
            points[:, 0].max(),
            points[:, 1].max(),
        )

        for key in self._keys(bounds):
            self.bins.setdefault(key, []).append(points)

    def nearby(self, bounds) -> list:
        """Every binned polygon whose box could meet the box ``bounds``.

        A polygon spanning several of the queried cells is returned once: the
        caller scans the result, and a duplicate would be wasted work.
        """
        found = {}

        for key in self._keys(bounds):
            for polygon in self.bins.get(key, ()):
                found[id(polygon)] = polygon

        return list(found.values())


def annulus_obstacle(
    center: np.ndarray,
    rng: np.random.Generator,
    min_vertices: int,
    max_vertices: int,
    min_radius: float,
    max_radius: float,
    polygons: list,
    course_bounds,
    *,
    forbidden_points=(),
    spanning_axis=None,
    containment_bounds=None,
    grid: "_ObstacleGrid" = None,
    max_attempts: int = 5000,
    raise_on_failure: bool = True,
):
    """Sample one convex annulus obstacle inside the current free space.

    Angles and radii retain the original annulus construction. A draw is
    accepted only when the polygon is convex, lies entirely within the
    containing box, contains its own nominal center, and shares no interior
    area with any obstacle already placed. Touching an existing obstacle
    along an edge or at a corner is allowed; overlapping it is not.

    Convexity is enforced by rejection rather than by hulling the draw. The
    two differ: hulling would accept a reflex draw and quietly replace it
    with a slightly larger obstacle, and doing that after the overlap test
    could grow an obstacle into a neighbour it had already been cleared
    against. Rejecting instead means the polygon that was tested is the
    polygon that is stored. Roughly one draw in eight is discarded at the
    default radii, which the attempt budget absorbs easily.

    ``forbidden_points`` are future line centers. Rejecting candidates that
    cover any of them prevents an earlier obstacle from consuming the center
    a later obstacle needs.

    ``containment_bounds`` replaces ``course_bounds`` when an obstacle is
    allowed to reach past the course walls, which only the two corner
    obstacles are. It relaxes the wall constraint alone: overlap with an
    existing obstacle stays forbidden.

    ``grid`` restricts the overlap test to obstacles binned near the
    candidate. Every obstacle it omits has a bounding box disjoint from the
    candidate's and therefore cannot overlap it, so the result is identical
    to scanning ``polygons`` and the cost stops growing with the course.
    ``polygons`` goes unread when a grid is supplied.

    ``spanning_axis`` forces two of the vertices onto the line through the
    center in that direction, one on each side, each at its own radius drawn
    from the annulus, and places one further vertex on each side of that line
    so the ray along the axis runs through the obstacle's interior rather
    than along an edge. Four vertices are therefore the minimum when an axis
    is given. The obstacle then reaches its full width across that direction
    instead of whatever width the angles happened to give it, which is how a
    normal-sized obstacle is made to block a specific line. Two opposite
    vertices also cap every angular gap at pi, so the sorted polygon cannot
    self-intersect.
    """
    center = np.asarray(center, dtype=float)
    bounds = course_bounds if containment_bounds is None else containment_bounds
    low_x, low_y, high_x, high_y = bounds

    forbidden = (
        np.asarray(forbidden_points, dtype=float)
        if len(forbidden_points)
        else None
    )

    # Only obstacles whose bounding box meets the candidate's can overlap it,
    # and every candidate drawn below has its vertices within max_radius of
    # the center, so this box encloses all of them.
    if grid is None:
        neighbours = [np.asarray(polygon, dtype=float) for polygon in polygons]
    else:
        neighbours = grid.nearby((
            center[0] - max_radius,
            center[1] - max_radius,
            center[0] + max_radius,
            center[1] + max_radius,
        ))

    # The center itself has to be free, or there is nothing to sample around.
    for neighbour in neighbours:
        if _point_in_convex_kernel(center, neighbour, 0.0):
            if raise_on_failure:
                raise RuntimeError(
                    "The requested obstacle center is not in the current "
                    "free space."
                )

            return None

    if not (low_x <= center[0] <= high_x and low_y <= center[1] <= high_y):
        if raise_on_failure:
            raise RuntimeError(
                "The requested obstacle center lies outside the region the "
                "obstacle must be contained by."
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
            # The two forced vertices sit half a turn apart on the spanning
            # axis; the rest are ordinary uniform draws. Sorting the combined
            # set keeps the polygon star shaped about the center, and no gap
            # can exceed pi because the forced pair already splits the circle
            # in half.
            axis_angle = np.arctan2(
                float(spanning_axis[1]),
                float(spanning_axis[0]),
            )

            # One free vertex is placed on each side of the axis before the
            # rest are drawn over the whole circle. Left to chance they can
            # all land on one side, which leaves a gap of exactly pi: the edge
            # joining the forced pair then runs straight through the center,
            # so the center and every point along the axis ray lie on the
            # boundary rather than inside it. Straddling makes every gap
            # strictly less than pi, which puts the whole ray out to the
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

        radii = rng.uniform(min_radius, max_radius, number_of_vertices)
        candidate = (
            center
            + radii[:, None] * np.column_stack((
                np.cos(angles),
                np.sin(angles),
            ))
        )

        # Every obstacle in the package is a convex polytope; a reflex draw
        # is discarded rather than repaired.
        if not polygon_is_convex(candidate):
            continue

        if polygon_area(candidate) <= 0.0:
            continue

        # The box is convex, so containing every vertex is the same as
        # containing the whole polygon.
        if (
            candidate[:, 0].min() < low_x
            or candidate[:, 0].max() > high_x
            or candidate[:, 1].min() < low_y
            or candidate[:, 1].max() > high_y
        ):
            continue

        normals, offsets = polygon_half_spaces(candidate)

        # The nominal center has to be strictly inside, which is what gives
        # the supplied line-placement point its meaning.
        if not point_in_convex(center, normals, offsets, tol=-1e-12):
            continue

        if forbidden is not None and np.any(
            np.all(forbidden @ normals.T + offsets <= 0.0, axis=1)
        ):
            continue

        # Vertices being free is not sufficient by itself: an edge can cross
        # an occupied polygon while both endpoints remain free. The separating
        # axis test covers the whole body, so obstacle interiors stay disjoint
        # while boundary contact remains allowed.
        contiguous = np.ascontiguousarray(candidate)

        if any(
            _convex_overlap_kernel(contiguous, neighbour, 1e-12)
            for neighbour in neighbours
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
    polygons: list,
) -> list[tuple[float, float]]:
    """Intervals of the wall-clipped line that lie in current free space.

    Each obstacle is convex, so the part of the line inside it is a single
    parameter interval found by clipping the line against that obstacle's
    half-spaces. Subtracting the union of those intervals from the clipped
    span leaves the free ones, which is the same answer a polygon-set
    difference would give without needing a polygon set.
    """
    low, high = span
    direction = end - start
    squared_length = float(np.dot(direction, direction))

    if squared_length <= 0.0 or high <= low:
        return []

    blocked = []

    for polygon in polygons:
        normals, offsets = polygon_half_spaces(polygon)

        if len(offsets) == 0:
            continue

        # f_i(t) = (a_i . d) t + (a_i . start + b_i), inside where every
        # f_i <= 0. Each face tightens one end of the surviving interval.
        denominators = normals @ direction
        numerators = normals @ start + offsets

        enter, exit_ = low, high

        for denominator, numerator in zip(denominators, numerators):
            if abs(denominator) <= 1e-15:
                # Parallel to this face: either the whole line is outside it
                # or the face constrains nothing.
                if numerator > 0.0:
                    enter = 1.0
                    exit_ = 0.0
                    break

                continue

            crossing = -numerator / denominator

            if denominator > 0.0:
                exit_ = min(exit_, crossing)
            else:
                enter = max(enter, crossing)

            if enter >= exit_:
                break

        if exit_ > enter:
            blocked.append((enter, exit_))

    if not blocked:
        return [(low, high)]

    blocked.sort()

    intervals = []
    cursor = low

    for interval_low, interval_high in blocked:
        if interval_low > cursor:
            intervals.append((cursor, min(interval_low, high)))

        cursor = max(cursor, interval_high)

        if cursor >= high:
            break

    if cursor < high:
        intervals.append((cursor, high))

    return [
        (interval_low, interval_high)
        for interval_low, interval_high in intervals
        if interval_high > interval_low
    ]

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
    polygons: list,
    course_bounds,
    center_bounds,
    attempts_per_obstacle: int = 64,
) -> int:
    """Add obstacles whose centers are drawn from the remaining free space.

    Sampling the center out of ``center_region`` with everything placed so
    far removed is what stops a center landing inside an existing
    obstacle; the annulus sampler then rejects any candidate whose body
    would overlap. Returns how many were actually placed, which falls
    short of ``count`` only when the course has run out of room.

    Both halves of that are done against a ``_ObstacleGrid`` rather than
    against the whole placed list. Drawing uniformly from the bounding box
    of ``center_region`` and throwing away the draws that land on an
    obstacle is uniform over the free part of it, which is the distribution
    the triangulation of that free part used to sample; the grid answers
    the "lands on an obstacle" question in constant time. Note that the two
    consume random numbers differently, so a seed that reproduced a given
    course before will not reproduce it now.
    """
    placed = 0

    if count <= 0:
        return placed

    # One cell per obstacle diameter: big enough that a candidate reads only
    # its own cell and the ones it straddles, small enough that each read
    # stays a handful of polygons at any course size.
    grid = _ObstacleGrid(2.0 * max_radius)

    for polygon in polygons:
        grid.add(polygon)

    low_x, low_y, high_x, high_y = center_bounds

    # A ceiling on draws as well as on centers, so a course with no room
    # left returns instead of spinning: without it the rejection loop keeps
    # drawing occupied points and never reaches the center count.
    maximum_draws = 64 * attempts_per_obstacle

    for _ in range(count):
        obstacle = None
        centers_tried = 0
        draws = 0

        while centers_tried < attempts_per_obstacle and draws < maximum_draws:
            draws += 1

            center = np.array([
                rng.uniform(low_x, high_x),
                rng.uniform(low_y, high_y),
            ])

            if any(
                _point_in_convex_kernel(center, neighbour, 0.0)
                for neighbour in grid.nearby(
                    (center[0], center[1], center[0], center[1])
                )
            ):
                continue

            centers_tried += 1

            obstacle = annulus_obstacle(
                center,
                rng,
                min_vertices,
                max_vertices,
                min_radius,
                max_radius,
                polygons,
                course_bounds,
                grid=grid,
                max_attempts=64,
                raise_on_failure=False,
            )

            if obstacle is not None:
                break

        if obstacle is None:
            return placed

        polygons.append(obstacle)
        grid.add(obstacle)
        placed += 1

    return placed

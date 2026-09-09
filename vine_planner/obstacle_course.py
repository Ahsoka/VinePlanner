import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
import numpy as np
import itertools
import shapely

from .constants import _LENGTH_TO_FEET
from shapely.plotting import plot_polygon
from attrs import define, field, validators
from shapely import Polygon
from .utils import (
    clip_segment,
    legend_label,
    resolve_pressure_units,
    obstacle_capacity,
    suggested_radius,
    ObstacleCapacityWarning,
    path_metrics,
    path_pressure,
    bridge_segment,
    annulus_obstacle,
    staggered_centers,
    sample_uniform_points,
    intersect_intervals,
    diameter_fit_count,
    place_random_obstacles,
    free_parameter_intervals
)


# Every artist on a 2D axes is drawn below this, so the legend is never
# occluded no matter how many paths are added later.
_LEGEND_ZORDER = 100

# Budget for the k-shortest-paths search that picks the two bridges in
# ``ObstacleCourse.random``. scipy's yen() needs K up front, so the search
# starts small and doubles; the cap only exists to turn a course that can
# never satisfy the disjointness requirement into an error instead of a hang.
_INITIAL_PATH_BUDGET = 16
_MAXIMUM_PATH_BUDGET = 4096

# Range the obstacle count is drawn from when ``ObstacleCourse.random`` is not
# given one. The low end sits above what the structured layout needs on a
# square course at the default radius, so an unspecified count still gets the
# structured layout rather than the free-placement fallback.
_DEFAULT_MINIMUM_OBSTACLES = 30
_DEFAULT_MAXIMUM_OBSTACLES = 40

# Corner of the course each randomly drawn goal is confined to, as a fraction
# of the dimensions: the start comes from the bottom-left box of this size and
# the goal from the mirrored top-right one. A quarter keeps the two boxes well
# clear of one another, so the pair is always separated by at least half the
# course in both axes.
_GOAL_BOX_FRACTION = 0.25


def _length_conversion(source_units: str, target_units: str) -> float:
    """Multiplier taking a length in ``source_units`` to ``target_units``.

    Both are resolved through the feet table, so this is exact for every pair
    the package supports.

    This is a *display* conversion and nothing more. It never touches the
    pressure model: a course is whatever physical size it is, and rescaling
    the numbers used to describe it does not change the robot, the maze, or
    the optimal path through it.
    """
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


def _relabel_axis(axis, limits, display_scale: float) -> None:
    """Show ``axis`` in display units while leaving the data alone.

    Tick positions are chosen so the *labels* are round numbers, which means
    picking them in display space and dividing back into coordinate space.
    Reformatting the default ticks would keep the positions round in the wrong
    units and print things like ``2.08333``.
    """
    low, high = (value * display_scale for value in limits)

    locator = ticker.MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10])
    display_ticks = locator.tick_values(low, high)

    axis.set_major_locator(
        ticker.FixedLocator(display_ticks / display_scale)
    )
    axis.set_major_formatter(
        ticker.FuncFormatter(
            lambda value, _position: f"{value * display_scale:g}"
        )
    )


class VisibilityMixin:
    """All-pairs visibility on top of a course's batched ``is_visible``.

    A course only has to answer "are these segments clear?"; turning that into
    the pair list a visibility graph needs is identical for every course, so it
    lives here rather than in each class.

    Chunking is the reason this is not left to the caller. The pair count grows
    as ``n**2``: at 5,000 vertices that is 12.5 million pairs, and the index
    arrays alone run to hundreds of megabytes before any geometry is touched.
    """

    def visible_pairs(self, points, chunk: int = 2_000_000):
        """Index arrays ``(i, j)``, ``i < j``, of every mutually visible pair."""
        points = np.asarray(points, dtype=float)
        count = len(points)

        if count < 2:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty.copy()

        first, second = [], []

        # Walk the upper triangle a row-block at a time so the index arrays
        # never exist in full.
        rows_per_chunk = max(1, chunk // max(count, 1))

        for low in range(0, count - 1, rows_per_chunk):
            high = min(low + rows_per_chunk, count - 1)

            rows = np.arange(low, high)
            widths = count - rows - 1

            row_index = np.repeat(rows, widths)
            column_index = (
                np.arange(len(row_index))
                - np.repeat(np.concatenate([[0], np.cumsum(widths)[:-1]]), widths)
                + row_index
                + 1
            )

            visible = np.asarray(
                self.is_visible(points[row_index], points[column_index])
            )

            first.append(row_index[visible])
            second.append(column_index[visible])

        return np.concatenate(first), np.concatenate(second)


@define
class ObstacleCourse(VisibilityMixin):
    width: float = field(default=25.0, converter=float, validator=[validators.instance_of(float), validators.gt(0)])
    height: float = field(default=25.0, converter=float, validator=[validators.instance_of(float), validators.gt(0)])
    obstacles: list[Polygon] = field(factory=list)

    obstacles_region = field(init=False)

    def __attrs_post_init__(self):
        self.obstacles_region = shapely.union_all(self.obstacles)
        shapely.prepare(self.obstacles_region)

    @property
    def dimension(self) -> int:
        """Number of coordinates a point in this course carries."""
        return 2

    def generate_goals(self, start=None, end=None):
        """Fill in whichever of ``start``/``end`` was not supplied.

        Returns the pair with maximum straight-line separation inside the free
        region: the farthest pair when both are missing, and the farthest
        valid point from the one that was given otherwise.

        ``PathPlanner`` calls this without knowing which course it holds, so
        every course class owns its own goal placement.
        """
        if start is not None and end is not None:
            return np.asarray(start, dtype=float), np.asarray(end, dtype=float)

        course_region = shapely.box(0.0, 0.0, self.width, self.height)

        if self.obstacles:
            valid_region = shapely.difference(
                course_region,
                self.obstacles_region,
            )
        else:
            valid_region = course_region

        if valid_region.is_empty or valid_region.area <= 0:
            raise RuntimeError(
                "The obstacle course has no valid area for start "
                "and end points."
            )

        # The farthest pair of points in a polygonal region can be found
        # among the vertices of its convex hull.
        convex_hull = valid_region.convex_hull

        if convex_hull.geom_type != "Polygon":
            raise RuntimeError(
                "The valid region does not contain enough area to "
                "generate distinct start and end points."
            )

        # The final exterior coordinate repeats the first coordinate,
        # so remove it.
        candidates = np.asarray(
            convex_hull.exterior.coords,
            dtype=float,
        )[:-1, :2]

        if len(candidates) < 2:
            raise RuntimeError(
                "Not enough valid candidate points were found."
            )

        if start is None and end is None:
            # Squared distance is enough, since sqrt() does not change which
            # pair is largest.
            differences = (
                candidates[:, np.newaxis, :] - candidates[np.newaxis, :, :]
            )

            squared_distances = np.sum(differences**2, axis=2)

            start_index, end_index = np.unravel_index(
                np.argmax(squared_distances),
                squared_distances.shape,
            )

            return candidates[start_index].copy(), candidates[end_index].copy()

        if start is None:
            fixed = np.asarray(end, dtype=float)
            squared_distances = np.sum((candidates - fixed) ** 2, axis=1)
            return candidates[np.argmax(squared_distances)].copy(), fixed

        fixed = np.asarray(start, dtype=float)
        squared_distances = np.sum((candidates - fixed) ** 2, axis=1)
        return fixed, candidates[np.argmax(squared_distances)].copy()

    def valid_goal(self, coords):
        """
        Check if a point is a valid start or end point.
        Ff the point is in the obstacle course and the point
        is not inside any obstacles then it is a valid goal.
        """
        point = shapely.Point(
            float(coords[0]),
            float(coords[1]),
        )
        return (
            # Check if it's inside to obstacle course
            0 <= coords[0] <= self.width
            and 0 <= coords[1] <= self.height
            # Check if the point is inside any obstacles
            and not self.obstacles_region.contains(point)
        )


    @classmethod
    def random(
        cls,
        width: float = 25,
        height: float = 25,
        num_obstacles: int = None,
        seed: int = None,
        radius: float = None,
        annulus_fraction: float = 0.005,
        corner_offset_fraction: float = 0.8,
        min_vertices: int = 3,
        max_vertices: int = 6,
        raise_if_insufficient: bool = False,
        random_start: bool = False,
    ):
        """Generate a random course of uniformly sized obstacles.

        Every obstacle is the same size as every other, so the layout does
        not advertise which route it was built to reward. The structure
        comes from where two of them sit rather than from how big they are:
        one obstacle is placed on the line from ``(0, 0)`` to
        ``(width, height)`` exactly one maximum radius from each end. Two
        of its vertices are forced onto the line through that center
        perpendicular to the start-goal line, one on each side, each at
        its own radius from the annulus, so the obstacle reaches its full
        width across the route rather than whatever width the angles
        happened to give it. An ordinary obstacle placed and shaped that
        way closes off the straight run between the goals as effectively
        as a large barrier further out.

        A temporary course holding only those two is then solved from
        ``(0, 0)`` to ``(width, height)``. The shortest path is retained,
        then the shortest path whose direct obstacle-to-obstacle segment
        shares neither endpoint with the first. Each of those two segments
        is shifted up and down by ``radius``, and the obstacles that fit at
        one diameter apart are spaced evenly along it and projected onto
        the two shifted copies alternately, so neighbours are staggered.
        One further obstacle then straddles ``(0, height)`` and one
        straddles ``(width, 0)``. Each sits just inside its corner on the
        corner bisector with a vertex forced outwards along it, which puts
        the corner point inside the obstacle and that vertex outside the
        course. Covering the corner removes the right-angle route around the
        outside of the course; the part lying outside is dropped from
        ``vertices()``, so it never becomes a turning point.

        Because the count along each segment follows from the obstacle
        diameter, the course needs a specific number of obstacles rather
        than accepting any number. Given more, the surplus is placed with
        centers drawn uniformly from whatever free space is left. Given
        fewer, the structured layout is abandoned and every obstacle is
        placed in free space instead, unless ``raise_if_insufficient`` asks
        for an error.

        Every obstacle is sampled against the current Shapely free-space
        geometry: the union of all previously generated polygons is
        subtracted from the course region, and both the sampled vertices
        and the whole candidate polygon must be covered by what is left, so
        obstacle interiors never overlap and nothing crosses a wall.

        Parameters
        ----------
        num_obstacles:
            How many obstacles the course holds, counting the two goal
            obstacles and the two corner ones. Left as ``None`` the count is
            drawn uniformly from ``_DEFAULT_MINIMUM_OBSTACLES`` through
            ``_DEFAULT_MAXIMUM_OBSTACLES``, currently 30 to 40.
        radius:
            Inner sampling radius, ``min_radius``. Defaults to four
            percent of the smaller course dimension. It sets the spacing
            along the two segments, so halving it roughly doubles how many
            obstacles the layout needs.
        annulus_fraction:
            ``max_radius`` is ``radius`` plus this fraction of
            ``min(width, height)``.
        corner_offset_fraction:
            How far the two corner obstacles sit inside their corner,
            measured along the corner bisector as a fraction of
            ``min_radius``. Raising it puts more of the obstacle inside the
            course; it must stay below one, which is exactly the condition
            that keeps the corner point inside the obstacle and the
            obstacle's outward vertex outside the course.
        min_vertices, max_vertices:
            Bounds on the vertex count, inclusive.
        raise_if_insufficient:
            Raise instead of falling back to free placement when
            ``num_obstacles`` is below what the structured layout needs.
        random_start:
            Return ``(course, start, end)`` with a goal drawn at random from
            each of the two opposite corner boxes, instead of the course
            alone. Both points are guaranteed valid: neither lies in the
            interior of an obstacle. Note that this changes the return type,
            so a caller that passes it has to unpack the result.
        """

        rng = np.random.default_rng(seed)

        # ------------------------------------------------------------------
        # Validate arguments
        # ------------------------------------------------------------------

        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive.")

        if num_obstacles is None:
            num_obstacles = int(
                rng.integers(
                    _DEFAULT_MINIMUM_OBSTACLES,
                    _DEFAULT_MAXIMUM_OBSTACLES + 1,
                )
            )

        if num_obstacles < 1:
            raise ValueError("num_obstacles must be at least one.")

        if not 0.0 <= corner_offset_fraction < 1.0:
            raise ValueError(
                "corner_offset_fraction must lie in [0, 1). At one the "
                "corner obstacle's outward vertex could fall on the corner "
                "instead of beyond it, and neither guarantee would hold."
            )

        if min_vertices < 3:
            raise ValueError("min_vertices must be at least 3.")

        if max_vertices < min_vertices:
            raise ValueError(
                "max_vertices must be greater than or equal to "
                "min_vertices."
            )

        smaller_dimension = min(width, height)

        if radius is None:
            radius = 0.04 * smaller_dimension

        if radius <= 0:
            raise ValueError("radius must be positive.")

        min_radius = radius
        max_radius = min_radius + annulus_fraction * smaller_dimension

        # Centers along a segment are spaced by one diameter, the
        # separation at which two obstacles drawn from this annulus cannot
        # touch whatever the draw does.
        diameter = 2.0 * max_radius

        # Nudges every sampled coordinate strictly inside the course, so
        # that 0 < x < width and 0 < y < height rather than 0 <= x <= width.
        epsilon = 1e-9 * smaller_dimension

        # Keeping every center at least max_radius from each wall puts the
        # whole sampling disk inside the course, which is what guarantees
        # that every vertex satisfies 0 < x < width and 0 < y < height.
        inset_low_x = max_radius + epsilon
        inset_low_y = max_radius + epsilon
        inset_high_x = width - max_radius - epsilon
        inset_high_y = height - max_radius - epsilon

        if inset_high_x <= inset_low_x or inset_high_y <= inset_low_y:
            raise ValueError(
                "radius is too large for the course: no obstacle center "
                "can sit far enough from the walls to keep its vertices "
                "inside."
            )

        # A request far above what the course can hold is worth catching
        # here rather than after a long run that quietly returns fewer
        # obstacles than were asked for. The estimate is deliberately a
        # ceiling, so this fires only on a request that cannot be met.
        capacity = obstacle_capacity(
            width,
            height,
            min_vertices,
            max_vertices,
            min_radius,
            max_radius,
        )

        if num_obstacles > capacity:
            recommended = suggested_radius(
                width,
                height,
                num_obstacles,
                min_vertices,
                max_vertices,
                max_radius - min_radius,
            )
            warnings.warn(
                f"{num_obstacles} obstacles were requested but a "
                f"{width:g} x {height:g} course fits roughly {capacity} at "
                f"radius={min_radius:g}; expect the generator to fall short. "
                f"Pass radius={recommended:.3g} or smaller to fit "
                f"{num_obstacles}.",
                ObstacleCapacityWarning,
                stacklevel=2,
            )

        course_region = shapely.box(0.0, 0.0, width, height)
        center_region = shapely.box(
            inset_low_x,
            inset_low_y,
            inset_high_x,
            inset_high_y,
        )

        polygons = []

        def report_shortfall():
            """Warn when the course holds fewer obstacles than were asked for.

            ``place_random_obstacles`` returns how many it managed and the
            structured stages can each fall back, so a shortfall is otherwise
            invisible: the caller gets a valid course that simply is not the
            one it requested.
            """
            if len(polygons) < num_obstacles:
                warnings.warn(
                    f"{num_obstacles} obstacles were requested but only "
                    f"{len(polygons)} could be placed; the course ran out of "
                    "room. Reduce num_obstacles or radius if the exact count "
                    "matters.",
                    ObstacleCapacityWarning,
                    stacklevel=3,
                )

        def free_layout():
            """Place every obstacle in free space and return the course."""
            del polygons[:]
            place_random_obstacles(
                num_obstacles,
                rng,
                min_vertices,
                max_vertices,
                min_radius,
                max_radius,
                polygons,
                course_region,
                center_region,
            )
            report_shortfall()
            return cls(width=width, height=height, obstacles=polygons)

        def random_goals(course):
            """A valid start and end drawn from opposite corners of the course.

            Sampling uniformly from the corner box with the obstacles
            subtracted makes a draw valid by construction rather than by
            retrying until one happens to be. ``valid_goal`` is still applied
            on top: a point landing exactly on an obstacle boundary is inside
            the sampled region, and a path may not start from one.
            """
            goals = []

            for low_x, low_y, high_x, high_y in (
                (
                    0.0,
                    0.0,
                    _GOAL_BOX_FRACTION * width,
                    _GOAL_BOX_FRACTION * height,
                ),
                (
                    (1.0 - _GOAL_BOX_FRACTION) * width,
                    (1.0 - _GOAL_BOX_FRACTION) * height,
                    width,
                    height,
                ),
            ):
                free_corner = shapely.difference(
                    shapely.box(low_x, low_y, high_x, high_y),
                    course.obstacles_region,
                )

                if (
                    free_corner.is_empty
                    or free_corner.area <= 0.0
                    or free_corner.geom_type
                    not in ("Polygon", "MultiPolygon")
                ):
                    raise RuntimeError(
                        "No free space is left in the corner a random goal "
                        "is drawn from. Reduce num_obstacles or radius."
                    )

                # One triangulation covers every attempt, so the retry costs
                # nothing beyond the validity test itself.
                for candidate in sample_uniform_points(free_corner, 64, rng):
                    if course.valid_goal(candidate):
                        goals.append(np.asarray(candidate, dtype=float))
                        break
                else:
                    raise RuntimeError(
                        "Could not draw a valid goal from a corner of the "
                        "course."
                    )

            return goals

        def finish(course):
            """Attach randomly drawn goals when they were asked for."""
            if not random_start:
                return course

            return (course, *random_goals(course))

        # The two goal obstacles plus one on each of the other two corners.
        reserved_obstacles = 4

        if num_obstacles < reserved_obstacles:
            if raise_if_insufficient:
                raise ValueError(
                    f"num_obstacles is {num_obstacles} but the two goal "
                    f"obstacles and the two corner obstacles alone need "
                    f"{reserved_obstacles}."
                )

            return finish(free_layout())

        # ------------------------------------------------------------------
        # One obstacle hard against each goal
        # ------------------------------------------------------------------

        goal_line = np.array([width, height], dtype=float)
        goal_direction = goal_line / np.linalg.norm(goal_line)

        # The obstacle is forced to reach its full width across the
        # start-goal line rather than along it, so a normal-sized shape
        # blocks the straight run as widely as its radius allows.
        spanning_axis = np.array([
            -goal_direction[1],
            goal_direction[0],
        ])

        anchors = []

        for goal, inward in ((np.zeros(2), 1.0), (goal_line, -1.0)):
            # One max_radius from the goal is as close as the obstacle can
            # sit without covering it. At that distance the two walls
            # meeting at the corner leave only a narrow band of usable
            # vertex angles, so a draw occasionally cannot be satisfied at
            # all; stepping the center outwards along the line recovers it
            # without moving the obstacle far from the goal.
            for scale in (1.0, 1.15, 1.35, 1.6, 2.0):
                obstacle = annulus_obstacle(
                    goal + inward * scale * max_radius * goal_direction,
                    rng,
                    min_vertices,
                    max_vertices,
                    min_radius,
                    max_radius,
                    polygons,
                    course_region,
                    spanning_axis=spanning_axis,
                    raise_on_failure=False,
                )

                if obstacle is not None:
                    polygons.append(obstacle)
                    anchors.append(obstacle)
                    break

        if len(anchors) < 2:
            return finish(free_layout())

        # ------------------------------------------------------------------
        # One obstacle straddling each of the two remaining corners
        # ------------------------------------------------------------------

        # Each obstacle sits a short way inside its corner along the corner
        # bisector, and is forced to put a vertex on that bisector pointing
        # outwards. Both of the properties this needs then follow from the
        # single condition ``corner_offset < min_radius``.
        #
        # The polygon is star shaped about its center, so along the outward
        # ray its boundary is exactly the forced vertex, at radius at least
        # ``min_radius``. The corner point lies on that same ray at
        # ``corner_offset``, which is nearer, so the corner is strictly
        # inside the obstacle and is no longer a place a path may turn. The
        # forced vertex lies further along the ray than the corner, and past
        # the corner the bisector leaves the course through both walls at
        # once, so that vertex is strictly out of bounds and part of the
        # obstacle is guaranteed to be outside.
        #
        # Sitting inside the corner rather than on it is what puts most of
        # the obstacle in bounds: on the corner only the inward quadrant
        # counts, whereas here the center clears each wall by
        # ``corner_offset / sqrt(2)``.
        #
        # Reaching past a wall is allowed here and nowhere else, so the
        # containment test is widened for these two draws alone; overlap
        # with an already placed obstacle is still forbidden. The outside
        # vertices are dropped by ``vertices()`` and never reach the
        # visibility graph.
        corner_offset = corner_offset_fraction * min_radius

        outside_region = shapely.box(
            -2.0 * max_radius,
            -2.0 * max_radius,
            width + 2.0 * max_radius,
            height + 2.0 * max_radius,
        )

        root_half = np.sqrt(0.5)

        for corner, outward in (
            ((0.0, height), np.array([-root_half, root_half])),
            ((width, 0.0), np.array([root_half, -root_half])),
        ):
            obstacle = annulus_obstacle(
                np.asarray(corner, dtype=float) - corner_offset * outward,
                rng,
                min_vertices,
                max_vertices,
                min_radius,
                max_radius,
                polygons,
                course_region,
                spanning_axis=outward,
                containment_region=outside_region,
                raise_on_failure=False,
            )

            if obstacle is None:
                return finish(free_layout())

            polygons.append(obstacle)

        # ------------------------------------------------------------------
        # The two shortest ways past the goal obstacles
        # ------------------------------------------------------------------

        from .path_planner import PathPlanner

        # Solve the simplified problem using only the two goal obstacles,
        # so the segments describe the way past them and are not deflected
        # by the corner clusters.
        simplified_course = cls(
            width=width,
            height=height,
            obstacles=list(anchors),
        )
        simplified_planner = PathPlanner(
            obstacle_course=simplified_course,
            start=np.array([0.0, 0.0]),
            end=np.array([width, height]),
        )

        # The planner keys its graph by integer index into `coords`, so the
        # search runs on indices and the result is mapped back to
        # coordinates for bridge_segment, which compares against the
        # obstacles' own vertex tuples.
        planner_coords = simplified_planner.coords

        obstacle_a_vertices = set(anchors[0].exterior.coords[:-1])
        obstacle_b_vertices = set(anchors[1].exterior.coords[:-1])

        # Paths arrive in nondecreasing total visibility-graph length, so
        # the first usable one is the shortest path. After fixing its
        # bridge (a_i, b_j), continue through the same ordered list and
        # retain the first path whose bridge (a_k, b_l) satisfies both
        # a_k != a_i and b_l != b_j. The second selected path is therefore
        # the shortest one obeying the endpoint-disjointness requirement,
        # rather than merely one using a different A-to-B edge. That
        # ordering guarantee is the whole reason a k-shortest-paths routine
        # is needed here instead of a plain shortest path.
        #
        # scipy's yen() is not a lazy generator: K is fixed up front and it
        # returns at most K paths. The budget is therefore doubled until
        # either two acceptable bridges are found or the graph is
        # exhausted, which is signalled by fewer than K paths coming back.
        chosen_bridges = []
        budget = _INITIAL_PATH_BUDGET

        while True:
            candidate_paths = simplified_planner.k_shortest_paths(budget)

            if not candidate_paths:
                break

            chosen_bridges = []
            first_a_vertex = None
            first_b_vertex = None

            for path_indices in candidate_paths:
                path_vertices = [
                    tuple(planner_coords[index]) for index in path_indices
                ]
                bridge = bridge_segment(
                    path_vertices,
                    obstacle_a_vertices,
                    obstacle_b_vertices,
                )

                if bridge is None:
                    continue

                a_vertex, b_vertex = bridge

                if not chosen_bridges:
                    chosen_bridges.append(bridge)
                    first_a_vertex = a_vertex
                    first_b_vertex = b_vertex
                    continue

                if a_vertex == first_a_vertex or b_vertex == first_b_vertex:
                    continue

                chosen_bridges.append(bridge)
                break

            if len(chosen_bridges) >= 2:
                break

            # Fewer paths than asked for means the graph has no more, so a
            # larger budget cannot help.
            if len(candidate_paths) < budget:
                break

            if budget >= _MAXIMUM_PATH_BUDGET:
                break

            budget = min(2 * budget, _MAXIMUM_PATH_BUDGET)

        # Translating y by +/- radius gives the two placement lines either
        # side of each bridge.
        vertical_offsets = (
            np.array([0.0, -1.0 * radius]),
            np.array([0.0, 1.0 * radius]),
        )

        bridge_specs = []

        for bridge_start, bridge_end in chosen_bridges:
            bridge_start = np.asarray(bridge_start, dtype=float)
            bridge_end = np.asarray(bridge_end, dtype=float)

            shifted_intervals = []

            for vertical_offset in vertical_offsets:
                shifted_start = bridge_start + vertical_offset
                shifted_end = bridge_end + vertical_offset

                span = clip_segment(
                    shifted_start,
                    shifted_end,
                    inset_low_x,
                    inset_low_y,
                    inset_high_x,
                    inset_high_y,
                )

                if span is None:
                    break

                free_intervals = free_parameter_intervals(
                    shifted_start,
                    shifted_end,
                    span,
                    polygons,
                )

                if not free_intervals:
                    break

                shifted_intervals.append(free_intervals)

            if len(shifted_intervals) < 2:
                continue

            # Use a common interval on which either vertical projection is
            # a valid center. Shapely line-minus-polygon differences remove
            # portions lying inside an obstacle already placed. When
            # several components remain, retain the longest so the centers
            # still form one evenly spaced run.
            common_intervals = intersect_intervals(*shifted_intervals)

            if not common_intervals:
                continue

            common_low, common_high = max(
                common_intervals,
                key=lambda interval: interval[1] - interval[0],
            )

            usable_length = float(
                np.linalg.norm(bridge_end - bridge_start)
            ) * (common_high - common_low)

            count = diameter_fit_count(usable_length, diameter)

            if count == 0:
                continue

            bridge_specs.append((
                bridge_start,
                bridge_end,
                (common_low, common_high),
                count,
            ))

        bridge_count = sum(count for *_, count in bridge_specs)
        needed = len(polygons) + bridge_count

        if num_obstacles < needed:
            if raise_if_insufficient:
                raise ValueError(
                    f"num_obstacles is {num_obstacles} but this layout "
                    f"needs {needed}: {len(polygons)} for the goal and "
                    f"corner obstacles and {bridge_count} "
                    "along the two segments."
                )

            return finish(free_layout())

        # A locally valid early shape can still consume the space a later
        # one needs. Retry the whole segment layout, which redraws the
        # random phase of each evenly spaced run without changing any
        # count, so an unlucky alignment between the two runs is not
        # retried identically.
        fixed_polygon_count = len(polygons)

        for _layout_attempt in range(200):
            del polygons[fixed_polygon_count:]

            line_centers = [
                center
                for bridge_start, bridge_end, span, _count in bridge_specs
                for center in staggered_centers(
                    bridge_start,
                    bridge_end,
                    span,
                    diameter,
                    radius,
                    rng,
                )
            ]

            layout_succeeded = True

            for center_index, center in enumerate(line_centers):
                obstacle = annulus_obstacle(
                    center,
                    rng,
                    min_vertices,
                    max_vertices,
                    min_radius,
                    max_radius,
                    polygons,
                    course_region,
                    forbidden_points=line_centers[center_index + 1:],
                    max_attempts=800,
                    raise_on_failure=False,
                )

                if obstacle is None:
                    layout_succeeded = False
                    break

                polygons.append(obstacle)

            if layout_succeeded:
                break
        else:
            del polygons[fixed_polygon_count:]

        # ------------------------------------------------------------------
        # Surplus obstacles, placed wherever there is still room
        # ------------------------------------------------------------------

        place_random_obstacles(
            num_obstacles - len(polygons),
            rng,
            min_vertices,
            max_vertices,
            min_radius,
            max_radius,
            polygons,
            course_region,
            center_region,
        )

        report_shortfall()

        return finish(
            cls(
                width=width,
                height=height,
                obstacles=polygons,
            )
        )

    def vertices(self) -> np.ndarray:
        """``(n, 2)`` obstacle corners that a path could turn around.

        Two kinds of corner are dropped. One is buried in the interior of the
        union, which happens where obstacles meet and is not on any boundary
        a path could follow. The other lies outside the course: the obstacles
        covering ``(0, height)`` and ``(width, 0)`` are centred on those
        corners and so reach past the walls, and a point out there is not
        somewhere a path may turn. Corners exactly on a wall are kept, since
        a path is free to turn against one.

        Returned as an array rather than a list of tuples so the caller can
        index vertices by integer; the array itself is the index-to-coordinate
        map.
        """
        points = list(
            itertools.chain.from_iterable(
                map(
                    lambda obstacle: obstacle.exterior.coords[:-1],
                    self.obstacles,
                )
            )
        )

        if not points:
            return np.zeros((0, 2), dtype=float)

        points = np.asarray(points, dtype=float)[:, :2]

        x = points[:, 0]
        y = points[:, 1]

        # One batched call each, rather than a predicate per corner.
        keep = (
            (x >= 0.0)
            & (x <= self.width)
            & (y >= 0.0)
            & (y <= self.height)
            & ~shapely.contains_xy(self.obstacles_region, x, y)
        )

        return points[keep]

    def is_visible(self, vertex1, vertex2):
        """Are the segments ``vertex1[k] -> vertex2[k]`` free of every obstacle?

        Accepts a single pair of points or two ``(n, 2)`` arrays, and returns a
        ``bool`` or an ``(n,)`` array to match.

        Shapely 2's predicates are numpy ufuncs that broadcast one geometry
        against an array of them, so the whole batch crosses into GEOS in a
        single call. ``shapely.prepare`` was applied to the obstacle region at
        construction, which builds its index once instead of per query.
        """
        start = np.asarray(vertex1, dtype=float)
        end = np.asarray(vertex2, dtype=float)

        scalar = start.ndim == 1 and end.ndim == 1

        start = np.atleast_2d(start)
        end = np.atleast_2d(end)
        start, end = np.broadcast_arrays(start, end)

        if len(start) == 0:
            return np.zeros(0, dtype=bool)

        # One C call builds every segment, rather than one Python object at a
        # time.
        coordinates = np.empty((2 * len(start), 2), dtype=float)
        coordinates[0::2] = start
        coordinates[1::2] = end

        lines = shapely.linestrings(
            coordinates,
            indices=np.repeat(np.arange(len(start)), 2),
        )

        region = self.obstacles_region

        # "touches" keeps a path that runs along an obstacle boundary legal,
        # which is what a minimum-pressure path wants to do.
        visible = shapely.disjoint(region, lines) | shapely.touches(region, lines)

        return bool(visible[0]) if scalar else visible

    def plot(
        self,
        ax: plt.Axes = None,
        show_vertices: bool = True,
        show_labels: bool = False,
        title: str = None,
        show: bool = True,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the obstacle course."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.figure

        for index, obstacle in enumerate(self.obstacles or [], start=1):
            if obstacle is None or obstacle.is_empty:
                continue

            plot_polygon(
                obstacle,
                ax=ax,
                add_points=show_vertices,
                facecolor="tab:blue",
                edgecolor="black",
                linewidth=1.5,
                alpha=0.4,
            )

            if show_labels:
                # Unlike the centroid, representative_point() is guaranteed
                # to lie within the polygon.
                label_point = obstacle.representative_point()
                ax.text(
                    label_point.x,
                    label_point.y,
                    str(index),
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontweight="bold",
                )

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")#, zorder=1)
        ax.set_ylabel("y")#, zorder=1)
        ax.set_title(title or "Obstacle Course")
        ax.grid(alpha=0.25)

        if show:
            plt.show()

        return fig, ax

    def plot_path(
        self,
        start,
        end,
        distance_path = None,
        angle_path = None,
        pressure_path = None,
        ax: plt.Axes = None,
        show_vertices: bool = True,
        show_labels: bool = False,
        title: str = None,
        show: bool = True,
        show_legend: bool = True,
        legend_info=("length", "angle", "pressure"),
        course_units: str = "ft",
        length_units: str | None = None,
        pressure_units: str = "psi",
        angle_units: str = "degrees",
        model_parameters: dict | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the course, goals, and any computed paths.

        ``course_units`` states what one coordinate unit physically is. That
        is a property of the course and it is what the pressure model is
        evaluated in; ``PathPlanner.plot`` supplies it from the planner.

        ``length_units`` is purely how lengths are *displayed*, and defaults
        to ``course_units``. Setting it rescales the reported path lengths and
        the axis tick labels and nothing else: an 18 in course shown in feet
        reads 1.5 ft, and every pressure is unchanged because the course did
        not change size.

        ``pressure_units`` is the same kind of setting for the reported
        pressure: ``"psi"``, ``"Pa"``, ``"kPa"``, ``"MPa"`` or ``"bar"``. The
        model is psi throughout, so this only changes how the result is
        written down.

        Angles are displayed in degrees by default; pass
        ``angle_units="radians"`` to report them in radians.
        """
        model_parameters = dict(model_parameters or {})

        normalized_course_units = str(course_units).lower()
        if normalized_course_units not in _LENGTH_TO_FEET:
            raise ValueError(
                "course_units must be one of: 'ft', 'in', 'm', or 'cm'."
            )
        normalized_length_units = (
            normalized_course_units
            if length_units is None
            else str(length_units).lower()
        )

        pressure_label, pressure_scale = resolve_pressure_units(pressure_units)

        # Coordinate units -> display units. The geometry is left in
        # coordinate units and only the numbers are scaled, so nothing that
        # feeds the pressure model is touched.
        display_scale = _length_conversion(
            normalized_course_units,
            normalized_length_units,
        )

        normalized_angle_units = angle_units.lower()
        if normalized_angle_units in {"degree", "degrees", "deg"}:
            angle_scale = 180.0 / np.pi
            angle_label = "deg"
        elif normalized_angle_units in {"radian", "radians", "rad"}:
            angle_scale = 1.0
            angle_label = "rad"
        else:
            raise ValueError(
                "angle_units must be 'degrees' or 'radians'."
            )
        fig, ax = self.plot(
            ax=ax,
            show_vertices=show_vertices,
            show_labels=show_labels,
            title=title,
            show=False,
        )

        ax.scatter(
            start[0],
            start[1],
            s=80,
            marker="o",
            color="green",
            label="start",
            clip_on=False,
            zorder=5,
        )
        ax.scatter(
            end[0],
            end[1],
            s=120,
            marker="*",
            color="purple",
            label="target",
            clip_on=False,
            zorder=5,
        )

        # Drawn worst-first so the minimum-pressure path lands on top where
        # the three overlap.
        drawn = (
            (distance_path, "min distance", "tab:red", "-", 4),
            (angle_path, "min angle", "tab:orange", "--", 5),
            (pressure_path, "min pressure", "magenta", "-.", 6),
        )

        for path, name, color, linestyle, zorder in drawn:
            if path is None:
                continue

            path_coords = np.asarray(path.coords, dtype=float)
            path_length, path_angle_radians = path_metrics(path)

            # The recursive model depends on where each bend sits along
            # the path, not just the totals, so the pressure is evaluated
            # from the path geometry rather than from (length, angle).
            # The model is always evaluated in the units the course is
            # actually in; the display setting cannot reach it.
            pressure = path_pressure(
                path,
                length_units=normalized_course_units,
                **model_parameters,
            )

            ax.plot(
                path_coords[:, 0],
                path_coords[:, 1],
                linewidth=2.5,
                linestyle=linestyle,
                color=color,
                label=legend_label(
                    name,
                    legend_info,
                    length=path_length * display_scale,
                    angle=path_angle_radians * angle_scale,
                    pressure=pressure * pressure_scale,
                    length_units=normalized_length_units,
                    pressure_units=pressure_label,
                    angle_units=angle_label,
                ),
                zorder=zorder,
            )

        # The obstacles and paths stay in coordinate units -- rescaling the
        # geometry would mean rebuilding every polygon -- so the axes are
        # relabelled instead. Ticks are chosen to be round numbers in the
        # *display* units and then mapped back to coordinate positions;
        # formatting the default ticks instead would leave 25 in reading as
        # 2.08333 ft.
        if display_scale != 1.0:
            _relabel_axis(ax.xaxis, ax.get_xlim(), display_scale)
            _relabel_axis(ax.yaxis, ax.get_ylim(), display_scale)

        ax.set_xlabel(f"x ({normalized_length_units})")
        ax.set_ylabel(f"y ({normalized_length_units})")

        # Matplotlib gives a legend zorder=5 by default, which sits *below*
        # the goal markers and the minimum-pressure path. Anything drawn on
        # the axes has to stay under it, so the legend is lifted clear of
        # every zorder used above rather than nudged just past the current
        # maximum. An opaque frame is part of the same fix: at the default
        # framealpha the obstacles show through the box and the entries are
        # no easier to read for being on top.
        if show_legend:
            legend = ax.legend(framealpha=1.0)
            legend.set_zorder(_LEGEND_ZORDER)

        if show:
            plt.show()

        return fig, ax

    def to_tikz(self, start=None, end=None, *args, **kwargs) -> str:
        """PGFPlots/TikZ source for this course; see ``tikz.course_tikz``.

        Imported here rather than at module scope because the exporter needs
        this module's unit handling, and a top-level import would close the
        cycle.
        """
        from .tikz import course_tikz

        return course_tikz(self, start, end, *args, **kwargs)


# ---------------------------------------------------------------------------
# Voxel-lattice obstacle course
# ---------------------------------------------------------------------------
# Face indexing.  The four side faces run counterclockwise about +z starting
# at +x; the bottom and top come last:
#
#     0 -> +x        3 -> -y
#     1 -> +y        4 -> -z   (bottom)
#     2 -> -x        5 -> +z   (top)
#
# An obstacle is the tuple (i, j, k, f): voxel (i, j, k) of the lattice, face
# f of that voxel.  Voxel (i, j, k) occupies
#     [i*s, (i+1)*s] x [j*s, (j+1)*s] x [k*s, (k+1)*s],  s = voxel_size,
# and face f is that voxel's boundary square.
#
# Solid model
# -----------
# A wall is a *box*, not a quad: ``thickness`` deep along its own normal into
# the interior of the voxel that owns it, and grown by ``dilation`` in its own
# plane.  The dilation is what makes the model work.  Testing a segment
# against each wall separately is only sound when
#
#     int(union of the walls) == union of int(each wall),
#
# and with flat quads every interior is empty, so the identity fails
# everywhere: two walls that merely touch leave a seam that lies on the
# boundary of both, and a segment through it grazes each one and is reported
# clear by both.  Growing every wall by the same ``dilation`` turns every
# contact -- coplanar edge to edge, coplanar corner to corner, perpendicular
# along an edge, and perpendicular at a single point -- into an overlap with
# positive volume, which restores the identity.  A segment through any former
# seam now enters the interior of at least one box and is caught.
#
# Only the outermost walls of the course extend past its boundary, and only by
# ``dilation``; vertices that land outside are dropped, since nothing can
# reach them.

_FACE_AXIS = (0, 1, 0, 1, 2, 2)
_FACE_SIGN = (1, 1, -1, -1, -1, 1)
_FACE_NAMES = ("+x", "+y", "-x", "-y", "-z", "+z")

# Face code of the negative / positive face of a voxel, indexed by axis.
_NEGATIVE_FACE = (2, 3, 4)      # -x, -y, -z
_POSITIVE_FACE = (0, 1, 5)      # +x, +y, +z

# The two in-plane axes of a wall, indexed by the axis it is perpendicular to.
_AXIS_PAIRS = ((1, 2), (0, 2), (0, 1))


@define
class ObstacleCourseVoxels(VisibilityMixin):
    width: float = field(default=18.0, converter=float, validator=[validators.instance_of(float), validators.gt(0)])
    height: float = field(default=18.0, converter=float, validator=[validators.instance_of(float), validators.gt(0)])
    depth: float = field(default=36.0, converter=float, validator=[validators.instance_of(float), validators.gt(0)])

    voxel_size: float = field(default=6.0, converter=float, validator=[validators.instance_of(float), validators.gt(0)])

    # Depth of a wall along its own normal, and how far it is grown in its own
    # plane.  Both default to 1% of a voxel: large enough to sit far above any
    # floating point tolerance, small enough not to distort the lattice.
    thickness: float | None = field(default=None)
    dilation: float | None = field(default=None)

    # How the wall edges are sampled into graph vertices. These belong here
    # rather than as arguments to vertices(): they determine the model just as
    # thickness and dilation do, and fixing them at construction is what lets
    # the vertex set be computed once and cached.
    spacing: float | None = field(default=None)
    discretization: object | None = field(default=None)

    obstacles: list[tuple[int, int, int, int]] = field(factory=list)

    # Solid model, built once: (n, 3) lower and upper corners of every wall.
    box_low: np.ndarray = field(init=False, default=None)
    box_high: np.ndarray = field(init=False, default=None)

    # Filled on the first call to vertices().
    _vertices: np.ndarray | None = field(init=False, default=None)

    @voxel_size.validator
    def _voxel_valid(self, attribute, value):
        # A validator has to *raise*; returning False is silently ignored by
        # attrs.  Float modulo is also unsafe here (0.3 % 0.1 != 0), so the
        # divisibility check is done on the rounded voxel count instead.
        for name in ("width", "height", "depth"):
            extent = getattr(self, name)
            count = round(extent / value)
            if count < 1 or abs(count * value - extent) > 1e-9 * max(extent, value):
                raise ValueError(
                    f"{name}={extent} is not a whole number of voxels of "
                    f"size {value}."
                )

    def __attrs_post_init__(self):
        # Neighbouring voxels share a wall, so two different (voxel, face)
        # tuples can name the same slab. Rejecting that here makes uniqueness
        # a property of the class rather than of one constructor.
        keys = [self.wall_key(*obstacle) for obstacle in self.obstacles]

        if len(set(keys)) != len(keys):
            duplicates = {key for key in keys if keys.count(key) > 1}
            raise ValueError(
                f"obstacles contains {len(keys) - len(set(keys))} duplicate "
                f"wall(s); offending wall keys: {sorted(duplicates)}."
            )

        if self.thickness is None:
            self.thickness = 0.01 * self.voxel_size

        if self.dilation is None:
            self.dilation = 0.01 * self.voxel_size

        for name in ("thickness", "dilation"):
            value = float(getattr(self, name))
            if not 0.0 < value < 0.5 * self.voxel_size:
                raise ValueError(
                    f"{name}={value} must lie in (0, voxel_size/2); above that "
                    "walls a whole voxel apart would merge."
                )
            setattr(self, name, value)

        if self.spacing is None:
            self.spacing = 0.5 * self.voxel_size

        if self.spacing <= 0.0:
            raise ValueError(f"spacing must be positive, got {self.spacing}.")

        if self.discretization is None:
            self.discretization = self.uniform_discretization

        bounds = [self.wall_bounds(*obstacle) for obstacle in self.obstacles]

        self.box_low = (
            np.array([low for low, _ in bounds], dtype=float)
            if bounds else np.zeros((0, 3))
        )
        self.box_high = (
            np.array([high for _, high in bounds], dtype=float)
            if bounds else np.zeros((0, 3))
        )

    # ------------------------------------------------------------------ #
    # Solid model
    # ------------------------------------------------------------------ #
    def wall_bounds(self, i: int, j: int, k: int, face: int):
        """``(low, high)`` corners of the solid box for one blocked wall.

        The box depends on the *wall*, never on which of the two neighbouring
        voxels was used to name it: the face is resolved to its canonical
        representative first.  Without that, the same plane could be thickened
        in opposite directions by two differently written obstacles, and two
        coplanar neighbours would meet in a zero-volume seam again.
        """
        cell = self.wall_key(i, j, k, face)
        axis = cell[3]

        canonical = self._wall_to_voxel_face(self.shape, axis, cell[:3])
        face = canonical[3]

        size = self.voxel_size
        corner = np.array(canonical[:3], dtype=float) * size

        plane = corner[axis] + size if _FACE_SIGN[face] > 0 else corner[axis]

        low = np.empty(3, dtype=float)
        high = np.empty(3, dtype=float)

        # Grow along the normal *into* the voxel that owns the face.
        if _FACE_SIGN[face] > 0:
            low[axis], high[axis] = plane - self.thickness, plane
        else:
            low[axis], high[axis] = plane, plane + self.thickness

        for in_plane_axis in _AXIS_PAIRS[axis]:
            low[in_plane_axis] = corner[in_plane_axis] - self.dilation
            high[in_plane_axis] = corner[in_plane_axis] + size + self.dilation

        return low, high

    @property
    def extent(self) -> np.ndarray:
        return np.array((self.width, self.height, self.depth), dtype=float)

    def inside_walls(self, points: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        """``(n,)`` bool: is each point strictly inside some wall?

        Contact with a wall surface is not "inside": a path is allowed to hug
        a wall, so only points that have penetrated one are rejected.
        """
        points = np.atleast_2d(np.asarray(points, dtype=float))

        if len(self.box_low) == 0:
            return np.zeros(len(points), dtype=bool)

        return (
            np.all(points[:, None, :] > self.box_low[None] + tol, axis=2)
            & np.all(points[:, None, :] < self.box_high[None] - tol, axis=2)
        ).any(axis=1)

    # ------------------------------------------------------------------ #
    # Planner interface
    # ------------------------------------------------------------------ #
    @property
    def dimension(self) -> int:
        """Number of coordinates a point in this course carries."""
        return 3

    def generate_goals(self, start=None, end=None):
        """Fill in whichever of ``start``/``end`` was not supplied.

        Opposite corners of the course box: the origin and
        ``(width, height, depth)``. Both lie on the boundary of the box, and a
        wall only ever touches a corner rather than covering it, so neither
        can be sealed in.
        """
        corners = (np.zeros(3, dtype=float), self.extent.copy())

        resolved = []

        for point, default in zip((start, end), corners):
            if point is None:
                if not self.valid_goal(default):
                    raise RuntimeError(
                        f"The default goal {tuple(default)} is inside a wall; "
                        "pass start/end explicitly."
                    )
                resolved.append(default)
            else:
                resolved.append(np.asarray(point, dtype=float))

        return resolved[0], resolved[1]

    def valid_goal(self, coords, tol: float = 1e-9) -> bool:
        """Is ``coords`` a legal start or end point?

        Legal means inside the closed course box and not buried in a wall.
        A point resting on a wall surface stays legal, matching
        :meth:`is_visible`, which treats surface contact as a graze.
        """
        point = np.asarray(coords, dtype=float)

        if point.shape != (3,):
            raise ValueError(
                f"a goal must contain exactly three coordinates, got {point.shape}."
            )

        if np.any(point < -tol) or np.any(point > self.extent + tol):
            return False

        return not bool(self.inside_walls(point[None, :], tol)[0])

    @staticmethod
    def _box_edges() -> list[tuple[int, int]]:
        """The twelve edges of a box, as pairs of corner indices.

        Bit ``b`` of a corner index selects the high side of axis ``b``, so two
        corners are joined by an edge exactly when their indices differ in one
        bit.
        """
        return [
            (corner, corner | (1 << axis))
            for corner in range(8)
            for axis in range(3)
            if not corner & (1 << axis)
        ]

    def box_corners(self) -> np.ndarray:
        """``(n, 8, 3)`` corners of every wall box."""
        if len(self.box_low) == 0:
            return np.zeros((0, 8, 3))

        mask = ((np.arange(8)[:, None] >> np.arange(3)) & 1).astype(bool)

        return np.where(
            mask[None, :, :],
            self.box_high[:, None, :],
            self.box_low[:, None, :],
        )

    def uniform_discretization(
        self,
        start: np.ndarray,
        end: np.ndarray,
        spacing: float | None = None,
    ) -> np.ndarray:
        """Evenly spaced points along one edge, endpoints included.

        The count follows the edge's *length*, not the edge itself, so a long
        edge is sampled more finely than a short one and every sample sits
        roughly ``spacing`` apart. That matters here because a wall box is very
        anisotropic: its two in-plane edges span a voxel while the four along
        its normal span only ``thickness``, and a fixed count per edge would
        pile up redundant nodes across that sliver.

        ``spacing`` defaults to the course's own ``spacing``, so vary it at
        construction: ``ObstacleCourseVoxels(..., spacing=1.0)``.
        """
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)

        if spacing is None:
            spacing = self.spacing

        if spacing <= 0.0:
            raise ValueError(f"spacing must be positive, got {spacing}.")

        length = float(np.linalg.norm(end - start))

        # At least one interval, so the endpoints always come back.
        intervals = max(1, int(round(length / spacing)))

        fractions = np.linspace(0.0, 1.0, intervals + 1)[:, None]

        return start[None, :] + fractions * (end - start)[None, :]

    def vertices(self) -> np.ndarray:
        """``(n, 3)`` points that should become nodes of the visibility graph.

        The corners of the wall boxes plus samples along their edges, since an
        optimal path in a polyhedral scene bends on edges and not only at
        corners. How the edges are sampled is fixed by ``spacing`` and
        ``discretization`` at construction, which makes the result a pure
        function of the course and so safe to cache.

        A point is kept only when it lies inside the course and is not buried
        inside another wall. Burying matters because a node inside a wall would
        let a path cross that wall in two hops, each of which merely touches
        its surface and so grazes legally on its own.

        Returned as an array rather than a list of tuples so the caller can
        index vertices by integer; the array itself is the index-to-coordinate
        map.
        """
        if self._vertices is not None:
            return self._vertices

        if len(self.box_low) == 0:
            self._vertices = np.zeros((0, 3), dtype=float)
            return self._vertices

        corners = self.box_corners()

        samples = [corners.reshape(-1, 3)]

        for box in corners:
            for first, second in self._box_edges():
                samples.append(
                    np.asarray(
                        self.discretization(box[first], box[second]),
                        dtype=float,
                    )
                )

        points = np.vstack(samples)

        # Corners are shared by three edges of their own box, and neighbouring
        # boxes overlap, so the same coordinate is produced many times.
        _, keep = np.unique(np.round(points, 9), axis=0, return_index=True)
        points = points[np.sort(keep)]

        # Dilation pushes the outermost walls a little past the course; those
        # points are unreachable, so they are dropped rather than planned to.
        inside_course = np.all(points >= -1e-9, axis=1) & np.all(
            points <= self.extent + 1e-9, axis=1
        )

        self._vertices = points[inside_course & ~self.inside_walls(points)]
        return self._vertices

    def is_visible(self, vertex1, vertex2, tol: float = 1e-9):
        """Are the segments ``vertex1[k] -> vertex2[k]`` free of every wall?

        Accepts a single pair of points or two ``(n, 3)`` arrays, and returns a
        ``bool`` or an ``(n,)`` array to match.

        Each box is clipped independently, which the dilation makes sound (see
        the note at the top of the file). Writing the segment as
        ``x(t) = p + t (q - p)`` and the box as ``lo <= x <= hi``, the six face
        distances are ``x - hi`` and ``lo - x``, each affine in ``t``, so
        ``g(t) = max`` over the six is convex and piecewise linear. Clipping
        gives ``[t0, t1]``, the whole of the segment meeting the closed box,
        and ``g`` vanishes at both ends of it. Convexity then leaves exactly
        two possibilities: ``g`` is identically zero across the interval,
        meaning the segment only skims the surface, or strictly negative
        throughout, meaning it passes through. Probing the middle separates
        them, so one evaluation per box decides it. A skim stays visible on
        purpose: a path is allowed to hug a wall.

        The loop runs over *boxes* while vectorising over *segments*, rather
        than building one ``(segments, boxes, 3)`` array. That array is far
        larger than cache at any useful size and the memory traffic costs more
        than the arithmetic saves; keeping it ``(segments, 3)`` also lets each
        box narrow the surviving set for the next one.
        """
        start = np.asarray(vertex1, dtype=float)
        end = np.asarray(vertex2, dtype=float)

        scalar = start.ndim == 1 and end.ndim == 1

        start = np.atleast_2d(start)
        end = np.atleast_2d(end)
        start, end = np.broadcast_arrays(start, end)

        visible = np.ones(len(start), dtype=bool)

        if len(start) == 0 or len(self.box_low) == 0:
            return bool(visible[0]) if scalar else visible

        direction = end - start

        # Degenerate segments touch nothing.
        visible &= np.einsum("ij,ij->i", direction, direction) > 1e-24

        segment_low = np.minimum(start, end)
        segment_high = np.maximum(start, end)

        # Largest walls first: they eliminate the most segments, and every
        # later box then works on a smaller surviving set.
        order = np.argsort(-np.prod(self.box_high - self.box_low, axis=1))

        for box in order:
            low, high = self.box_low[box], self.box_high[box]

            alive = np.flatnonzero(visible)
            if alive.size == 0:
                break

            # Broad phase: an axis-aligned overlap test rejects most segments
            # for the cost of six comparisons.
            near = alive[
                np.all(segment_high[alive] >= low, axis=1)
                & np.all(segment_low[alive] <= high, axis=1)
            ]
            if near.size == 0:
                continue

            origin = start[near]
            step = direction[near]

            parallel = np.abs(step) <= 1e-12
            safe = np.where(parallel, 1.0, step)

            lower = (low - origin) / safe
            upper = (high - origin) / safe

            entering = np.minimum(lower, upper)
            exiting = np.maximum(lower, upper)

            outside = parallel & ((origin < low) | (origin > high))
            entering = np.where(parallel, np.where(outside, np.inf, 0.0), entering)
            exiting = np.where(parallel, np.where(outside, -np.inf, 1.0), exiting)

            first = np.maximum(entering.max(axis=1), 0.0)
            last = np.minimum(exiting.min(axis=1), 1.0)

            touched = np.flatnonzero(first <= last)
            if touched.size == 0:
                continue

            # Probe the middle of the contact interval. The depth is a real
            # distance, so `tol` keeps its geometric meaning.
            middle = (
                origin[touched]
                + (0.5 * (first[touched] + last[touched]))[:, None] * step[touched]
            )
            depth = np.maximum(middle - high, low - middle).max(axis=1)

            visible[near[touched[depth < -0.5 * tol]]] = False

        return bool(visible[0]) if scalar else visible

    # ------------------------------------------------------------------ #
    # Lattice geometry
    # ------------------------------------------------------------------ #
    @property
    def shape(self) -> tuple[int, int, int]:
        """Number of voxels along x, y and z."""
        return (
            round(self.width / self.voxel_size),
            round(self.height / self.voxel_size),
            round(self.depth / self.voxel_size),
        )

    @property
    def num_voxels(self) -> int:
        return int(np.prod(self.shape))

    @property
    def num_faces(self) -> int:
        """Total number of (voxel, face) slots, aliases included."""
        return 6 * self.num_voxels

    @property
    def num_walls(self) -> int:
        """Number of *distinct* walls, i.e. the sampling universe.

        A wall perpendicular to axis ``a`` sits on one of ``n_a + 1`` grid
        planes and spans one voxel along each of the other two axes, so the
        count is the sum over axes of ``(n_a + 1) * n_b * n_c``. This is
        strictly smaller than :attr:`num_faces` because every interior wall
        is shared by two voxels.
        """
        return sum(
            int(np.prod(shape))
            for shape in self._wall_shapes(self.shape)
        )

    @staticmethod
    def _wall_shapes(
        counts: tuple[int, int, int],
        interior_only: bool = False,
    ) -> list[tuple[int, int, int]]:
        """Index-space shape of the wall lattice, one entry per axis.

        Along the perpendicular axis the wall index runs over grid *planes*,
        of which there are ``n + 1`` (or ``n - 1`` if the two boundary planes
        are excluded); along the other two it runs over voxels.
        """
        shapes = []

        for axis in range(3):
            shape = list(counts)
            shape[axis] = counts[axis] - 1 if interior_only else counts[axis] + 1
            shapes.append(tuple(max(size, 0) for size in shape))

        return shapes

    @staticmethod
    def _wall_to_voxel_face(
        counts: tuple[int, int, int],
        axis: int,
        cell: tuple[int, int, int],
        interior_only: bool = False,
    ) -> tuple[int, int, int, int]:
        """Canonical ``(i, j, k, face)`` naming the wall ``(axis, cell)``.

        The wall is claimed by the voxel on its negative side, which exists
        for every plane except the far boundary; that one is claimed by the
        last voxel via its positive face. Exactly one representative per
        wall, so duplicates cannot arise.
        """
        voxel = list(cell)
        plane = voxel[axis] + (1 if interior_only else 0)

        if plane < counts[axis]:
            voxel[axis] = plane
            face = _NEGATIVE_FACE[axis]
        else:
            voxel[axis] = counts[axis] - 1
            face = _POSITIVE_FACE[axis]

        return (voxel[0], voxel[1], voxel[2], face)

    def _validate_face(self, i: int, j: int, k: int, face: int) -> None:
        if not 0 <= face < 6:
            raise ValueError(f"face must be in 0..5, got {face}.")

        for index, count, name in zip((i, j, k), self.shape, "xyz"):
            if not 0 <= index < count:
                raise ValueError(
                    f"voxel index {name}={index} outside 0..{count - 1}."
                )

    @staticmethod
    def face_normal(face: int) -> np.ndarray:
        """Unit outward normal of a face, as a ``(3,)`` array."""
        if not 0 <= face < 6:
            raise ValueError(f"face must be in 0..5, got {face}.")

        normal = np.zeros(3, dtype=float)
        normal[_FACE_AXIS[face]] = float(_FACE_SIGN[face])
        return normal

    def face_center(self, i: int, j: int, k: int, face: int) -> np.ndarray:
        """Centre of one blocked face, as a ``(3,)`` array.

        The square is flat: its centre lies exactly on the grid plane the two
        neighbouring voxels share, so the two voxels describing the same wall
        give the same point.
        """
        self._validate_face(i, j, k, face)

        size = self.voxel_size
        low = np.array((i, j, k), dtype=float) * size

        centre = low + 0.5 * size
        axis = _FACE_AXIS[face]
        centre[axis] = low[axis] + size if _FACE_SIGN[face] > 0 else low[axis]

        return centre

    def face_corners(self, i: int, j: int, k: int, face: int) -> np.ndarray:
        """The four corners of a blocked face, as a ``(4, 3)`` array.

        Ordered around the square, so consecutive rows (and the last back to
        the first) are its four edges.
        """
        self._validate_face(i, j, k, face)

        size = self.voxel_size
        low = np.array((i, j, k), dtype=float) * size

        axis = _FACE_AXIS[face]
        plane = low[axis] + size if _FACE_SIGN[face] > 0 else low[axis]

        first_axis, second_axis = _AXIS_PAIRS[axis]

        # Walk the square, so consecutive corners are joined by an edge.
        walk = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))

        corners = np.empty((4, 3), dtype=float)
        corners[:, axis] = plane
        corners[:, first_axis] = [low[first_axis] + u * size for u, _ in walk]
        corners[:, second_axis] = [low[second_axis] + v * size for _, v in walk]

        return corners

    def wall_key(self, i: int, j: int, k: int, face: int) -> tuple[int, int, int, int]:
        """Identifier of the *physical* wall a face refers to.

        Neighbouring voxels share a wall, so (0, 0, 0, 0) and (1, 0, 0, 2)
        describe the same slab.  Both map to the same key here.
        """
        axis = _FACE_AXIS[face]
        cell = [i, j, k]
        plane = cell[axis] + (1 if _FACE_SIGN[face] > 0 else 0)
        cell[axis] = plane
        return (cell[0], cell[1], cell[2], axis)

    # ------------------------------------------------------------------ #
    # Random construction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _default_voxel_size(width: float, height: float, depth: float) -> float:
        return min(width / 4.0, height / 4.0, depth / 4.0)

    @staticmethod
    def _voxel_counts(
        width: float,
        height: float,
        depth: float,
        voxel_size: float,
    ) -> tuple[int, int, int]:
        return tuple(
            round(extent / voxel_size)
            for extent in (width, height, depth)
        )

    @classmethod
    def random_faces(
        cls,
        width: float = 18.0,
        height: float = 18.0,
        depth: float = 36.0,
        voxel_size: float | None = None,
        spacing: float | None = None,
        discretization=None,
        min_faces: int = 1,
        max_faces: int | None = None,
        interior_only: bool = True,
        seed: int = None,
    ):
        """Block off between ``min_faces`` and ``max_faces`` walls.

        The count is drawn uniformly from that inclusive range, then the
        walls are drawn uniformly without replacement from the wall lattice.
        Sampling walls rather than (voxel, face) pairs is what makes the
        result duplicate free by construction: each wall appears exactly once
        in the index space, so no draw can name the same slab twice.

        ``interior_only`` drops the walls lying on the outer boundary of the
        course, which coincide with the container itself.
        """
        if voxel_size is None:
            voxel_size = cls._default_voxel_size(width, height, depth)

        counts = cls._voxel_counts(width, height, depth, voxel_size)
        shapes = cls._wall_shapes(counts, interior_only)
        sizes = [int(np.prod(shape)) for shape in shapes]
        total_walls = sum(sizes)

        if max_faces is None:
            max_faces = total_walls

        if not 0 <= min_faces <= max_faces <= total_walls:
            raise ValueError(
                f"Require 0 <= min_faces <= max_faces <= {total_walls}, got "
                f"min_faces={min_faces}, max_faces={max_faces}."
            )

        rng = np.random.default_rng(seed)
        number_of_walls = int(rng.integers(min_faces, max_faces + 1))

        # One flat draw over the concatenated per-axis wall blocks. Drawing
        # an axis first would over-weight whichever axis has fewer walls,
        # since the three blocks are not the same size on a non-cubic grid.
        flat = rng.choice(total_walls, size=number_of_walls, replace=False)
        flat.sort()

        obstacles = []
        offset = 0

        for axis, (shape, size) in enumerate(zip(shapes, sizes)):
            block = flat[(flat >= offset) & (flat < offset + size)] - offset
            offset += size

            if block.size == 0:
                continue

            for cell in zip(*np.unravel_index(block, shape)):
                obstacles.append(
                    cls._wall_to_voxel_face(
                        counts,
                        axis,
                        tuple(int(index) for index in cell),
                        interior_only,
                    )
                )

        return cls(
            width=width,
            height=height,
            depth=depth,
            voxel_size=voxel_size,
            spacing=spacing,
            discretization=discretization,
            obstacles=sorted(obstacles),
        )

    @classmethod
    def random(
        cls,
        width: float = 18.0,
        height: float = 18.0,
        depth: float = 36.0,
        voxel_size: float | None = None,
        spacing: float | None = None,
        discretization=None,
        fill_fraction: tuple[float, float] = (0.10, 0.3),
        interior_only: bool = True,
        seed: int = None,
    ):
        """Block off a random fraction of the walls.

        ``fill_fraction`` is a ``(low, high)`` pair of fractions of the total
        number of walls. Expressing the amount as a fraction rather than a
        count keeps the density of the course fixed as the lattice is
        refined, since the wall count grows as the cube of the resolution.
        """
        low_fraction, high_fraction = fill_fraction

        if not 0.0 <= low_fraction <= high_fraction <= 1.0:
            raise ValueError(
                "fill_fraction must be a (low, high) pair with "
                f"0 <= low <= high <= 1, got {fill_fraction}."
            )

        if voxel_size is None:
            voxel_size = cls._default_voxel_size(width, height, depth)

        counts = cls._voxel_counts(width, height, depth, voxel_size)
        total_walls = sum(
            int(np.prod(shape))
            for shape in cls._wall_shapes(counts, interior_only)
        )

        return cls.random_faces(
            width=width,
            height=height,
            depth=depth,
            voxel_size=voxel_size,
            spacing=spacing,
            discretization=discretization,
            min_faces=round(low_fraction * total_walls),
            max_faces=round(high_fraction * total_walls),
            interior_only=interior_only,
            seed=seed,
        )

    # ------------------------------------------------------------------ #
    def plot(
        self,
        plotter=None,
        color: str = "cornflowerblue",
        opacity: float = 1.0,
        show_edges: bool = True,
        edge_color: str = "black",
        show_boundary: bool = True,
        show_voxel_grid: bool = False,
        show_axes: bool = True,
        title: str = None,
        window_size: tuple[int, int] = (900, 700),
        jupyter_backend: str = "trame",
        show: bool = True,
    ):
        """Render the course with PyVista.

        Each wall is drawn as a flat ``pyvista.Plane``: one quad, so the only
        edges shown are the four real ones.

        ``jupyter_backend="trame"`` gives an interactive widget inside the
        notebook; pass ``"static"`` for a plain image (useful when exporting
        the notebook) or ``"html"`` for a self-contained scene.
        """
        import pyvista as pv

        own_plotter = plotter is None

        if own_plotter:
            plotter = pv.Plotter(notebook=True, window_size=window_size)

        for obstacle in self.obstacles:
            # Plane subdivides into i_resolution x j_resolution quads and
            # defaults to 10 x 10, which is what draws a grid across the face
            # once show_edges is on. One quad per wall leaves four edges.
            plane = pv.Plane(
                center=self.face_center(*obstacle),
                direction=self.face_normal(obstacle[3]),
                i_size=self.voxel_size,
                j_size=self.voxel_size,
                i_resolution=1,
                j_resolution=1,
            )
            plotter.add_mesh(
                plane,
                color=color,
                opacity=opacity,
                show_edges=show_edges,
                edge_color=edge_color,
                line_width=1.0,
                # A wall has no inside, so give the back face the same
                # material; otherwise half the course renders unlit.
                backface_params={"color": color, "opacity": opacity},
            )

        if show_voxel_grid:
            grid_type = getattr(pv, "ImageData", None) or pv.UniformGrid
            counts = self.shape
            grid = grid_type(
                dimensions=(counts[0] + 1, counts[1] + 1, counts[2] + 1),
                spacing=(self.voxel_size,) * 3,
                origin=(0.0, 0.0, 0.0),
            )
            plotter.add_mesh(
                grid,
                style="wireframe",
                color="gray",
                opacity=0.25,
                line_width=1.0,
            )

        if show_boundary:
            outline = pv.Box(
                bounds=(0.0, self.width, 0.0, self.height, 0.0, self.depth),
                level=0,
            ).outline()
            plotter.add_mesh(outline, color="black", line_width=3.0)

        if show_axes:
            plotter.show_axes()
            # The boundary outline already spans the full course, so the
            # ruler picks up the right extent from the scene bounds.
            plotter.show_bounds(
                xtitle="x",
                ytitle="y",
                ztitle="z",
                grid=False,
                location="outer",
            )

        if title:
            plotter.add_title(title)

        plotter.camera_position = "iso"

        if own_plotter:
            plotter.set_background("white")

        if show:
            plotter.show(jupyter_backend=jupyter_backend)

        return plotter

    # ------------------------------------------------------------------ #
    def plot_debug(
        self,
        plotter=None,
        color: str = "cornflowerblue",
        opacity: float = 0.55,
        show_edges: bool = True,
        edge_color: str = "black",
        show_vertices: bool = True,
        vertex_color: str = "crimson",
        vertex_size: float = 10.0,
        show_boundary: bool = True,
        show_voxel_grid: bool = False,
        show_axes: bool = True,
        title: str = None,
        window_size: tuple[int, int] = (900, 700),
        jupyter_backend: str = "trame",
        show: bool = True,
    ):
        """Render the course exactly as the planner sees it.

        :meth:`plot` draws each wall as a flat quad, which reads better but is
        a simplification: the planner works on solid, mutually overlapping
        boxes.  This method draws those boxes, so the thickness and the
        overlaps at every junction are visible, along with the graph vertices.
        Use it to check the solid model; use :meth:`plot` for figures.

        A semi transparent default is deliberate -- the overlaps are the point,
        and they are only visible through the faces.

        The points drawn are exactly the nodes the planner would build, since
        how the wall edges are sampled is fixed by the course's ``spacing`` and
        ``discretization`` rather than chosen per call.
        """
        import pyvista as pv

        own_plotter = plotter is None

        if own_plotter:
            plotter = pv.Plotter(notebook=True, window_size=window_size)

        for low, high in zip(self.box_low, self.box_high):
            # level=0 keeps the box a plain six-quad hexahedron, so the edges
            # drawn are the twelve real ones.
            box = pv.Box(
                bounds=(low[0], high[0], low[1], high[1], low[2], high[2]),
                level=0,
                quads=True,
            )
            plotter.add_mesh(
                box,
                color=color,
                opacity=opacity,
                show_edges=show_edges,
                edge_color=edge_color,
                line_width=1.0,
            )

        if show_vertices:
            # vertices() hands back an (n, 3) array, so test its length; the
            # truth value of an array is ambiguous.
            vertices = self.vertices()

            if len(vertices):
                plotter.add_mesh(
                    pv.PolyData(vertices),
                    color=vertex_color,
                    point_size=vertex_size,
                    render_points_as_spheres=True,
                )

        if show_voxel_grid:
            grid_type = getattr(pv, "ImageData", None) or pv.UniformGrid
            counts = self.shape
            grid = grid_type(
                dimensions=(counts[0] + 1, counts[1] + 1, counts[2] + 1),
                spacing=(self.voxel_size,) * 3,
                origin=(0.0, 0.0, 0.0),
            )
            plotter.add_mesh(
                grid,
                style="wireframe",
                color="gray",
                opacity=0.25,
                line_width=1.0,
            )

        if show_boundary:
            outline = pv.Box(
                bounds=(0.0, self.width, 0.0, self.height, 0.0, self.depth),
                level=0,
            ).outline()
            plotter.add_mesh(outline, color="black", line_width=3.0)

        if show_axes:
            plotter.show_axes()
            plotter.show_bounds(
                xtitle="x",
                ytitle="y",
                ztitle="z",
                grid=False,
                location="outer",
            )

        if title:
            plotter.add_title(title)

        plotter.camera_position = "iso"

        if own_plotter:
            plotter.set_background("white")

        if show:
            plotter.show(jupyter_backend=jupyter_backend)

        return plotter

    # ------------------------------------------------------------------ #
    def plot_path(
        self,
        start,
        end,
        distance_path=None,
        angle_path=None,
        pressure_path=None,
        plotter=None,
        color: str = "cornflowerblue",
        opacity: float = 1.0,
        show_edges: bool = True,
        edge_color: str = "black",
        show_boundary: bool = True,
        show_voxel_grid: bool = False,
        show_axes: bool = True,
        line_width: float = 6.0,
        marker_radius: float | None = None,
        title: str = None,
        window_size: tuple[int, int] = (900, 700),
        jupyter_backend: str = "trame",
        show: bool = True,
        course_units: str = "ft",
        length_units: str | None = None,
        pressure_units: str = "psi",
        angle_units: str = "degrees",
        model_parameters: dict | None = None,
    ):
        """Render the course, the goals, and any computed paths with PyVista.

        The course itself is drawn by :meth:`plot`, so the walls appear as flat
        quads; :meth:`plot_debug` shows the solid boxes the planner actually
        uses. Paths are drawn as tubes so they stay legible where they run
        along a wall surface.

        Legend entries are bare names only. Length, cumulative turning and
        pressure all belong to :meth:`PathPlanner.plot_pressure`; crowding them
        into this legend made it unreadable. ``course_units``,
        ``length_units``, ``pressure_units``, ``angle_units`` and
        ``model_parameters`` are accepted for signature parity with the
        polygonal course --
        ``PathPlanner.plot`` forwards them to whichever course it holds -- and
        are unused here because nothing in this view reports a length
        or a pressure.
        """
        import pyvista as pv

        plotter = self.plot(
            plotter=plotter,
            color=color,
            opacity=opacity,
            show_edges=show_edges,
            edge_color=edge_color,
            show_boundary=show_boundary,
            show_voxel_grid=show_voxel_grid,
            show_axes=show_axes,
            title=title,
            window_size=window_size,
            show=False,
        )

        if marker_radius is None:
            marker_radius = 0.12 * self.voxel_size

        for point, marker_color, label in (
            (start, "green", "start"),
            (end, "purple", "end"),
        ):
            if point is None:
                continue

            plotter.add_mesh(
                pv.Sphere(
                    radius=marker_radius,
                    center=np.asarray(point, dtype=float),
                ),
                color=marker_color,
                label=label,
            )

        for path, path_color, label in (
            (distance_path, "red", "min distance"),
            (angle_path, "orange", "min angle"),
            (pressure_path, "magenta", "min pressure"),
        ):
            if path is None:
                continue

            coordinates = np.asarray(path.coords, dtype=float)

            # MultipleLines needs at least two points; a degenerate path would
            # otherwise raise inside VTK.
            if len(coordinates) < 2:
                continue

            plotter.add_mesh(
                pv.MultipleLines(points=coordinates),
                color=path_color,
                line_width=line_width,
                label=label,
                render_lines_as_tubes=True,
            )

        plotter.add_legend(bcolor="white", face="line")

        if show:
            plotter.show(jupyter_backend=jupyter_backend)

        return plotter

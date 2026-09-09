import heapq

import matplotlib.pyplot as plt
import numpy as np

from typing import overload
from attrs import define, field
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as _csgraph_dijkstra

try:
    from scipy.sparse.csgraph import yen as _csgraph_yen
except ImportError as _error:  # pragma: no cover
    # yen() arrived in SciPy 1.14.0. It replaced this package's last
    # networkx dependency, so an older SciPy leaves no fallback.
    raise ImportError(
        "vine_planner requires SciPy >= 1.14.0 for "
        "scipy.sparse.csgraph.yen, which provides the k-shortest-paths "
        "search used by ObstacleCourse.random."
    ) from _error

from shapely import LineString

from .constants import (
    YIELD_PRESSURE_PSI,
    LENGTH_FRICTION_PSI_PER_FT,
    TAIL_TENSION_PSI,
    CURVATURE_FRICTION_COEFFICIENT,
    _LENGTH_TO_FEET,
)
from .obstacle_course import (
    ObstacleCourse,
    ObstacleCourseVoxels,
    _LEGEND_ZORDER,
    _length_conversion,
)
from .utils import (
    pressure_profile,
    path_metrics,
    path_pressure,
    legend_label,
    resolve_pressure_units,
)


# ---------------------------------------------------------------------------
# Why the line graph is implicit
# ---------------------------------------------------------------------------
#
# The line graph has 2m nodes (one per directed traversal of a visibility
# edge) and
#
#     sum_v deg(v) * (deg(v) - 1)
#
# arcs. That count is driven by the *second moment* of the degree
# distribution, so a handful of high-degree vertices in an open region cost
# far more than the average degree suggests. At n = 5,000 vertices with mean
# degree 50 it is roughly 12 million arcs; materialising them as an object
# graph with tuple keys and a per-arc attribute dict runs to several
# gigabytes, and the numpy intermediates needed to build it are another
# gigabyte on top.
#
# None of it has to exist. Storing the visibility graph in CSR over *directed*
# edges makes the successors of the line-graph node e = (u, v) exactly the
# contiguous slice
#
#     indices[indptr[v] : indptr[v + 1]]
#
# so a line-graph node is just an index into the CSR arrays and its successor
# set is a slice view. The whole structure is then the CSR itself plus three
# companion arrays of length 2m, on the order of ten megabytes at the size
# above.
#
# The multiplicative factor is what forces the issue rather than merely
# rewarding it. The affine update is
#
#     tau' = alpha * tau + beta,   alpha = exp(mu_c * theta),  beta = f * l_out
#
# and beta depends only on the outgoing edge, so it is a 2m array. alpha
# depends on the *arc*, so there are sum deg^2 of them and they cannot be
# stored at any useful size. They are computed one deg(v)-wide vector per pop
# instead, from precomputed unit direction vectors.
#
#
# Why one-pass Dijkstra is valid
# ---------------------------------------------------------------------------
#
# The augmented state (directed edge, accumulated tension) restores the Markov
# property that the raw vertex state loses, and the DP mapping H = alpha J +
# beta is monotone for alpha > 0. This is Bertsekas' affine monotonic model
# (Dynamic Programming and Optimal Control, section 4.5). With
#
#     alpha >= 1   and   beta >= 0
#
# labels are nondecreasing along every path, which is precisely the condition
# that makes label-setting sound: the minimum open label can never be improved
# by a longer prefix, so a settled node stays settled. Both hold here whenever
# mu_c >= 0 and f >= 0, and the constructor checks them rather than assuming.
#
# The minimum-turning planner is the same recursion with alpha == 1 and
# beta = theta, so it runs through the identical kernel. That is deliberate:
# the proxy-metric baseline differs from the proposed planner only in dropping
# the multiplicative coupling, and sharing the graph, the queue and the tie
# breaking removes any question of the baseline being handicapped by a
# separate implementation.


@define
class PathPlanner:
    obstacle_course: ObstacleCourse | ObstacleCourseVoxels

    start: np.ndarray | None = field(default=None)
    end: np.ndarray | None = field(default=None)

    # Units the course coordinates are expressed in. The friction term of the
    # pressure model is psi per foot, so the planner has to know this to build
    # its costs.
    #
    # This is a *model* parameter, not a display one: it sets the drag term
    # beta = f * l * scale, and therefore how heavily length is weighed
    # against bend amplification in the objective. Two planners differing only
    # in this field are solving different problems and can return different
    # paths. There is no default that is right for every course -- 'ft' is
    # only the least surprising one -- so set it to whatever the coordinates
    # actually mean.
    length_units: str = field(default="ft")

    yield_pressure: float = field(default=YIELD_PRESSURE_PSI)
    length_friction: float = field(default=LENGTH_FRICTION_PSI_PER_FT)
    tail_tension: float = field(default=TAIL_TENSION_PSI)
    curvature_coefficient: float = field(default=CURVATURE_FRICTION_COEFFICIENT)

    # Nodes are integers indexing into `coords`; the array is the only place
    # coordinates live. Keying by float tuples instead means every cost
    # calculation has to rebuild arrays out of them.
    coords: np.ndarray | None = field(init=False, default=None)
    start_index: int | None = field(init=False, default=None)
    end_index: int | None = field(init=False, default=None)

    # CSR of the visibility graph over *directed* edges. Every undirected
    # visibility edge appears twice, once in each orientation, and `indices`
    # is sorted within each row.
    indptr: np.ndarray | None = field(init=False, default=None)
    indices: np.ndarray | None = field(init=False, default=None)

    # Companion arrays, all indexed by directed edge (i.e. by line-graph
    # node). `reverse[e]` is the slot holding the opposite traversal, which
    # doubles as the U-turn successor to exclude and as a free lookup for the
    # tail vertex of e: source(e) == indices[reverse[e]].
    edge_length: np.ndarray | None = field(init=False, default=None)
    edge_direction: np.ndarray | None = field(init=False, default=None)
    edge_drag: np.ndarray | None = field(init=False, default=None)
    reverse: np.ndarray | None = field(init=False, default=None)

    distance_path: LineString | None = field(init=False, default=None)
    total_distance: float | None = field(init=False, default=None)

    angle_path: LineString | None = field(init=False, default=None)
    total_angle: float | None = field(init=False, default=None)

    pressure_path: LineString | None = field(init=False, default=None)
    total_pressure: float | None = field(init=False, default=None)

    def __attrs_post_init__(self):
        self._validate_model()
        self._goal_generator()
        self.create_graph()
        self.create_line_graph()

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    @start.validator
    @end.validator
    def _point_validator(self, attribute, value):
        # None means that the point will be generated automatically.
        if value is None:
            return

        value = np.asarray(value, dtype=float)

        # The course decides how many coordinates a point carries, so a new
        # course class needs no change here.
        dimension = self.obstacle_course.dimension

        if value.shape != (dimension,):
            raise ValueError(
                f"{attribute.name} must contain exactly {dimension} coordinates."
            )

        if not self.obstacle_course.valid_goal(value):
            raise ValueError(
                f"{attribute.name} is outside the obstacle course or inside an obstacle."
            )

    @length_units.validator
    def _length_units_validator(self, attribute, value):
        if str(value).lower() not in _LENGTH_TO_FEET:
            raise ValueError(
                "length_units must be one of: 'ft', 'in', 'm', or 'cm'."
            )

    def _validate_model(self) -> None:
        """Check the conditions the label-setting argument rests on.

        ``alpha >= 1`` and ``beta >= 0`` are what make the labels
        nondecreasing along a path, and hence what make one-pass Dijkstra
        correct rather than merely convenient. They are cheap to check and
        expensive to have silently violated, so they are checked.
        """
        if self.curvature_coefficient < 0.0:
            raise ValueError(
                "curvature_coefficient must be nonnegative: a negative value "
                "makes alpha < 1, so labels could decrease along a path and "
                "label-setting Dijkstra would no longer be valid."
            )

        if self.length_friction < 0.0:
            raise ValueError(
                "length_friction must be nonnegative: a negative value makes "
                "beta < 0 and breaks the affine monotonic model."
            )

        if self.tail_tension < 0.0:
            raise ValueError("tail_tension must be nonnegative.")

    def _goal_generator(self) -> None:
        """Fill in whichever of start/end was not supplied.

        Where the goals go depends entirely on the geometry, so each course
        class owns that choice and this is only the hand-off.
        """
        self.start, self.end = self.obstacle_course.generate_goals(
            self.start,
            self.end,
        )

    # ------------------------------------------------------------------ #
    # Derived quantities
    # ------------------------------------------------------------------ #

    def _resolve_display_units(self, length_units) -> tuple[str, float]:
        """Normalise a requested display unit and its conversion factor.

        Returns ``(units, scale)`` where ``scale`` takes a length in course
        coordinate units to ``units``.

        Display units and course units are different things and only the
        second one is physics. ``self.length_units`` says how big the course
        actually is, which fixes the drag term and therefore the objective.
        This says how to write the resulting numbers down, which fixes
        nothing at all -- an 18 in course reported in feet is the same course,
        the same robot and the same optimal path, printed as 1.5.
        """
        if length_units is None:
            return str(self.length_units).lower(), 1.0

        target = str(length_units).lower()

        return target, _length_conversion(self.length_units, target)

    @property
    def length_scale(self) -> float:
        """Feet per course length unit."""
        return _LENGTH_TO_FEET[str(self.length_units).lower()]

    @property
    def model_parameters(self) -> dict:
        """The pressure-model parameters, for forwarding to ``utils``."""
        return {
            "yield_pressure": self.yield_pressure,
            "length_friction": self.length_friction,
            "tail_tension": self.tail_tension,
            "curvature_coefficient": self.curvature_coefficient,
        }

    @property
    def vertex_count(self) -> int:
        return 0 if self.coords is None else len(self.coords)

    @property
    def edge_count(self) -> int:
        """Directed visibility edges, i.e. implicit line-graph nodes."""
        return 0 if self.indices is None else len(self.indices)

    @property
    def line_graph_arc_count(self) -> int:
        """``sum_v deg(v) * (deg(v) - 1)``, never materialised.

        Reported because it is the quantity the complexity analysis turns
        on, and because it is the number this rewrite exists to avoid
        allocating.
        """
        if self.indptr is None:
            return 0

        degree = np.diff(self.indptr).astype(np.int64)
        return int((degree * (degree - 1)).sum())

    # ------------------------------------------------------------------ #
    # Graph construction
    # ------------------------------------------------------------------ #

    def create_graph(self):
        """Build the visibility graph over the course vertices plus the goals.

        The graph is stored as a CSR adjacency over directed edges rather than
        as an object graph. That layout is what the implicit line graph needs:
        the outgoing edges of a vertex occupy one contiguous block, so the
        successors of a line-graph node are a slice rather than a lookup.
        """
        course = self.obstacle_course

        vertices = np.asarray(course.vertices(), dtype=float)
        start = np.asarray(self.start, dtype=float)
        end = np.asarray(self.end, dtype=float)

        coords = np.vstack([vertices, start[None, :], end[None, :]])

        # A goal may coincide with an obstacle vertex. Left in place that
        # creates two nodes at one point joined by a zero-length edge, which
        # has no turning angle and no direction vector.
        _, first_seen, inverse = np.unique(
            np.round(coords, 9),
            axis=0,
            return_index=True,
            return_inverse=True,
        )

        keep = np.sort(first_seen)
        position = np.empty(len(keep), dtype=np.int64)
        position[np.argsort(first_seen)] = np.arange(len(keep))

        self.coords = coords[keep]
        self.start_index = int(position[inverse[len(coords) - 2]])
        self.end_index = int(position[inverse[len(coords) - 1]])

        count = len(self.coords)

        first, second = course.visible_pairs(self.coords)
        first = np.asarray(first, dtype=np.int64)
        second = np.asarray(second, dtype=np.int64)

        # Each undirected edge as two directed edges. `pair` tags both halves
        # with the undirected edge they came from, which is what lets the
        # reverse permutation be recovered by a single sort below.
        sources = np.concatenate([first, second])
        targets = np.concatenate([second, first])
        pair = np.concatenate([
            np.arange(len(first), dtype=np.int64),
            np.arange(len(first), dtype=np.int64),
        ])

        # Sorting by (source, target) groups each vertex's outgoing edges into
        # a contiguous block *and* leaves `indices` sorted within the block.
        order = np.lexsort((targets, sources))
        sources = sources[order]
        targets = targets[order]
        pair = pair[order]

        degree = np.bincount(sources, minlength=count)

        self.indptr = np.concatenate(
            [[0], np.cumsum(degree)]
        ).astype(np.int32)
        self.indices = targets.astype(np.int32)

        vectors = self.coords[targets] - self.coords[sources]
        lengths = np.linalg.norm(vectors, axis=1)

        if len(lengths) and np.any(lengths <= 0.0):
            raise RuntimeError(
                "The visibility graph contains a zero-length edge; the "
                "coordinate deduplication in create_graph() should have "
                "prevented this."
            )

        self.edge_length = lengths

        # Unit directions turn every turning angle into a dot product between
        # two rows, so relaxing a line-graph node is one matrix-vector product
        # against a contiguous slice.
        self.edge_direction = (
            vectors / lengths[:, None]
            if len(lengths)
            else np.zeros((0, self.coords.shape[1]), dtype=float)
        )

        # beta of the affine update, in psi. Depends only on the outgoing
        # edge, so it is 2m values rather than one per arc.
        self.edge_drag = (
            self.length_friction * lengths * self.length_scale
        )

        # The two halves of each undirected edge sit adjacently once sorted by
        # their shared tag, so the permutation falls out without a search.
        self.reverse = np.empty(len(targets), dtype=np.int32)
        if len(targets):
            grouped = np.argsort(pair, kind="stable")
            left, right = grouped[0::2], grouped[1::2]
            self.reverse[left] = right
            self.reverse[right] = left

        return self.indptr, self.indices

    def create_line_graph(self):
        """Verify the implicit line graph is well formed.

        There is nothing to build: the line graph *is* the CSR, read as
        "node = directed edge, successors = the outgoing block of the head
        vertex minus the reversal". This method exists to keep the
        construction sequence explicit and to check the invariants the search
        relies on, since a silently malformed reverse permutation would
        produce plausible but wrong paths rather than an error.
        """
        if self.indices is None:
            raise RuntimeError("create_graph() must run before create_line_graph().")

        if self.edge_count:
            # r(r(e)) == e, and the two halves share endpoints in swapped
            # order.
            if not np.array_equal(self.reverse[self.reverse], np.arange(self.edge_count)):
                raise RuntimeError("The reverse permutation is not an involution.")

            tails = self.indices[self.reverse]
            heads = self.indices

            if not np.array_equal(self.indices[self.reverse[self.reverse]], heads):
                raise RuntimeError("The reverse permutation does not preserve heads.")

            if np.any(tails == heads):
                raise RuntimeError("The visibility graph contains a self-loop.")

        return self

    # ------------------------------------------------------------------ #
    # Adjacency helpers
    # ------------------------------------------------------------------ #

    def edge_sources(self) -> np.ndarray:
        """Tail vertex of every directed edge.

        Not stored: it is ``indices[reverse]``, and materialising it is only
        worth doing when the whole array is wanted at once.
        """
        return self.indices[self.reverse]

    def outgoing(self, vertex: int) -> slice:
        """The CSR block of directed edges leaving ``vertex``."""
        return slice(int(self.indptr[vertex]), int(self.indptr[vertex + 1]))

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #

    def _dijkstra_line_graph(
        self,
        objective: str,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Label-setting search over the implicit line graph.

        ``objective`` selects the affine update:

        ==============  ===========================  =====================
        objective       alpha                        beta
        ==============  ===========================  =====================
        ``"angle"``     ``1``                        ``theta`` (per arc)
        ``"pressure"``  ``exp(mu_c * theta)``        ``f * l_out``
        ==============  ===========================  =====================

        Returns the vertex-index path, the label at the terminal directed
        edge, and the full label array. The label array is the whole point of
        returning it: its sublevel sets are the reachable goal sets used by
        the comparison study, so the sweep is free once the search has run.

        The queue uses lazy deletion rather than decrease-key. That keeps the
        inner loop to array writes, at the cost of a heap that can in
        principle grow to the arc count; ``settled`` discards the stale pops.
        """
        if objective not in {"angle", "pressure"}:
            raise ValueError("objective must be 'angle' or 'pressure'.")

        pressure = objective == "pressure"

        edge_count = self.edge_count
        label = np.full(edge_count, np.inf, dtype=float)
        parent = np.full(edge_count, -1, dtype=np.int32)
        settled = np.zeros(edge_count, dtype=bool)

        if self.start_index == self.end_index:
            raise RuntimeError("The start and the goal are the same vertex.")

        block = self.outgoing(self.start_index)
        seeds = np.arange(block.start, block.stop, dtype=np.int64)

        if seeds.size == 0:
            raise RuntimeError(
                f"No collision-free {objective} path exists between start "
                "and end: the start vertex sees nothing."
            )

        # tau_0 = T, and the first segment's drag is charged before any bend.
        # For the turning objective the first segment is free.
        label[seeds] = (
            self.tail_tension + self.edge_drag[seeds]
            if pressure
            else 0.0
        )

        queue = [(float(label[e]), int(e)) for e in seeds]
        heapq.heapify(queue)

        mu = float(self.curvature_coefficient)
        terminal = -1

        while queue:
            cost, edge = heapq.heappop(queue)

            if settled[edge]:
                continue
            settled[edge] = True

            head = int(self.indices[edge])

            if head == self.end_index:
                terminal = edge
                break

            low = int(self.indptr[head])
            high = int(self.indptr[head + 1])

            if high <= low:
                continue

            # One matrix-vector product against a contiguous slice gives every
            # turning angle out of this vertex at once.
            cosines = self.edge_direction[low:high] @ self.edge_direction[edge]
            angles = np.arccos(np.clip(cosines, -1.0, 1.0))

            if pressure:
                candidate = cost * np.exp(mu * angles) + self.edge_drag[low:high]
            else:
                candidate = cost + angles

            slots = np.arange(low, high, dtype=np.int64)

            # `reverse[edge]` is the slot of the opposite traversal, which
            # lives in exactly this block. Excluding it is what forbids the
            # uncharged U-turn a vine robot cannot perform.
            improved = (
                (slots != int(self.reverse[edge]))
                & ~settled[low:high]
                & (candidate < label[low:high])
            )

            if not improved.any():
                continue

            targets = slots[improved]
            values = candidate[improved]

            label[targets] = values
            parent[targets] = edge

            for slot, value in zip(targets.tolist(), values.tolist()):
                heapq.heappush(queue, (value, slot))

        if terminal < 0:
            raise RuntimeError(
                f"No collision-free {objective} path exists between start "
                "and end."
            )

        chain = []
        edge = terminal
        while edge >= 0:
            chain.append(edge)
            edge = int(parent[edge])
        chain.reverse()

        # The walk is the tail of the first directed edge followed by the head
        # of each in turn.
        path_indices = [int(self.indices[self.reverse[chain[0]]])]
        path_indices.extend(int(self.indices[e]) for e in chain)

        if (
            path_indices[0] != self.start_index
            or path_indices[-1] != self.end_index
        ):
            raise RuntimeError(
                "The reconstructed path does not run from the start to the "
                "goal."
            )

        return np.asarray(path_indices, dtype=np.int64), float(label[terminal]), label

    def compute_distance_path(self) -> LineString:
        """Compute and store the minimum-distance visibility-graph path.

        Runs on the base graph, not the line graph. A shortest path under
        positive weights never contains a reversal, so forbidding U-turns
        would not change the answer, and searching the vertex state space is
        ``O(m log n)`` instead of ``O(sum deg^2)``.
        """
        distances, predecessors = _csgraph_dijkstra(
            self.to_scipy_sparse(),
            directed=True,
            indices=self.start_index,
            return_predecessors=True,
        )

        if not np.isfinite(distances[self.end_index]):
            raise RuntimeError(
                "No collision-free distance path exists between start and end."
            )

        path_indices = [self.end_index]
        while path_indices[-1] != self.start_index:
            path_indices.append(int(predecessors[path_indices[-1]]))
        path_indices.reverse()

        self.distance_path = LineString(self.coords[path_indices])
        self.total_distance = float(distances[self.end_index])
        return self.distance_path

    def compute_angle_path(self) -> LineString:
        """Compute and store the minimum-cumulative-turn path.

        This is the proxy-metric baseline: the same recursion as the pressure
        planner with the multiplicative factor set to one.
        """
        path_indices, total, _ = self._dijkstra_line_graph("angle")

        self.angle_path = LineString(self.coords[path_indices])
        self.total_angle = total
        return self.angle_path

    def compute_pressure_path(self, verify: bool = True) -> LineString:
        """Compute and store the globally minimum-growth-pressure path.

        The label carried through the search is

            M_i = tau_{i-1} + f * l_i,

        the tension accumulated through segment ``i`` but before the bend at
        its head is charged. That is the quantity the affine update acts on,
        and it makes the terminal case fall out rather than needing the
        phantom zero-angle bend the evaluator uses: the label at a directed
        edge arriving at the goal is already ``tau_{n-1} + f * l_n``, so

            P = Y + M_n.

        ``verify`` re-evaluates the returned geometry through
        ``utils.path_pressure`` and checks it against the search label. It is
        cheap, and it is the one assertion that catches an indexing error in
        the CSR arithmetic before it reaches a figure.
        """
        path_indices, tension, _ = self._dijkstra_line_graph("pressure")

        self.pressure_path = LineString(self.coords[path_indices])
        self.total_pressure = float(self.yield_pressure + tension)

        if verify:
            evaluated = path_pressure(
                self.pressure_path,
                length_units=self.length_units,
                **self.model_parameters,
            )

            if not np.isclose(evaluated, self.total_pressure, rtol=1e-9, atol=1e-12):
                raise RuntimeError(
                    "The search label and the evaluated path pressure "
                    f"disagree: {self.total_pressure!r} from the labelling, "
                    f"{evaluated!r} from utils.path_pressure. This means the "
                    "planner and the pressure model have drifted apart."
                )

        return self.pressure_path

    def compute_all_paths(self) -> dict[str, LineString]:
        """Run all three planners and return them by name."""
        return {
            "distance": self.compute_distance_path(),
            "angle": self.compute_angle_path(),
            "pressure": self.compute_pressure_path(),
        }

    def vertex_pressure_field(self) -> np.ndarray:
        """Minimum growth pressure required to reach each vertex, in psi.

        Taken from a single pressure search: the label at a directed edge is
        the optimal tension on arrival, so the value at a vertex is the
        minimum over its incoming edges. Vertices that cannot be reached carry
        ``inf``.

        This is the sublevel-set machinery the reachable-workspace study
        needs; rendering it as filled contours gives the pressure scalar field
        rather than a binary reachable region.
        """
        try:
            _, _, label = self._dijkstra_line_graph("pressure")
        except RuntimeError:
            label = np.full(self.edge_count, np.inf, dtype=float)

        field_values = np.full(self.vertex_count, np.inf, dtype=float)

        if self.edge_count:
            np.minimum.at(field_values, self.indices, label)

        field_values += self.yield_pressure
        field_values[self.start_index] = self.yield_pressure + self.tail_tension

        return field_values

    # ------------------------------------------------------------------ #
    # Sparse-matrix views
    # ------------------------------------------------------------------ #

    def to_scipy_sparse(self) -> csr_matrix:
        """The visibility graph as a SciPy CSR matrix of edge lengths.

        Built on demand and not cached: the arrays it wraps are already the
        planner's own, and holding a second reference to them buys nothing.
        Every undirected edge is present in both orientations, so
        ``directed=True`` is correct for the ``csgraph`` routines.
        """
        count = self.vertex_count

        return csr_matrix(
            (self.edge_length, self.indices, self.indptr),
            shape=(count, count),
        )

    def k_shortest_paths(
        self,
        count: int,
        *,
        source: int | None = None,
        target: int | None = None,
    ) -> list[np.ndarray]:
        """The ``count`` shortest loopless paths, by total length.

        Returns vertex-index arrays in nondecreasing total length. Fewer than
        ``count`` paths come back when the graph has no more to give, which is
        how a caller detects exhaustion rather than a truncated result.

        This wraps ``scipy.sparse.csgraph.yen``. Note that it is not a lazy
        stream: ``count`` is fixed up front, so a caller searching for a path
        with some property has to choose a budget and grow it.
        """
        if count < 1:
            raise ValueError("count must be at least 1.")

        source = self.start_index if source is None else int(source)
        target = self.end_index if target is None else int(target)

        distances, predecessors = _csgraph_yen(
            self.to_scipy_sparse(),
            source=source,
            sink=target,
            K=int(count),
            directed=True,
            return_predecessors=True,
        )

        paths = []

        for row in range(len(np.atleast_1d(distances))):
            if not np.isfinite(distances[row]):
                continue

            # scipy marks a vertex that the path does not visit with -9999,
            # including the source, so the walk terminates on the source
            # itself rather than on the sentinel.
            sequence = [target]
            node = target

            while node != source:
                node = int(predecessors[row, node])

                if node < 0:
                    sequence = None
                    break

                sequence.append(node)

            if sequence is None:
                continue

            sequence.reverse()
            paths.append(np.asarray(sequence, dtype=np.int64))

        return paths

    # ------------------------------------------------------------------ #
    # Plotting
    # ------------------------------------------------------------------ #

    def plot_pressure(
        self,
        ax: plt.Axes = None,
        *,
        length_units: str = None,
        angle_units: str = "degrees",
        pressure_units: str = "psi",
        points_per_segment: int = 50,
        title: str = None,
        show: bool = True,
        show_legend: bool = True,
        legend_info=("length", "angle", "pressure"),
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot required vine-robot pressure versus deployed path length.

        Any computed distance, angle and pressure paths are plotted on the
        same axes. Raises ``RuntimeError`` when none has been computed.

        ``length_units`` sets how arclength is *displayed* and defaults to the
        planner's course units. It is a pure rescaling of the x axis applied
        after the search: the pressure model is always evaluated in
        ``self.length_units``, so plotting an 18 in course in feet reports the
        same pressures against an axis reading 1.5 instead of 18. To state
        that the course is a different physical size, which does change the
        optimal path, construct a new planner with the intended
        ``length_units``: it is a model parameter that the drag term is baked
        into at graph construction, so assigning it afterwards leaves the
        planner inconsistent rather than restating the problem.

        ``pressure_units`` is the same kind of setting for the y axis:
        ``"psi"``, ``"Pa"``, ``"kPa"``, ``"MPa"`` or ``"bar"``. The model is
        psi throughout, so this rescales the plotted curve and its legend
        together and cannot alter the physics or the path that was chosen.

        Cumulative turning is reported in the legend for reference. It does
        not determine the pressure on its own: two paths with equal length and
        equal total turning can end at different pressures depending on where
        along the path the bends fall.
        """
        available = (
            self.distance_path,
            self.angle_path,
            self.pressure_path,
        )

        if all(path is None for path in available):
            raise RuntimeError(
                "No path has been computed. Call compute_distance_path(), "
                "compute_angle_path() or compute_pressure_path() before "
                "plot_pressure()."
            )

        normalized_length_units, display_scale = self._resolve_display_units(
            length_units
        )

        pressure_label, pressure_scale = resolve_pressure_units(pressure_units)

        normalized_angle_units = angle_units.lower()
        if normalized_angle_units in {"degree", "degrees", "deg"}:
            angle_scale = 180.0 / np.pi
            angle_label = "deg"
        elif normalized_angle_units in {"radian", "radians", "rad"}:
            angle_scale = 1.0
            angle_label = "rad"
        else:
            raise ValueError("angle_units must be 'degrees' or 'radians'.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        # Drawn worst-first so the minimum-pressure curve lands on top.
        styles = (
            (self.distance_path, "min distance", "tab:red", "-", None),
            (self.angle_path, "min angle", "tab:orange", "--", self.total_angle),
            (self.pressure_path, "min pressure", "magenta", "-.", None),
        )

        for path, name, color, linestyle, reported_angle in styles:
            if path is None:
                continue

            _, measured_angle_radians = path_metrics(path)
            angle_radians = (
                measured_angle_radians
                if reported_angle is None
                else reported_angle
            )

            # The profile is always computed in the units the course is
            # actually in, so the physics cannot be changed from here. Only
            # the returned arclengths are rescaled, for the axis.
            lengths, pressures = pressure_profile(
                path,
                length_units=self.length_units,
                points_per_segment=points_per_segment,
                **self.model_parameters,
            )
            lengths = lengths * display_scale
            pressures = pressures * pressure_scale

            ax.plot(
                lengths,
                pressures,
                linewidth=2.5,
                color=color,
                linestyle=linestyle,
                label=legend_label(
                    name,
                    legend_info,
                    length=float(lengths[-1]),
                    angle=angle_radians * angle_scale,
                    pressure=float(pressures[-1]),
                    length_units=normalized_length_units,
                    angle_units=angle_label,
                    pressure_units=pressure_label,
                    pressure_name="final pressure",
                ),
            )

        ax.set_xlabel(f"Path length ({normalized_length_units})")
        ax.set_ylabel(f"Pressure ({pressure_label})")
        ax.set_title(title or "Vine Robot Pressure vs. Path Length")
        ax.grid(alpha=0.25)

        # Matched to the course plot. Nothing here currently outranks the
        # default legend zorder, but the profiles are the obvious thing to
        # start layering, and a legend that sometimes hides is worse than one
        # that never does.
        if show_legend:
            legend = ax.legend(framealpha=1.0)
            legend.set_zorder(_LEGEND_ZORDER)

        if show:
            plt.show()

        return fig, ax

    def to_tikz(self, *args, **kwargs) -> str:
        """PGFPlots/TikZ source for the course plot.

        The counterpart of ``plot``: the goals, the computed paths, the course
        units and the model parameters are supplied from the planner, so the
        pressures quoted in the figure are the ones the search optimised.
        """
        from .tikz import course_tikz

        kwargs["course_units"] = self.length_units
        kwargs.setdefault("model_parameters", self.model_parameters)

        return course_tikz(
            self.obstacle_course,
            self.start,
            self.end,
            self.distance_path,
            self.angle_path,
            self.pressure_path,
            *args,
            **kwargs,
        )

    def pressure_to_tikz(self, *args, **kwargs) -> str:
        """PGFPlots/TikZ source for the pressure profile; see
        ``tikz.pressure_tikz``."""
        from .tikz import pressure_tikz

        return pressure_tikz(self, *args, **kwargs)

    @overload
    def plot(
        self,
        ax: plt.Axes = ...,
        show_vertices: bool = ...,
        show_labels: bool = ...,
        title: str = ...,
        show: bool = ...,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Polygonal course (``ObstacleCourse``): Matplotlib.

        Draws the obstacles, the start and end markers, and any computed paths
        on one set of 2D axes, and returns ``(figure, axes)``.
        """

    @overload
    def plot(
        self,
        plotter=...,
        color: str = ...,
        opacity: float = ...,
        show_edges: bool = ...,
        edge_color: str = ...,
        show_boundary: bool = ...,
        show_voxel_grid: bool = ...,
        show_axes: bool = ...,
        line_width: float = ...,
        marker_radius: float | None = ...,
        title: str = ...,
        window_size: tuple[int, int] = ...,
        jupyter_backend: str = ...,
        show: bool = ...,
    ):
        """Voxel course (``ObstacleCourseVoxels``): PyVista.

        Renders the walls, the start and end spheres, and any computed paths
        as tubes with a legend, and returns the ``pyvista.Plotter``.
        """

    def plot(self, *args, **kwargs):
        """Plot the course, the goals, and any computed paths.

        Every argument is forwarded to the course's ``plot_path``, so the
        parameters depend on which course is held; see the overloads above.
        Positional arguments continue from the parameter after the goals and
        paths, which this method supplies.

        The pressure model parameters and the course units are supplied from
        the planner, so the pressures quoted in the legend always come from
        the model the search optimised. A ``length_units`` argument, if given,
        is a display choice layered on top of that and is passed straight
        through.
        """
        kwargs["course_units"] = self.length_units
        kwargs.setdefault("model_parameters", self.model_parameters)

        return self.obstacle_course.plot_path(
            self.start,
            self.end,
            self.distance_path,
            self.angle_path,
            self.pressure_path,
            *args,
            **kwargs,
        )

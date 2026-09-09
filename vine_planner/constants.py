# ---------------------------------------------------------------------------
# Pressure units
# ---------------------------------------------------------------------------
#
# The pressure model is psi throughout: the calibrated parameters are psi and
# psi per foot, and `vine_robot_pressure` returns psi. Pressure units are a
# display concern only, exactly as `_LENGTH_TO_FEET` is for an axis label --
# converting a result cannot change the physics or the path that was chosen.
#
# Values are psi per unit, mirroring `_LENGTH_TO_FEET`. Keys are lower case
# because callers write "kpa" as often as "kPa"; the display spelling is
# recovered from `_PRESSURE_LABELS`, which is why the two tables have to keep
# the same key set.

# Exact by definition: 1 lbf = 4.4482216152605 N and 1 in = 0.0254 m, so
# 1 psi = 4.4482216152605 / 0.0254^2 Pa. Written out rather than as a rounded
# 6894.76 so that a round trip through pascals is exact in binary floating
# point.
_PSI_TO_PASCAL = 4.4482216152605 / (0.0254 ** 2)

_PRESSURE_TO_PSI = {
    "psi": 1.0,
    "pa": 1.0 / _PSI_TO_PASCAL,
    "kpa": 1.0e3 / _PSI_TO_PASCAL,
    "mpa": 1.0e6 / _PSI_TO_PASCAL,
    "bar": 1.0e5 / _PSI_TO_PASCAL,
}

_PRESSURE_LABELS = {
    "psi": "psi",
    "pa": "Pa",
    "kpa": "kPa",
    "mpa": "MPa",
    "bar": "bar",
}

YIELD_PRESSURE_PSI = 0.278
LENGTH_FRICTION_PSI_PER_FT = 0.0341
TAIL_TENSION_PSI = 0.787
CURVATURE_FRICTION_COEFFICIENT = 0.315

_LENGTH_TO_FEET = {
    "ft": 1.0,
    "in": 1.0 / 12.0,
    "m": 3.280839895013123,
    "cm": 0.03280839895013123,
}

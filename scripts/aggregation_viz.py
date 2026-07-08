"""
Visualization of the relate -> overlay(profile) -> sum aggregation process.

This module renders how point events that have been *extended* into short
linear events get aggregated onto a regularly segmented road. The picture is
driven by the real ``linref`` math so it always matches library behavior:

- Extended points and road segments are built as :class:`EventsData`.
- ``overlay(..., profile=...)`` produces the profile-weighted overlap matrix,
  where each weight is the integral of the event's value profile over the
  overlapping region (see ``linref.events.relate.overlay``).
- The per-segment totals are ``weights.T @ values`` -- exactly what
  ``EventsRelation.sum`` computes.

The function is fully parameterized so the same scene can be re-rendered for
different extension distances, segment lengths, and profile shapes.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from linref.events.base import EventsData
from linref.events.relate import overlay
from linref.events.profile import resolve_profile


def _as_dense(matrix) -> np.ndarray:
    """Coerce an overlay result (sparse or dense) to a 2D numpy array."""
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=float)


def _normalize_extend(extend) -> tuple[float, float]:
    """Return (left, right) extension distances from a scalar or 2-tuple."""
    if np.isscalar(extend):
        return float(extend), float(extend)
    left, right = extend
    return float(left), float(right)


def compute_aggregation(
    points,
    values=None,
    extend=0.25,
    segment_length=0.1,
    profile="triangular",
    road_extent=None,
):
    """
    Run the relate -> overlay(profile) -> sum aggregation.

    Parameters
    ----------
    points : array-like
        Point event locations (e.g., mileposts).
    values : array-like, optional
        Value carried by each point. Defaults to 1.0 for every point.
    extend : float or (float, float), default 0.25
        Distance each point is extended. A scalar extends both directions
        equally; a 2-tuple gives (left, right) distances.
    segment_length : float, default 0.1
        Length of each regular road segment.
    profile : str or EventProfile, default 'triangular'
        Value distribution profile ('uniform', 'triangular', 'parabolic',
        'trapezoidal', or an ``EventProfile`` instance).
    road_extent : (float, float), optional
        (min, max) extent of the segmented road. Defaults to a padded span
        covering every extended point.

    Returns
    -------
    dict
        Keys: ``points``, ``values``, ``ext_left``, ``ext_right``,
        ``begs``, ``ends`` (extended source events), ``seg_edges``,
        ``seg_begs``, ``seg_ends``, ``weights`` (points x segments),
        ``seg_values`` (aggregated per-segment totals), and ``profile``.
    """
    points = np.asarray(points, dtype=float)
    if values is None:
        values = np.ones_like(points)
    values = np.asarray(values, dtype=float)

    ext_l, ext_r = _normalize_extend(extend)
    begs = points - ext_l
    ends = points + ext_r

    # Determine the segmented road extent, snapped to the segment grid.
    if road_extent is None:
        pad = segment_length
        lo = np.floor((begs.min() - pad) / segment_length) * segment_length
        hi = np.ceil((ends.max() + pad) / segment_length) * segment_length
    else:
        lo, hi = road_extent
        lo = np.floor(lo / segment_length) * segment_length
        hi = np.ceil(hi / segment_length) * segment_length

    n_segments = int(round((hi - lo) / segment_length))
    seg_edges = lo + np.arange(n_segments + 1) * segment_length
    seg_begs = seg_edges[:-1]
    seg_ends = seg_edges[1:]

    # Build linref events and compute the profile-weighted overlap.
    left = EventsData(begs=begs, ends=ends)
    right = EventsData(begs=seg_begs, ends=seg_ends)
    weights = _as_dense(
        overlay(left, right, normalize=True, norm_by="left", profile=profile)
    )

    # Aggregate: this is exactly EventsRelation.sum(axis=0, method='overlay').
    seg_values = weights.T @ values

    return {
        "points": points,
        "values": values,
        "ext_left": ext_l,
        "ext_right": ext_r,
        "begs": begs,
        "ends": ends,
        "seg_edges": seg_edges,
        "seg_begs": seg_begs,
        "seg_ends": seg_ends,
        "weights": weights,
        "seg_values": seg_values,
        "profile": profile,
    }


def plot_aggregation(
    points,
    values=None,
    extend=0.25,
    segment_length=0.1,
    profile="triangular",
    road_extent=None,
    xlabel="Route location (mi)",
    title=None,
    save_path=None,
    figsize=(11, 6.5),
    point_colors=None,
):
    """
    Render the extend -> profile -> segment -> aggregate process.

    Produces three vertically stacked panels sharing an x-axis:

    1. Extended points with their value-density profile curves.
    2. The regularly segmented road.
    3. The aggregated per-segment value bars.

    See :func:`compute_aggregation` for the parameter definitions. Returns
    ``(fig, axes, result)`` where ``result`` is the dict from
    :func:`compute_aggregation`.
    """
    result = compute_aggregation(
        points=points,
        values=values,
        extend=extend,
        segment_length=segment_length,
        profile=profile,
        road_extent=road_extent,
    )
    points = result["points"]
    values = result["values"]
    begs = result["begs"]
    ends = result["ends"]
    seg_edges = result["seg_edges"]
    seg_begs = result["seg_begs"]
    seg_ends = result["seg_ends"]
    seg_values = result["seg_values"]

    prof = resolve_profile(profile)
    prof_name = type(prof).__name__.replace("Profile", "").lower()

    if point_colors is None:
        cmap = plt.get_cmap("tab10")
        point_colors = [cmap(i % 10) for i in range(len(points))]

    x_lo, x_hi = seg_edges[0], seg_edges[-1]

    fig, axes = plt.subplots(
        3, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 3], "hspace": 0.18},
    )
    ax_top, ax_mid, ax_bot = axes

    # ------------------------------------------------------------------ #
    # Panel 1: extended points and their value-density profiles.
    # ------------------------------------------------------------------ #
    ax_top.axhline(0, color="0.5", lw=1.2, zorder=1)
    t = np.linspace(0.0, 1.0, 400)
    peak_density = 0.0
    for i, (pt, val, beg, end, color) in enumerate(
        zip(points, values, begs, ends, point_colors)
    ):
        length = end - beg
        x = beg + t * length
        # Density (value per unit length): profile integrates to 1 over [0,1],
        # so dividing by physical length yields value/mi; area under = val.
        density = val * prof(t) / length
        peak_density = max(peak_density, density.max())
        ax_top.fill_between(x, 0, density, color=color, alpha=0.25, zorder=2)
        ax_top.plot(x, density, color=color, lw=1.8, zorder=3,
                    label=f"point {i + 1} @ {pt:g} (value={val:g})")
        # Extension span bracket just below the baseline.
        ax_top.annotate(
            "", xy=(beg, 0), xytext=(end, 0),
            arrowprops=dict(arrowstyle="<->", color=color, lw=1.2),
        )
        ax_top.plot([pt], [0], marker="v", color=color, ms=9,
                    markeredgecolor="black", zorder=4)
        ax_top.axvline(pt, color=color, ls=":", lw=1.0, alpha=0.6, zorder=1)

    ax_top.set_ylim(-0.12 * peak_density, 1.18 * peak_density)
    ax_top.set_ylabel("value density\n(value / mi)")
    ax_top.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_top.set_title(
        title
        or f"Extended points -> {prof_name} profile -> "
           f"{segment_length:g}-mi segment aggregation",
        fontsize=12, fontweight="bold",
    )

    # ------------------------------------------------------------------ #
    # Panel 2: the regularly segmented road.
    # ------------------------------------------------------------------ #
    ax_mid.set_ylim(0, 1)
    for j, (sb, se) in enumerate(zip(seg_begs, seg_ends)):
        shade = "0.85" if j % 2 == 0 else "0.70"
        ax_mid.add_patch(
            Rectangle((sb, 0), se - sb, 1, facecolor=shade,
                      edgecolor="white", lw=0.8, zorder=2)
        )
    for edge in seg_edges:
        ax_mid.plot([edge, edge], [0, 1], color="0.4", lw=0.6, zorder=3)
    ax_mid.set_yticks([])
    ax_mid.set_ylabel(
        f"road\n({segment_length:g} mi\nsegments)",
        rotation=0, ha="right", va="center", fontsize=9,
    )

    # ------------------------------------------------------------------ #
    # Panel 3: aggregated per-segment value bars.
    # ------------------------------------------------------------------ #
    centers = 0.5 * (seg_begs + seg_ends)
    bars = ax_bot.bar(
        centers, seg_values, width=segment_length * 0.94,
        color="#4C72B0", edgecolor="white", lw=0.6, zorder=3,
    )
    # Label non-trivial bars with their value.
    thresh = seg_values.max() * 0.02 if seg_values.max() > 0 else 0
    for rect, v in zip(bars, seg_values):
        if v > thresh:
            ax_bot.text(
                rect.get_x() + rect.get_width() / 2, v,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7,
            )
    ax_bot.set_ylabel("aggregated\nvalue / segment")
    ax_bot.set_xlabel(xlabel)
    ax_bot.set_ylim(0, max(seg_values.max() * 1.25, 1e-9))
    ax_bot.grid(axis="y", ls=":", alpha=0.4)

    # Conservation check annotation.
    total_in = values.sum()
    total_out = seg_values.sum()
    ax_bot.text(
        0.01, 0.95,
        f"input total = {total_in:g}   aggregated total = {total_out:.4f}",
        transform=ax_bot.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, lw=0.5),
    )

    for ax in axes:
        ax.set_xlim(x_lo, x_hi)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, axes, result


def main():
    """Render the two demonstration scenarios to PNG files."""
    # Scenario A: single point, triangular profile.
    plot_aggregation(
        points=[1.0],
        values=[1.0],
        extend=0.25,
        segment_length=0.1,
        profile="triangular",
        save_path="scripts/aggregation_single_triangular.png",
    )
    # Scenario B: two overlapping points summed.
    plot_aggregation(
        points=[1.0, 1.3],
        values=[1.0, 1.0],
        extend=0.25,
        segment_length=0.1,
        profile="triangular",
        save_path="scripts/aggregation_two_points_triangular.png",
    )
    print("Saved example figures to scripts/.")


if __name__ == "__main__":
    main()

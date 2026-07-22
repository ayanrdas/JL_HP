"""
Boundary edge roughness / effective mode number from closed (x, y) contours.

Outputs:
  - mode_number.csv / mode_number_vs_frame.png  (n_eff for every frame)
  - dr_vs_theta.csv + dr_vs_theta_evolution.gif
  - correlation_vs_theta.csv + correlation_vs_theta_evolution.gif
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation, PillowWriter


N_ANGLE = 360
CORR_THRESHOLD = 1.0 / np.e
DEFAULT_GIF_FPS = 5


def radius_vs_angle(boundary: np.ndarray, n_angle: int = N_ANGLE):
    cx, cy = boundary.mean(axis=0)
    dx = boundary[:, 0] - cx
    dy = boundary[:, 1] - cy
    angles = np.arctan2(dy, dx)
    radii = np.hypot(dx, dy)

    order = np.argsort(angles)
    angles = angles[order]
    radii = radii[order]

    angles, unique_idx = np.unique(angles, return_index=True)
    radii = radii[unique_idx]

    if angles.size < 8:
        raise ValueError("Too few unique boundary angles to measure roughness.")

    theta = np.linspace(-np.pi, np.pi, n_angle, endpoint=False)
    angles_ext = np.concatenate([angles - 2 * np.pi, angles, angles + 2 * np.pi])
    radii_ext = np.concatenate([radii, radii, radii])
    radius = np.interp(theta, angles_ext, radii_ext)

    return theta, radius, np.array([cx, cy])


def first_zero_crossing_lag(C: np.ndarray) -> float:
    """Angular lag (rad) where the autocorrelation first crosses zero."""
    n = len(C)
    half = n // 2
    dtheta = 2.0 * np.pi / n

    for k in range(1, half):
        if C[k] <= 0.0:
            prev = C[k - 1]
            denom = prev - C[k]
            frac = prev / denom if denom > 1e-15 else 0.0
            return float((k - 1 + frac) * dtheta)

    return np.nan


def mode_number_from_lag(zero_lag_rad: float) -> float:
    """n_eff = pi / (2 * dtheta_0)."""
    if not np.isfinite(zero_lag_rad) or zero_lag_rad <= 0:
        return np.nan
    return float(np.pi / (2.0 * zero_lag_rad))


def roughness_from_radius(radius: np.ndarray):
    dr = radius - np.mean(radius)
    sigma = float(np.sqrt(np.mean(dr ** 2)))

    if sigma < 1e-12:
        return sigma, np.ones(len(dr)), np.nan

    F = np.fft.fft(dr)
    corr = np.fft.ifft(F * np.conj(F)).real
    C = corr / corr[0]

    half = len(C) // 2
    corr_length_lag = np.nan
    for k in range(1, half):
        if C[k] <= CORR_THRESHOLD:
            corr_length_lag = float(k)
            break

    return sigma, C, corr_length_lag


def analyze_boundary(boundary: np.ndarray, n_angle: int = N_ANGLE) -> dict:
    """Analyze one closed boundary; returns dict including mode_number (n_eff)."""
    theta, radius, center = radius_vs_angle(boundary, n_angle=n_angle)
    sigma, C, lag = roughness_from_radius(radius)

    dtheta = 2.0 * np.pi / n_angle
    mean_r = float(np.mean(radius))
    corr_length_rad = float(lag * dtheta) if np.isfinite(lag) else np.nan
    corr_length_arc_local = (
        float(lag * mean_r * dtheta) if np.isfinite(lag) else np.nan
    )
    zero_lag = first_zero_crossing_lag(C)

    return {
        "theta": theta,
        "radius": radius,
        "center": center,
        "mean_radius": mean_r,
        "sigma": sigma,
        "relative_roughness": sigma / mean_r if mean_r > 0 else np.nan,
        "C": C,
        "corr_length_lag": lag,
        "corr_length_rad": corr_length_rad,
        "corr_length_arc_px_local": corr_length_arc_local,
        "zero_crossing_rad": zero_lag,
        "mode_number": mode_number_from_lag(zero_lag),
        "dr": radius - mean_r,
    }


def n_eff_from_boundary(boundary: np.ndarray, n_angle: int = N_ANGLE) -> float:
    """Convenience: return n_eff for one boundary, or NaN on failure."""
    try:
        return float(analyze_boundary(boundary, n_angle=n_angle)["mode_number"])
    except Exception:
        return float(np.nan)


def write_mode_number_csv(
    frames: Sequence[int],
    mode_numbers: Sequence[float],
    save_path: str | Path,
) -> Path:
    """CSV with columns: frame, n_eff."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([
        np.asarray(frames, dtype=float),
        np.asarray(mode_numbers, dtype=float),
    ])
    np.savetxt(
        save_path,
        data,
        delimiter=",",
        header="frame,n_eff",
        comments="",
    )
    return save_path


def plot_mode_number(
    frames: Sequence[int],
    mode_numbers: Sequence[float],
    save_path: str | Path,
) -> Path:
    """n_eff = pi / (2 * dtheta_0) vs frame."""
    save_path = Path(save_path)
    frames_arr = np.asarray(frames, dtype=float)
    mode_arr = np.asarray(mode_numbers, dtype=float)
    finite = np.isfinite(mode_arr)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(frames_arr, mode_arr, "-o", ms=3, color="seagreen")

    n_missing = int((~finite).sum())
    if n_missing:
        ax.plot([], [], " ", label=f"{n_missing} frame(s) with no zero crossing")
        ax.legend(frameon=False, fontsize=8)

    ax.set_xlabel("frame")
    ax.set_ylabel(r"$n_{\mathrm{eff}} = \pi / (2\,\Delta\theta_0)$")
    ax.set_title("Effective mode number vs frame")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def save_mode_number_outputs(
    frames: Sequence[int],
    mode_numbers: Sequence[float],
    output_dir: str | Path,
) -> Tuple[Path, Path]:
    """Write n_eff CSV and plot into the channel output directory."""
    output_dir = Path(output_dir)
    csv_path = write_mode_number_csv(
        frames, mode_numbers, output_dir / "mode_number.csv"
    )
    png_path = plot_mode_number(
        frames, mode_numbers, output_dir / "mode_number_vs_frame.png"
    )
    return csv_path, png_path


def write_dr_vs_theta_csv(
    frame_results: Sequence[Tuple[int, Optional[dict]]],
    save_path: str | Path,
) -> Path:
    """Long-format CSV: frame, theta_rad, dr_px (all frames)."""
    save_path = Path(save_path)
    rows = []
    for frame, result in frame_results:
        if result is None:
            continue
        theta = result["theta"]
        dr = result["dr"]
        for th, d in zip(theta, dr):
            rows.append((frame, th, d))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        np.savetxt(
            save_path,
            np.asarray(rows, dtype=float),
            delimiter=",",
            header="frame,theta_rad,dr_px",
            comments="",
        )
    else:
        np.savetxt(
            save_path,
            np.empty((0, 3)),
            delimiter=",",
            header="frame,theta_rad,dr_px",
            comments="",
        )
    return save_path


def write_correlation_vs_theta_csv(
    frame_results: Sequence[Tuple[int, Optional[dict]]],
    save_path: str | Path,
) -> Path:
    """Long-format CSV: frame, lag_rad, C (first half of autocorrelation)."""
    save_path = Path(save_path)
    rows = []
    for frame, result in frame_results:
        if result is None:
            continue
        C = result["C"]
        half = len(C) // 2
        lag_rad = np.arange(half) * (2.0 * np.pi / len(C))
        for lag, cval in zip(lag_rad, C[:half]):
            rows.append((frame, lag, cval))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        np.savetxt(
            save_path,
            np.asarray(rows, dtype=float),
            delimiter=",",
            header="frame,lag_rad,C",
            comments="",
        )
    else:
        np.savetxt(
            save_path,
            np.empty((0, 3)),
            delimiter=",",
            header="frame,lag_rad,C",
            comments="",
        )
    return save_path


def _valid_results(
    frame_results: Sequence[Tuple[int, Optional[dict]]],
) -> List[Tuple[int, dict]]:
    return [(f, r) for f, r in frame_results if r is not None]


def write_dr_vs_theta_gif(
    frame_results: Sequence[Tuple[int, Optional[dict]]],
    save_path: str | Path,
    fps: int = DEFAULT_GIF_FPS,
) -> Optional[Path]:
    """GIF of dr(theta) evolving over time."""
    valid = _valid_results(frame_results)
    if len(valid) < 1:
        return None

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    dr_max = max(float(np.max(np.abs(r["dr"]))) for _, r in valid)
    if dr_max < 1e-6:
        dr_max = 1.0

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    artists = []
    for frame, result in valid:
        (line,) = ax.plot(
            result["theta"], result["dr"], color="steelblue", lw=1.2, animated=True
        )
        title = ax.text(
            0.5,
            1.02,
            f"dr vs theta  |  frame {frame}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            animated=True,
        )
        zero = ax.axhline(0.0, color="0.5", lw=0.8, animated=True)
        artists.append([line, zero, title])

    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-1.05 * dr_max, 1.05 * dr_max)
    ax.set_xlabel("theta (rad)")
    ax.set_ylabel("dr (px)")
    ax.set_title("Radial fluctuation vs theta")

    ani = ArtistAnimation(fig, artists, interval=1000 // max(fps, 1), blit=True)
    ani.save(str(save_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    return save_path


def write_correlation_vs_theta_gif(
    frame_results: Sequence[Tuple[int, Optional[dict]]],
    save_path: str | Path,
    fps: int = DEFAULT_GIF_FPS,
) -> Optional[Path]:
    """GIF of C(dtheta) evolving over time."""
    valid = _valid_results(frame_results)
    if len(valid) < 1:
        return None

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(valid[0][1]["C"])
    half = n // 2
    lag_max = (half - 1) * (2.0 * np.pi / n)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    artists = []
    for frame, result in valid:
        C = result["C"]
        lag_rad = np.arange(half) * (2.0 * np.pi / len(C))
        (line,) = ax.plot(
            lag_rad, C[:half], color="darkorange", lw=1.5, animated=True
        )
        title = ax.text(
            0.5,
            1.02,
            f"C(dtheta) vs lag  |  frame {frame}  |  n_eff={result['mode_number']:.2f}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            animated=True,
        )
        zero = ax.axhline(0.0, color="0.5", lw=0.8, animated=True)
        thresh = ax.axhline(
            CORR_THRESHOLD, color="0.5", ls="--", lw=0.9, animated=True
        )
        artists.append([line, zero, thresh, title])

    ax.set_xlim(0.0, lag_max)
    ax.set_ylim(-1.3, 1.05)
    ax.set_xlabel("angular lag dtheta (rad)")
    ax.set_ylabel("C(dtheta)")
    ax.set_title("dr autocorrelation vs angular lag")

    ani = ArtistAnimation(fig, artists, interval=1000 // max(fps, 1), blit=True)
    ani.save(str(save_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    return save_path


def save_roughness_time_series_outputs(
    frame_results: Sequence[Tuple[int, Optional[dict]]],
    output_dir: str | Path,
    fps: int = DEFAULT_GIF_FPS,
) -> dict:
    """
    Save n_eff (all frames), dr/C CSVs, and evolution GIFs into output_dir.
    frame_results: list of (frame_index, analyze_boundary dict or None)
    """
    output_dir = Path(output_dir)
    frames = [f for f, _ in frame_results]
    mode_numbers = [
        (r["mode_number"] if r is not None else np.nan) for _, r in frame_results
    ]

    csv_neff, png_neff = save_mode_number_outputs(frames, mode_numbers, output_dir)
    csv_dr = write_dr_vs_theta_csv(frame_results, output_dir / "dr_vs_theta.csv")
    csv_c = write_correlation_vs_theta_csv(
        frame_results, output_dir / "correlation_vs_theta.csv"
    )
    gif_dr = write_dr_vs_theta_gif(
        frame_results, output_dir / "dr_vs_theta_evolution.gif", fps=fps
    )
    gif_c = write_correlation_vs_theta_gif(
        frame_results, output_dir / "correlation_vs_theta_evolution.gif", fps=fps
    )

    return {
        "mode_number_csv": csv_neff,
        "mode_number_png": png_neff,
        "dr_csv": csv_dr,
        "correlation_csv": csv_c,
        "dr_gif": gif_dr,
        "correlation_gif": gif_c,
    }

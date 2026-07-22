"""Bright-particle edge detection on raw grayscale frames."""

import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import binary_closing, disk, remove_small_objects


def normalize_frame(frame):
    """Normalize a grayscale frame to the range [0, 1]."""
    frame_float = frame.astype(np.float32)
    image_min = frame_float.min()
    image_max = frame_float.max()
    return (frame_float - image_min) / (image_max - image_min + 1e-8)


def select_particle_component(labels, smooth):
    """
    Keep the main bright particle.

    Small bright blobs in the timestamp corner are ignored so burned-in
    text like "t=..." does not get picked as the particle.
    """
    properties = regionprops(labels, intensity_image=smooth)

    if len(properties) == 0:
        raise RuntimeError("No particle was detected.")

    height, width = smooth.shape
    best_label = None
    best_score = -np.inf

    for region in properties:
        cy, cx = region.centroid
        mean_intensity = float(region.mean_intensity)
        score = region.area * (1.0 + mean_intensity)

        in_timestamp_corner = cy > 0.85 * height and cx > 0.70 * width
        if region.area < 150 and in_timestamp_corner:
            score *= 0.05

        if score > best_score:
            best_score = score
            best_label = region.label

    return labels == best_label


def bright_particle_mask(smooth):
    """
    Segment the bright particle against a dark background.

    Uses Otsu on the non-background intensities so the threshold locks
    onto the bright rim instead of the faint outer halo.
    """
    positive = smooth[smooth > 0.02]

    if positive.size >= 20:
        threshold = threshold_otsu(positive)
    else:
        threshold = threshold_otsu(smooth)

    threshold = float(np.clip(0.65 * threshold, 0.08, 0.50))

    mask = smooth >= threshold

    rough_area = int(mask.sum())
    close_radius = 1 if rough_area < 500 else (2 if rough_area < 6000 else 3)
    mask = binary_closing(mask, footprint=disk(close_radius))
    mask = ndimage.binary_fill_holes(mask)

    min_size = 5 if rough_area < 200 else 25
    mask = remove_small_objects(mask.astype(bool), min_size=min_size)

    labels = label(mask)
    mask = select_particle_component(labels, smooth)

    return mask, threshold


def smooth_boundary_from_mask(mask, center=None):
    """
    Build a smooth closed (x, y) boundary around the bright mask.

    1. Soften the binary mask and take the 0.5 iso-contour.
    2. Reparameterize in polar coordinates around the centroid.
    3. Smooth the radius periodically for a clean closed curve.
    """
    area = float(np.sum(mask))

    if area == 0:
        raise RuntimeError("The particle mask is empty.")

    if center is None:
        cy, cx = ndimage.center_of_mass(mask)
    else:
        cx, cy = center

    equivalent_radius = np.sqrt(area / np.pi)
    mask_sigma = float(np.clip(0.04 * equivalent_radius, 0.5, 1.6))

    smooth_mask = gaussian_filter(mask.astype(np.float32), sigma=mask_sigma)
    contours = find_contours(smooth_mask, level=0.5)

    if len(contours) == 0:
        raise RuntimeError("No particle boundary was found.")

    def contour_score(contour):
        mean_y, mean_x = contour.mean(axis=0)
        distance_penalty = (mean_x - cx) ** 2 + (mean_y - cy) ** 2
        return len(contour) - 0.05 * distance_penalty

    contour = max(contours, key=contour_score)
    points = np.column_stack((contour[:, 1], contour[:, 0]))

    height, width = mask.shape
    center_near_border = (
        cx < 8 or cy < 8 or cx > width - 9 or cy > height - 9
    )

    if equivalent_radius < 8 or center_near_border or len(points) < 20:
        return points

    angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
    radii = np.hypot(points[:, 0] - cx, points[:, 1] - cy)

    order = np.argsort(angles)
    angles = angles[order]
    radii = radii[order]

    unique_angles, unique_index = np.unique(angles, return_index=True)
    unique_radii = radii[unique_index]

    if unique_angles.size < 8:
        return points

    angle_grid = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    angle_extended = np.concatenate([
        unique_angles - 2.0 * np.pi,
        unique_angles,
        unique_angles + 2.0 * np.pi,
    ])
    radius_extended = np.concatenate([
        unique_radii,
        unique_radii,
        unique_radii,
    ])

    radius_grid = np.interp(angle_grid, angle_extended, radius_extended)
    radius_sigma = float(
        np.clip(0.012 * 360.0 * (equivalent_radius / 25.0), 1.2, 4.5)
    )
    radius_grid = gaussian_filter1d(radius_grid, sigma=radius_sigma, mode="wrap")

    boundary = np.column_stack((
        cx + radius_grid * np.cos(angle_grid),
        cy + radius_grid * np.sin(angle_grid),
    ))

    boundary[:, 0] = np.clip(boundary[:, 0], 0, width - 1)
    boundary[:, 1] = np.clip(boundary[:, 1], 0, height - 1)

    return boundary


def get_boundary(mask, center=None):
    """Public boundary helper used by segment_frame."""
    return smooth_boundary_from_mask(mask, center=center)


def segment_frame(frame):
    """
    Segment the bright particle on a raw grayscale frame.

    Returns
    -------
    mask : bool ndarray
    boundary : (N, 2) array of (x, y) edge points
    center : (cx, cy)
    overlay : BGR uint8 image with red boundary
    isolated : copy of frame with background zeroed
    """
    if frame is None:
        raise ValueError("The input frame is None.")

    if frame.ndim != 2:
        raise ValueError("segment_frame expects a grayscale frame.")

    img = normalize_frame(frame)
    smooth = gaussian(img, sigma=1.0, preserve_range=True)
    mask, _threshold = bright_particle_mask(smooth)

    cy, cx = ndimage.center_of_mass(mask)
    center = (float(cx), float(cy))
    boundary = get_boundary(mask, center=center)

    frame_display = np.round(img * 255).astype(np.uint8)
    height, width = frame_display.shape
    scale = 4

    frame_large = cv2.resize(
        frame_display,
        (width * scale, height * scale),
        interpolation=cv2.INTER_CUBIC,
    )
    overlay_large = cv2.cvtColor(frame_large, cv2.COLOR_GRAY2BGR)
    boundary_large = np.round(boundary * scale).astype(np.int32).reshape(-1, 1, 2)
    line_thickness = max(2, scale)

    cv2.polylines(
        overlay_large,
        [boundary_large],
        isClosed=True,
        color=(0, 0, 255),
        thickness=line_thickness,
        lineType=cv2.LINE_AA,
    )
    overlay = cv2.resize(
        overlay_large,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )

    isolated = frame.copy()
    isolated[~mask] = 0

    return mask, boundary, center, overlay, isolated

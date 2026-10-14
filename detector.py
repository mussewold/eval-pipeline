from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .models import (
    AcPayeeCandidate,
    AcPayeeEvidence,
    AnalysisResult,
    BoundingBox,
    ChequeLocalizationEvidence,
    LineCrossingEvidence,
    LineEvidence,
    Measurements,
    OrientationEvidence,
)

from .ocr import (
    OCRReader,
    OCRToken,
    OCRUnavailable,
    edit_distance,
    normalized_text,
    phrase_distance,
    read_ocr,
)


class ImageError(ValueError):
    pass


@dataclass(frozen=True)
class Segment:
    x1: int
    y1: int
    x2: int
    y2: int
    angle: float
    length: float

    @property
    def midpoint(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass(frozen=True)
class Pair:
    first: Segment
    second: Segment
    score: float
    angle_difference: float
    separation: float
    overlap: float


@dataclass(frozen=True)
class LayoutLine:
    x: int
    y: int
    width: int


@dataclass(frozen=True)
class ChequeCandidate:
    quad: np.ndarray
    geometry_score: float
    is_full_frame: bool = False


@dataclass(frozen=True)
class NormalizedCheque:
    image: np.ndarray
    gray: np.ndarray
    metrics: dict[str, float]
    localization: ChequeLocalizationEvidence
    orientation: OrientationEvidence
    ocr_tokens: list[OCRToken]
    ocr_error: str | None


def normalize_angle(angle: float) -> float:
    """Return an undirected image-space line angle in [0, 180)."""
    return angle % 180.0


def is_crossing_angle(angle: float, config: dict[str, Any]) -> bool:
    """Accept diagonal strokes only; cheque form lines are horizontal or vertical."""
    angle = normalize_angle(angle)
    validation = config["detection"]["validation"]
    minimum = validation["diagonal_min_angle"]
    maximum = validation["diagonal_max_angle"]
    return minimum <= angle <= maximum or 180 - maximum <= angle <= 180 - minimum


def _angle_difference(a: float, b: float) -> float:
    difference = abs(normalize_angle(a) - normalize_angle(b))
    return min(difference, 180.0 - difference)


def _projection_overlap(first: Segment, second: Segment) -> float:
    angle = math.radians((first.angle + second.angle) / 2)
    ux, uy = math.cos(angle), -math.sin(angle)

    def interval(segment: Segment) -> tuple[float, float]:
        values = (segment.x1 * ux + segment.y1 * uy, segment.x2 * ux + segment.y2 * uy)
        return min(values), max(values)

    a0, a1 = interval(first)
    b0, b1 = interval(second)
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    return overlap / max(1.0, min(a1 - a0, b1 - b0))


def _projection_gap(first: Segment, second: Segment) -> float:
    """Return the pixel gap between segments measured along their mean direction."""
    angle = math.radians((first.angle + second.angle) / 2)
    ux, uy = math.cos(angle), -math.sin(angle)

    def interval(segment: Segment) -> tuple[float, float]:
        values = (segment.x1 * ux + segment.y1 * uy, segment.x2 * ux + segment.y2 * uy)
        return min(values), max(values)

    a0, a1 = interval(first)
    b0, b1 = interval(second)
    return max(0.0, max(a0, b0) - min(a1, b1))


def _perpendicular_separation(first: Segment, second: Segment) -> float:
    angle = math.radians((first.angle + second.angle) / 2)
    nx, ny = math.sin(angle), math.cos(angle)
    ax, ay = first.midpoint
    bx, by = second.midpoint
    return abs((bx - ax) * nx + (by - ay) * ny)


def score_pair(
    first: Segment,
    second: Segment,
    roi_width: int,
    roi_height: int,
    config: dict[str, Any],
    reference_diagonal: float | None = None,
) -> Pair | None:
    """Validate and score two segments as a possible cheque crossing mark."""
    detection = config["detection"]
    validation = detection["validation"]
    scoring = detection["scoring"]
    weights = scoring["weights"]
    roi_diagonal = reference_diagonal or math.hypot(roi_width, roi_height)

    # Reject candidates early unless they form a nearly parallel, diagonal
    # pair in the expected part of the search region.
    angle_difference = _angle_difference(first.angle, second.angle)
    if angle_difference > validation["max_angle_difference_degrees"]:
        return None
    mean_angle = normalize_angle((first.angle + second.angle) / 2)
    if not is_crossing_angle(mean_angle, config):
        return None
    for segment in (first, second):
        center_x, center_y = segment.midpoint
        if center_x / roi_width > validation["max_segment_center_x_ratio"]:
            return None
        if center_y / roi_height > validation["max_segment_center_y_ratio"]:
            return None
    separation = _perpendicular_separation(first, second)
    separation_ratio = separation / roi_diagonal
    if not (
        validation["min_pair_separation_ratio"]
        <= separation_ratio
        <= validation["max_pair_separation_ratio"]
    ):
        return None
    overlap = _projection_overlap(first, second)
    if overlap < validation["min_overlap_ratio"]:
        return None

    # Combine independent geometric signals into one evidence score. These
    # weights are operational thresholds to calibrate on labeled cheque data.
    # Scoring scales are intentionally independent from validation limits.
    # Making validation more permissive therefore does not inflate the score
    # of a pair that was already valid.
    angle_score = max(0.0, 1.0 - angle_difference / scoring["angle_zero_score_degrees"])
    # Real crossing text often breaks each ink stroke into shorter visible segments.
    length_score = min(
        1.0,
        min(first.length, second.length) / (roi_diagonal * scoring["full_length_ratio"]),
    )
    separation_score = max(
        0.0,
        1.0
        - abs(separation_ratio - scoring["ideal_separation_ratio"])
        / scoring["separation_zero_score_deviation_ratio"],
    )
    score = (
        weights["angle"] * angle_score
        + weights["length"] * length_score
        + weights["overlap"] * overlap
        + weights["separation"] * separation_score
    )
    return Pair(first, second, score, angle_difference, separation, overlap)


def _order_quad(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype="float32").reshape(4, 2)
    ordered = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0], ordered[2] = points[np.argmin(sums)], points[np.argmax(sums)]
    ordered[1], ordered[3] = points[np.argmin(differences)], points[np.argmax(differences)]
    return ordered


def _warp_quad(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    top_left, top_right, bottom_right, bottom_left = _order_quad(points)
    width = round(max(np.linalg.norm(bottom_right - bottom_left), np.linalg.norm(top_right - top_left)))
    height = round(max(np.linalg.norm(top_right - bottom_right), np.linalg.norm(top_left - bottom_left)))
    if width < 2 or height < 2:
        raise ImageError("Cheque candidate is too small")
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )
    transform = cv2.getPerspectiveTransform(
        np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32"),
        destination,
    )
    return cv2.warpPerspective(image, transform, (width, height))


def _resize_cheque(image: np.ndarray, target_width: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = target_width / width
    if abs(scale - 1.0) <= 0.02:
        return image
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, (target_width, max(1, round(height * scale))), interpolation=interpolation)


def _quad_bbox(quad: np.ndarray) -> tuple[float, float, float, float]:
    x1, y1 = np.min(quad, axis=0)
    x2, y2 = np.max(quad, axis=0)
    return float(x1), float(y1), float(x2), float(y2)


def _bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = _quad_bbox(first)
    bx1, by1, bx2, by2 = _quad_bbox(second)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / max(1.0, union)


def _candidate_quads(image: np.ndarray, config: dict[str, Any]) -> list[ChequeCandidate]:
    """Generate rectangular cheque proposals from borders and merged page content."""
    settings = config["localization"]
    source_height, source_width = image.shape[:2]
    detection_scale = min(1.0, 1800 / max(source_width, source_height))
    work = (
        cv2.resize(image, None, fx=detection_scale, fy=detection_scale, interpolation=cv2.INTER_AREA)
        if detection_scale < 1
        else image
    )
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 140)
    edge_closed = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, width // 90), max(3, height // 120))),
        iterations=2,
    )
    _, ink = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    merged_ink = cv2.morphologyEx(
        ink,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, width // 35), max(5, height // 90))),
        iterations=2,
    )
    # A cheque embedded in a scan often has no usable outer contour. Merge its
    # text rows and form rules into one structural block, while suppressing
    # scanner/page borders before the wide close operation. This produces a
    # proposal from the cheque content itself without requiring OCR.
    structural_edges = edges.copy()
    border_x = max(3, round(width * 0.02))
    border_y = max(3, round(height * 0.015))
    structural_edges[:border_y, :] = 0
    structural_edges[-border_y:, :] = 0
    structural_edges[:, :border_x] = 0
    structural_edges[:, -border_x:] = 0
    structural_block = cv2.morphologyEx(
        structural_edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(9, width // 12), max(5, height // 50)),
        ),
        iterations=2,
    )
    structural_block = cv2.dilate(
        structural_block,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
        iterations=1,
    )
    image_area = width * height
    proposals: list[ChequeCandidate] = []

    # A cheque photographed on a desk, held by a hand, or surrounded by an A4
    # page often has weak/broken outer contours. Paper still forms a broad,
    # bright, low-saturation band across many rows and columns. Its supported
    # extents provide a robust whole-cheque proposal before any crop can remove
    # content. A nearly full-frame result is skipped because the ordinary full
    # frame proposal already represents it without resampling.
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    paper_pixels = (
        (hsv[:, :, 1] < settings["paper_max_saturation"])
        & (hsv[:, :, 2] > settings["paper_min_brightness"])
    )
    axis_support = settings["paper_min_axis_support"]
    supported_rows = np.flatnonzero(np.mean(paper_pixels, axis=1) >= axis_support)
    supported_columns = np.flatnonzero(np.mean(paper_pixels, axis=0) >= axis_support)
    if supported_rows.size and supported_columns.size:
        top, bottom = int(supported_rows[0]), int(supported_rows[-1])
        left, right = int(supported_columns[0]), int(supported_columns[-1])
        paper_width = right - left + 1
        paper_height = bottom - top + 1
        paper_area_ratio = paper_width * paper_height / image_area
        paper_aspect = max(paper_width, paper_height) / max(
            1, min(paper_width, paper_height)
        )
        inset = max(left, top, width - 1 - right, height - 1 - bottom)
        if (
            inset >= min(width, height) * 0.015
            and settings["min_area_ratio"] <= paper_area_ratio <= settings["max_area_ratio"]
            and settings["min_aspect_ratio"] <= paper_aspect <= settings["max_aspect_ratio"]
        ):
            # Keep a narrow safety band. Handwritten crossings can
            # touch or slightly overrun the printed paper boundary, especially
            # at the upper-left corner. This is deliberately based on the
            # detected cheque size rather than on the complete source frame.
            vertical_padding = max(1, round(paper_height * 0.03))
            horizontal_padding = 1
            left = max(0, left - horizontal_padding)
            right = min(width - 1, right + horizontal_padding)
            top = max(0, top - vertical_padding)
            bottom = min(height - 1, bottom + vertical_padding)
            paper_quad = np.array(
                [[left, top], [right, top], [right, bottom], [left, bottom]],
                dtype="float32",
            ) / detection_scale
            proposals.append(ChequeCandidate(paper_quad, 1.0))

    for mask in (edge_closed, merged_ink, structural_block):
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:80]:
            rect = cv2.minAreaRect(contour)
            rect_width, rect_height = rect[1]
            if min(rect_width, rect_height) < 30:
                continue
            rect_area = rect_width * rect_height
            area_ratio = rect_area / image_area
            aspect = max(rect_width, rect_height) / max(1.0, min(rect_width, rect_height))
            if not (
                settings["min_area_ratio"] <= area_ratio <= settings["max_area_ratio"]
                and settings["min_aspect_ratio"] <= aspect <= settings["max_aspect_ratio"]
            ):
                continue
            fill = min(1.0, cv2.contourArea(contour) / max(1.0, rect_area))
            aspect_score = max(0.0, 1.0 - abs(aspect - 2.2) / 2.0)
            area_score = min(1.0, area_ratio / 0.20)
            geometry_score = 0.50 * aspect_score + 0.30 * area_score + 0.20 * fill
            quad = cv2.boxPoints(rect) / detection_scale
            quad[:, 0] = np.clip(quad[:, 0], 0, source_width - 1)
            quad[:, 1] = np.clip(quad[:, 1], 0, source_height - 1)
            proposals.append(ChequeCandidate(quad.astype("float32"), geometry_score))

    # A photographed cheque may span the complete image width while unrelated
    # desk/background content remains below it. Its left, top, and right edges
    # then coincide with the photo boundary, so contour extraction sees only
    # internal cheque boxes. Long horizontal edges can still identify the
    # cheque's lower boundary and form a full-width top-band proposal.
    source_aspect = max(source_width, source_height) / min(source_width, source_height)
    if source_aspect < settings["min_aspect_ratio"]:
        row_support = np.mean(edges > 0, axis=1)
        minimum_y = round(width / settings["max_aspect_ratio"])
        maximum_y = min(height - 1, round(width / settings["min_aspect_ratio"]))
        eligible_rows = np.flatnonzero(
            (row_support >= settings["min_full_width_boundary_support"])
            & (np.arange(height) >= minimum_y)
            & (np.arange(height) <= maximum_y)
        )
        groups = np.split(eligible_rows, np.where(np.diff(eligible_rows) > 3)[0] + 1)
        boundaries: list[tuple[float, int]] = []
        for group in groups:
            if group.size == 0:
                continue
            boundary_y = int(group[np.argmax(row_support[group])])
            boundaries.append((float(row_support[boundary_y]), boundary_y))
        for support, boundary_y in sorted(boundaries, reverse=True)[:4]:
            candidate_width = width
            candidate_height = boundary_y + 1
            aspect = candidate_width / max(1, candidate_height)
            area_ratio = candidate_width * candidate_height / image_area
            if not (
                settings["min_area_ratio"] <= area_ratio <= settings["max_area_ratio"]
                and settings["min_aspect_ratio"] <= aspect <= settings["max_aspect_ratio"]
            ):
                continue
            aspect_score = max(0.0, 1.0 - abs(aspect - 2.2) / 2.0)
            coverage_score = min(1.0, area_ratio / 0.65)
            boundary_score = min(
                1.0, support / settings["min_full_width_boundary_support"]
            )
            quad = np.array(
                [[0, 0], [width - 1, 0], [width - 1, boundary_y], [0, boundary_y]],
                dtype="float32",
            ) / detection_scale
            proposals.append(
                ChequeCandidate(
                    quad,
                    0.45 * aspect_score
                    + 0.35 * coverage_score
                    + 0.20 * boundary_score,
                )
            )

    # Dark scanner beds can connect the cheque to long bright side rails and
    # defeat contour extraction. On predominantly dark sources, locate the
    # contiguous row band containing substantial bright paper instead.
    if float(np.median(gray)) < 80:
        bright = gray > max(90, int(np.percentile(gray, 70)))
        active_rows = np.flatnonzero(np.mean(bright, axis=1) >= 0.15)
        row_groups = np.split(active_rows, np.where(np.diff(active_rows) > 1)[0] + 1)
        for group in sorted(row_groups, key=len, reverse=True)[:2]:
            if len(group) < height * 0.05:
                continue
            top, bottom = int(group[0]), int(group[-1])
            column_support = np.mean(bright[top : bottom + 1], axis=0)
            active_columns = np.flatnonzero(column_support >= 0.12)
            if active_columns.size == 0:
                continue
            left, right = int(active_columns[0]), int(active_columns[-1])
            candidate_width = right - left + 1
            candidate_height = bottom - top + 1
            aspect = max(candidate_width, candidate_height) / max(
                1, min(candidate_width, candidate_height)
            )
            area_ratio = candidate_width * candidate_height / image_area
            if not (
                settings["min_area_ratio"] <= area_ratio <= settings["max_area_ratio"]
                and settings["min_aspect_ratio"] <= aspect <= settings["max_aspect_ratio"]
            ):
                continue
            quad = np.array(
                [[left, top], [right, top], [right, bottom], [left, bottom]],
                dtype="float32",
            ) / detection_scale
            aspect_score = max(0.0, 1.0 - abs(aspect - 2.2) / 2.0)
            area_score = min(1.0, area_ratio / 0.20)
            proposals.append(
                ChequeCandidate(quad, 0.55 * aspect_score + 0.45 * area_score)
            )

    if settings["min_aspect_ratio"] <= source_aspect <= settings["max_aspect_ratio"]:
        full_quad = np.array(
            [[0, 0], [source_width - 1, 0], [source_width - 1, source_height - 1], [0, source_height - 1]],
            dtype="float32",
        )
        aspect_score = max(0.0, 1.0 - abs(source_aspect - 2.2) / 2.0)
        proposals.append(
            ChequeCandidate(full_quad, 0.90 + 0.10 * aspect_score, is_full_frame=True)
        )

    retained: list[ChequeCandidate] = []
    for proposal in sorted(proposals, key=lambda item: item.geometry_score, reverse=True):
        if any(_bbox_iou(proposal.quad, existing.quad) >= settings["duplicate_iou"] for existing in retained):
            continue
        retained.append(proposal)
        if len(retained) >= settings["max_candidates"]:
            break
    return retained


def _layout_orientation_score(image: np.ndarray) -> float:
    """Prefer the direction placing the MICR-like dense band at the bottom."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height = gray.shape[0]
    ink = gray < 150
    band = max(1, round(height * 0.24))
    top_density = float(np.mean(ink[:band]))
    bottom_density = float(np.mean(ink[-band:]))
    return bottom_density / max(1e-6, top_density + bottom_density)


def _structure_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    horizontal = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, width // 16), 1)),
    )
    line_ratio = min(1.0, np.count_nonzero(horizontal) / max(1.0, width * height * 0.025))
    content_ratio = float(np.mean(gray < 200))
    content_score = min(1.0, content_ratio / 0.12)
    return 0.55 * line_ratio + 0.45 * content_score


def _ocr_score(tokens: list[OCRToken], anchors: list[str]) -> float:
    if not tokens:
        return 0.0
    normalized = [normalized_text(token.text) for token in tokens]
    readable = sum(token.confidence * min(12, len(text)) for token, text in zip(tokens, normalized))
    readable_score = min(1.0, readable / 45.0)
    anchor_hits = sum(any(anchor in text for text in normalized) for anchor in anchors)
    return min(1.0, 0.65 * readable_score + 0.35 * min(1.0, anchor_hits / 3.0))


def _ocr_orientation_score(
    tokens: list[OCRToken], anchors: list[str], image_height: int
) -> float:
    """Score both OCR readability and whether cheque fields occupy expected rows."""
    readability = _ocr_score(tokens, anchors)
    if not tokens:
        return readability

    clues: list[tuple[float, float]] = []
    for token in tokens:
        text = normalized_text(token.text)
        if not text:
            continue
        center_y = ((token.box[1] + token.box[3]) / 2) / max(1, image_height)
        target: float | None = None
        weight = token.confidence
        if "DATE" in text:
            target, weight = 0.18, weight * 1.4
        elif "CHEQUE" in text:
            target, weight = 0.20, weight * 1.1
        elif text.startswith("PAY") or "PAYEE" in text:
            target, weight = 0.28, weight * 1.2
        elif "RUPEES" in text or "BEARER" in text:
            target = 0.35
        elif "SIGNATURE" in text:
            target, weight = 0.78, weight * 1.3
        else:
            digit_ratio = sum(character.isdigit() for character in text) / len(text)
            if len(text) >= 8 and digit_ratio >= 0.65:
                target, weight = 0.90, weight * 1.5
        if target is not None:
            position_score = max(0.0, 1.0 - abs(center_y - target) / 0.65)
            clues.append((position_score, weight))

    if not clues:
        return readability
    position_score = sum(score * weight for score, weight in clues) / sum(
        weight for _, weight in clues
    )
    return 0.5 * readability + 0.5 * position_score


def _orient_candidate(
    image: np.ndarray,
    config: dict[str, Any],
    ocr_reader: OCRReader,
) -> tuple[np.ndarray, list[OCRToken], OrientationEvidence, str | None, float]:
    source_height, source_width = image.shape[:2]
    if source_height > source_width:
        first = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        first_rotation = 90
    else:
        first = image
        first_rotation = 0
    first = _resize_cheque(first, config["normalization"]["target_width"])
    second = cv2.rotate(first, cv2.ROTATE_180)
    second_rotation = (first_rotation + 180) % 360

    token_sets: list[list[OCRToken]] = [[], []]
    errors: list[str] = []
    for index, candidate in enumerate((first, second)):
        try:
            token_sets[index] = ocr_reader(candidate)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    anchors = config["orientation"]["anchor_words"]
    readability_scores = [_ocr_score(tokens, anchors) for tokens in token_sets]
    orientation_scores = [
        _ocr_orientation_score(tokens, anchors, candidate.shape[0])
        for tokens, candidate in zip(token_sets, (first, second))
    ]
    maximum = max(orientation_scores)
    margin = abs(orientation_scores[0] - orientation_scores[1])
    relative_margin = margin / max(1e-6, maximum)
    if maximum > 0 and relative_margin >= config["orientation"]["minimum_ocr_margin"]:
        selected = int(orientation_scores[1] > orientation_scores[0])
        confidence = min(1.0, 0.55 + relative_margin * 0.45)
        method = "ocr"
    elif errors and not token_sets[0] and not token_sets[1]:
        # Preserve the established direction when OCR is genuinely unavailable.
        # Portrait inputs still receive the historical clockwise landscape turn.
        selected = 0
        confidence = 0.5
        method = "layout_fallback"
    else:
        layout_scores = [_layout_orientation_score(first), _layout_orientation_score(second)]
        selected = int(layout_scores[1] > layout_scores[0])
        confidence = min(1.0, 0.5 + abs(layout_scores[0] - layout_scores[1]))
        method = "layout_fallback"

    chosen_image = (first, second)[selected]
    chosen_tokens = token_sets[selected]
    orientation = OrientationEvidence(
        rotation_degrees_clockwise=(first_rotation, second_rotation)[selected],
        method=method,
        confidence=round(confidence, 3),
    )
    return (
        chosen_image,
        chosen_tokens,
        orientation,
        "; ".join(errors) or None,
        readability_scores[selected],
    )


def _is_full_frame(quad: np.ndarray, source_width: int, source_height: int) -> bool:
    x1, y1, x2, y2 = _quad_bbox(quad)
    return (
        x1 <= source_width * 0.03
        and y1 <= source_height * 0.03
        and x2 >= source_width * 0.97
        and y2 >= source_height * 0.97
    )


def _load_and_normalize(
    path: Path,
    config: dict[str, Any],
    ocr_reader: OCRReader | None = None,
) -> NormalizedCheque:
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ImageError("Unsupported image type; use JPG or PNG")
    try:
        # Pillow applies the camera/scanner EXIF transform explicitly. Relying
        # on OpenCV here is version-dependent and can make portrait cheques
        # enter localization with the wrong dimensions.
        with Image.open(path) as opened:
            upright_source = ImageOps.exif_transpose(opened).convert("RGB")
            source = cv2.cvtColor(np.asarray(upright_source), cv2.COLOR_RGB2BGR)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise ImageError("Image is unreadable or corrupt")
    height, width = source.shape[:2]
    normalization = config["normalization"]
    if width < normalization["min_width"] or height < normalization["min_height"]:
        raise ImageError(
            f"Image is too small; minimum is {normalization['min_width']}x{normalization['min_height']}"
        )

    reader = ocr_reader or read_ocr
    candidates = _candidate_quads(source, config)
    evaluated: list[
        tuple[float, ChequeCandidate, np.ndarray, list[OCRToken], OrientationEvidence, str | None]
    ] = []
    for candidate in candidates:
        try:
            warped = (
                source.copy()
                if candidate.is_full_frame
                else _warp_quad(source, candidate.quad)
            )
            oriented, tokens, orientation, ocr_error, ocr_score = _orient_candidate(
                warped, config, reader
            )
        except (cv2.error, ImageError):
            continue
        # Candidate evidence combines the outer cheque geometry, horizontal
        # form structure, readable cheque anchors, and the MICR-like dense
        # content expected along the lower long edge after orientation.
        score = (
            0.45 * candidate.geometry_score
            + 0.25 * _structure_score(oriented)
            + 0.20 * ocr_score
            + 0.10 * _layout_orientation_score(oriented)
        )
        evaluated.append((score, candidate, oriented, tokens, orientation, ocr_error))

    if not evaluated:
        raise ImageError("cheque could not be located")
    evaluated.sort(key=lambda item: item[0], reverse=True)
    full_frame_options = [
        item for item in evaluated if item[1].is_full_frame
    ]
    # A source frame with cheque proportions is already a valid cheque view.
    # Prefer it over small high-contrast logos or decorative rectangles.
    best = full_frame_options[0] if full_frame_options else evaluated[0]
    if best[0] < config["localization"]["minimum_score"]:
        raise ImageError("cheque could not be located")
    competing = [item for item in evaluated if item is not best]
    if competing and not full_frame_options:
        runner_up = competing[0]
        if (
            best[0] - runner_up[0] < config["localization"]["ambiguity_margin"]
            and _bbox_iou(best[1].quad, runner_up[1].quad) < config["localization"]["duplicate_iou"]
            and bool(best[3])
            and bool(runner_up[3])
        ):
            raise ImageError("multiple cheque candidates were found")

    score, candidate, image, tokens, orientation, ocr_error = best
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=normalization["clahe_clip_limit"], tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(gray)
    image_height, image_width = gray.shape
    full_frame = candidate.is_full_frame
    localization = ChequeLocalizationEvidence(
        status="full_frame" if full_frame else "located",
        confidence=round(min(1.0, score), 3),
        source_quad=np.rint(_order_quad(candidate.quad)).astype(int).tolist(),
        output_width=image_width,
        output_height=image_height,
    )
    metrics = {
        "blur_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "contrast_std": float(gray.std()),
        "dark_pixel_ratio": float(np.mean(gray < 100)),
        "aspect_ratio": float(image_width / image_height),
        "perspective_rectified": float(not full_frame),
    }
    return NormalizedCheque(
        image=image,
        gray=enhanced,
        metrics=metrics,
        localization=localization,
        orientation=orientation,
        ocr_tokens=tokens,
        ocr_error=ocr_error,
    )


def _quality(metrics: dict[str, float], config: dict[str, Any]) -> tuple[str, list[str]]:
    thresholds = config["quality"]
    reasons: list[str] = []
    if metrics["blur_variance"] < thresholds["blur_variance_min"]:
        reasons.append("Image appears blurred")
    if metrics["contrast_std"] < thresholds["contrast_std_min"]:
        reasons.append("Image contrast is low")
    if metrics["dark_pixel_ratio"] < thresholds["dark_pixel_min"]:
        reasons.append("Very little foreground detail is visible")
    if not 1.6 <= metrics["aspect_ratio"] <= 3.3:
        reasons.append("Cheque framing or aspect ratio is unusual")
    return ("low" if reasons else "good"), reasons


def _deduplicate(
    segments: list[Segment], roi_diagonal: float, config: dict[str, Any]
) -> list[Segment]:
    """Keep one representative for each physical ink stroke.

    Edge and Hough detection often return both edges or several overlapping
    fragments of one thick stroke. Because segments are processed longest
    first, duplicates can be discarded while retaining the best representative.
    """
    validation = config["detection"]["validation"]
    retained: list[Segment] = []
    for segment in sorted(segments, key=lambda item: item.length, reverse=True):
        duplicate = False
        for other in retained:
            same_angle = (
                _angle_difference(segment.angle, other.angle)
                <= validation["same_stroke_max_angle_difference_degrees"]
            )
            same_stroke_distance = (
                _perpendicular_separation(segment, other) / roi_diagonal
                <= validation["same_stroke_max_separation_ratio"]
            )
            projections_touch = (
                _projection_gap(segment, other) / roi_diagonal
                <= validation["same_stroke_max_gap_ratio"]
            )
            if same_angle and same_stroke_distance and projections_touch:
                duplicate = True
                break
        if not duplicate:
            retained.append(segment)
    return retained[:40]


def _edge_support(segment: Segment, edges: np.ndarray, radius: int = 2) -> float:
    """Return the fraction of a segment backed by nearby edge pixels."""
    sample_count = max(2, round(segment.length))
    x_values = np.linspace(segment.x1, segment.x2, sample_count)
    y_values = np.linspace(segment.y1, segment.y2, sample_count)
    height, width = edges.shape
    supported = 0
    for x_value, y_value in zip(x_values, y_values):
        x = int(round(x_value))
        y = int(round(y_value))
        patch = edges[
            max(0, y - radius) : min(height, y + radius + 1),
            max(0, x - radius) : min(width, x + radius + 1),
        ]
        supported += int(bool(np.any(patch)))
    return supported / sample_count


def _find_segments(
    gray_roi: np.ndarray,
    config: dict[str, Any],
    reference_diagonal: float | None = None,
) -> tuple[list[Segment], np.ndarray]:
    """Extract plausible diagonal ink strokes from the crossing search area."""
    detection = config["detection"]
    kernel_size = int(config["normalization"]["blur_kernel"])
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred = cv2.GaussianBlur(gray_roi, (kernel_size, kernel_size), 0)
    edges = cv2.Canny(blurred, detection["canny_low"], detection["canny_high"])
    height, width = gray_roi.shape
    diagonal = reference_diagonal or math.hypot(width, height)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=detection["hough_threshold"],
        minLineLength=diagonal * detection["min_line_length_ratio"],
        maxLineGap=diagonal * detection["max_line_gap_ratio"],
    )
    segments: list[Segment] = []
    if lines is not None:
        for raw in lines[:, 0]:
            x1, y1, x2, y2 = map(int, raw)
            length = math.hypot(x2 - x1, y2 - y1)
            # Negate image-space dy so labels follow the familiar Cartesian
            # convention: / is a positive angle and \\ is a negative angle.
            angle = normalize_angle(math.degrees(math.atan2(-(y2 - y1), x2 - x1)))
            segment = Segment(x1, y1, x2, y2, angle, length)
            if (
                is_crossing_angle(angle, config)
                and _edge_support(segment, edges)
                >= detection["validation"]["min_line_edge_support_ratio"]
            ):
                segments.append(segment)
    return _deduplicate(segments, diagonal, config), edges


def _best_pairs(
    segments: list[Segment],
    roi_width: int,
    roi_height: int,
    config: dict[str, Any],
    reference_diagonal: float | None = None,
) -> list[Pair]:
    pairs = [
        pair
        for index, first in enumerate(segments)
        for second in segments[index + 1 :]
        if (
            pair := score_pair(
                first,
                second,
                roi_width,
                roi_height,
                config,
                reference_diagonal,
            )
        )
        is not None
    ]
    return sorted(pairs, key=lambda pair: pair.score, reverse=True)


def _pair_center(pair: Pair) -> tuple[float, float]:
    points = (
        (pair.first.x1, pair.first.y1),
        (pair.first.x2, pair.first.y2),
        (pair.second.x1, pair.second.y1),
        (pair.second.x2, pair.second.y2),
    )
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _pair_mean_angle(pair: Pair) -> float:
    return normalize_angle((pair.first.angle + pair.second.angle) / 2)


def _distinct_pairs(
    pairs: list[Pair], reference_diagonal: float, config: dict[str, Any]
) -> list[Pair]:
    """Collapse raw Hough combinations representing the same visible crossing."""
    decision = config["detection"]["decision"]
    representatives: list[Pair] = []
    for pair in pairs:  # Already sorted by score, so the best representative wins.
        center_x, center_y = _pair_center(pair)
        duplicate = False
        for representative in representatives:
            other_x, other_y = _pair_center(representative)
            center_distance = math.hypot(center_x - other_x, center_y - other_y)
            if (
                center_distance / reference_diagonal
                <= decision["same_candidate_center_distance_ratio"]
                and _angle_difference(
                    _pair_mean_angle(pair), _pair_mean_angle(representative)
                )
                <= decision["same_candidate_max_angle_difference_degrees"]
            ):
                duplicate = True
                break
        if not duplicate:
            representatives.append(pair)
    return representatives


def _pair_gap_dark_ratio(pair: Pair, gray_roi: np.ndarray, config: dict[str, Any]) -> float:
    """Measure dark ink in the central strip between two candidate lines."""
    validation = config["detection"]["validation"]
    angle = math.radians((pair.first.angle + pair.second.angle) / 2)
    ux, uy = math.cos(angle), -math.sin(angle)

    def ordered_endpoints(segment: Segment) -> list[tuple[int, int]]:
        points = [(segment.x1, segment.y1), (segment.x2, segment.y2)]
        return sorted(points, key=lambda point: point[0] * ux + point[1] * uy)

    first = ordered_endpoints(pair.first)
    second = ordered_endpoints(pair.second)
    start = tuple(round((first[0][axis] + second[0][axis]) / 2) for axis in (0, 1))
    end = tuple(round((first[1][axis] + second[1][axis]) / 2) for axis in (0, 1))
    thickness = max(
        1, round(pair.separation * validation["pair_gap_sample_width_ratio"])
    )
    mask = np.zeros(gray_roi.shape, dtype=np.uint8)
    cv2.line(mask, start, end, 255, thickness)
    pixels = gray_roi[mask > 0]
    if pixels.size == 0:
        return 1.0
    return float(np.mean(pixels < validation["pair_gap_dark_pixel_threshold"]))


def _is_top_boundary_artifact(
    pair: Pair,
    roi_height: int,
    reference_diagonal: float,
    config: dict[str, Any],
) -> bool:
    """Reject two short edges of a skewed crop boundary, not close real crossings."""
    validation = config["detection"]["validation"]

    def touches_top(segment: Segment) -> bool:
        return min(segment.y1, segment.y2) <= (
            roi_height * validation["boundary_artifact_margin_ratio"]
        )

    return (
        touches_top(pair.first)
        and touches_top(pair.second)
        and pair.separation / reference_diagonal
        <= validation["boundary_artifact_max_separation_ratio"]
        and max(pair.first.length, pair.second.length) / reference_diagonal
        <= validation["boundary_artifact_max_length_ratio"]
    )


def _layout_lines(gray: np.ndarray, config: dict[str, Any]) -> list[LayoutLine]:
    """Extract long horizontal cheque-field rules without recognizing text."""
    height, width = gray.shape
    settings = config["region"]["dynamic"]
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel_width = max(3, round(width * settings["horizontal_kernel_ratio"]))
    horizontal = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1)),
    )
    contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minimum_width = width * settings["min_horizontal_length_ratio"]
    maximum_thickness = max(3, height * settings["max_horizontal_thickness_ratio"])
    lines = []
    for contour in contours:
        x, y, line_width, line_height = cv2.boundingRect(contour)
        if line_width >= minimum_width and line_height <= maximum_thickness:
            lines.append(LayoutLine(x=x, y=y + line_height // 2, width=line_width))
    return lines


def _resolve_region(
    gray: np.ndarray,
    config: dict[str, Any],
    lines: list[LayoutLine] | None = None,
) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    """Set ROI ends from cheque form geometry, with proportional fallbacks."""
    height, width = gray.shape
    region = config["region"]
    x1 = round(width * region["x_start"])
    y1 = round(height * region["y_start"])
    fallback_x2 = round(width * region["x_end"])
    fallback_y2 = round(height * region["y_end"])
    diagnostics: dict[str, Any] = {
        "roi_x_source": "configured_fallback",
        "roi_y_source": "configured_fallback",
    }
    if not region["dynamic"]["enabled"]:
        diagnostics.update(
            roi_x_source="configured_fixed",
            roi_y_source="configured_fixed",
        )
        return (x1, y1, fallback_x2, fallback_y2), diagnostics

    anchors = region["dynamic"]
    if lines is None:
        lines = _layout_lines(gray, config)
    min_x2 = round(width * anchors["min_x_end_ratio"])
    max_x2 = round(width * anchors["max_x_end_ratio"])
    min_y2 = round(height * anchors["min_y_end_ratio"])
    max_y2 = round(height * anchors["max_y_end_ratio"])

    x2 = fallback_x2
    payee_lines = [
        line
        for line in lines
        if height * anchors["payee_min_y_ratio"] <= line.y <= height * anchors["payee_max_y_ratio"]
    ]
    if payee_lines:
        line = min(payee_lines, key=lambda item: item.x)
        candidate = line.x - round(width * anchors["x_margin_ratio"])
        if max(x1 + 1, min_x2) <= candidate <= max_x2:
            x2 = candidate
            diagnostics.update(roi_x_source="payee_field_geometry")

    y2 = fallback_y2
    date_lines = [line for line in lines if line.x >= width * anchors["date_min_x_ratio"]]
    if date_lines:
        line = min(date_lines, key=lambda item: item.y)
        candidate = line.y - round(height * anchors["y_margin_ratio"])
        if max(y1 + 1, min_y2) <= candidate <= max_y2:
            y2 = candidate
            diagnostics.update(roi_y_source="date_field_geometry")

    diagnostics["layout_line_count"] = len(lines)
    return (x1, y1, x2, y2), diagnostics


def _token_center_in_region(token: OCRToken, region: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = token.box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    rx1, ry1, rx2, ry2 = region
    return rx1 <= center_x <= rx2 and ry1 <= center_y <= ry2


def _ocr_phrase_groups(tokens: list[OCRToken]) -> list[list[OCRToken]]:
    """Build short same-line OCR windows so split A/C and Payee tokens can match."""
    rows: list[list[OCRToken]] = []
    for token in sorted(tokens, key=lambda item: ((item.box[1] + item.box[3]) / 2, item.box[0])):
        center_y = (token.box[1] + token.box[3]) / 2
        height = max(1, token.box[3] - token.box[1])
        for row in rows:
            row_center = sum((item.box[1] + item.box[3]) / 2 for item in row) / len(row)
            row_height = max(max(1, item.box[3] - item.box[1]) for item in row)
            if abs(center_y - row_center) <= max(height, row_height) * 0.8:
                row.append(token)
                break
        else:
            rows.append([token])

    groups: list[list[OCRToken]] = []
    for row in rows:
        row.sort(key=lambda item: item.box[0])
        for start in range(len(row)):
            for length in range(1, min(4, len(row) - start + 1)):
                window = row[start : start + length]
                if len(window) > 1:
                    gap_limit = max(item.box[3] - item.box[1] for item in window) * 4
                    if any(
                        right.box[0] - left.box[2] > gap_limit
                        for left, right in zip(window, window[1:])
                    ):
                        continue
                groups.append(window)
    return groups


def _strip_fuzzy_only_suffix(value: str, max_edits: int) -> tuple[str, bool]:
    """Remove a trailing OCR-corrupted ONLY token when it is sufficiently close."""
    if value.endswith("ONLY"):
        return value[:-4], True
    minimum_length = max(2, len("ONLY") - max_edits)
    maximum_length = min(len(value), len("ONLY") + max_edits)
    matches: list[tuple[int, int]] = []
    for suffix_length in range(minimum_length, maximum_length + 1):
        distance = edit_distance(value[-suffix_length:], "ONLY")
        if distance <= max_edits:
            matches.append((distance, suffix_length))
    if not matches:
        return value, False
    _, suffix_length = min(matches, key=lambda item: (item[0], abs(item[1] - 4)))
    return value[:-suffix_length], True


def _ac_payee_candidate_evidence(
    tokens: list[OCRToken], source: str, config: dict[str, Any]
) -> list[AcPayeeCandidate]:
    """Expose phrase-level OCR candidates and the math used to assess them."""
    candidates: list[AcPayeeCandidate] = []
    seen: set[tuple[str, tuple[int, int, int, int]]] = set()
    settings = config["text_detection"]
    for group in _ocr_phrase_groups(tokens):
        display = " ".join(token.text for token in group)
        normalized = normalized_text(display)
        candidate_like = normalized.startswith("A") or any(
            marker in normalized for marker in ("PAYEE", "PAE")
        )
        x1 = min(token.box[0] for token in group)
        y1 = min(token.box[1] for token in group)
        x2 = max(token.box[2] for token in group)
        y2 = max(token.box[3] for token in group)
        group_box = (x1, y1, x2, y2)
        key = (normalized, group_box)
        if not normalized or not candidate_like or key in seen:
            continue
        seen.add(key)

        average_ocr = sum(token.confidence for token in group) / len(group)
        without_only, has_only_suffix = _strip_fuzzy_only_suffix(
            normalized, settings["only_suffix_max_edits"]
        )
        scored_targets: list[tuple[int, str, int]] = []
        for canonical, configured_edits in (
            ("ACPAYEE", settings["ac_payee_max_edits"]),
            ("ACCOUNTPAYEE", settings["account_payee_max_edits"]),
        ):
            allowed = (
                settings["ac_payee_only_max_edits"]
                if canonical == "ACPAYEE" and has_only_suffix
                else configured_edits
            )
            scored_targets.append((phrase_distance(without_only, canonical), canonical, allowed))
        distance, canonical, allowed_edits = min(scored_targets, key=lambda item: item[0])
        lexical_score = max(0.0, 1.0 - distance / len(canonical))
        match_confidence = average_ocr * lexical_score
        match_method = "whole_phrase"
        eligible = (
            average_ocr >= settings["min_ocr_confidence"]
            and without_only.startswith("A")
            and len(without_only) <= len(canonical) + allowed_edits
            and distance <= allowed_edits
        )

        component_distance: int | None = None
        if without_only.startswith("A"):
            for prefix_length in range(1, min(3, len(without_only) - 2) + 1):
                account_part = without_only[:prefix_length]
                payee_part = without_only[prefix_length:]
                account_distance = edit_distance(account_part, "AC")
                payee_distance = edit_distance(payee_part, "PAYEE")
                if (
                    account_distance <= settings["fuzzy_account_max_edits"]
                    and payee_distance <= settings["fuzzy_payee_max_edits"]
                ):
                    component_distance = account_distance + payee_distance
                    break
        if not eligible and component_distance is not None and average_ocr >= settings["min_ocr_confidence"]:
            canonical = "ACPAYEE"
            distance = component_distance
            allowed_edits = settings["fuzzy_account_max_edits"] + settings["fuzzy_payee_max_edits"]
            lexical_score = max(0.0, 1.0 - distance / len(canonical))
            match_confidence = min(
                average_ocr,
                settings["automatic_accept_threshold"] - 0.001,
            )
            match_method = "component_fuzzy"
            eligible = True

        rejection_reason = None
        if not eligible:
            if average_ocr < settings["min_ocr_confidence"]:
                rejection_reason = "OCR confidence below minimum"
            elif not without_only.startswith("A"):
                rejection_reason = "missing account marker"
            elif len(without_only) > len(canonical) + allowed_edits:
                rejection_reason = "candidate is longer than the allowed phrase length"
            else:
                rejection_reason = "edit distance exceeds allowed maximum"
        candidates.append(
            AcPayeeCandidate(
                text=display,
                normalized_text=normalized,
                confidence=round(average_ocr, 3),
                source=source,
                box=BoundingBox(
                    x=x1,
                    y=y1,
                    width=max(1, x2 - x1),
                    height=max(1, y2 - y1),
                ),
                canonical_text=canonical,
                edit_distance=distance,
                allowed_edit_distance=allowed_edits,
                canonical_length=len(canonical),
                lexical_score=round(lexical_score, 3),
                match_confidence=round(min(1.0, match_confidence), 3),
                match_method=match_method if eligible else "none",
                eligible=eligible,
                rejection_reason=rejection_reason,
            )
        )
    return candidates


def _merge_ac_payee_candidates(
    *candidate_lists: list[AcPayeeCandidate],
) -> list[AcPayeeCandidate]:
    merged: list[AcPayeeCandidate] = []
    seen: set[tuple[str, str, int, int, int, int]] = set()
    for candidates in candidate_lists:
        for candidate in candidates:
            box = candidate.box
            key = (
                candidate.source,
                candidate.normalized_text,
                box.x,
                box.y,
                box.width,
                box.height,
            )
            if key not in seen:
                seen.add(key)
                merged.append(candidate)
    return merged


def _find_ac_payee(
    tokens: list[OCRToken],
    region: tuple[int, int, int, int],
    config: dict[str, Any],
    ocr_available: bool,
    candidate_source: str = "primary",
) -> AcPayeeEvidence:
    if not config["text_detection"]["enabled"]:
        return AcPayeeEvidence(status="unavailable")
    if not ocr_available:
        return AcPayeeEvidence(status="unavailable")

    inside = [token for token in tokens if _token_center_in_region(token, region)]
    candidates = _ac_payee_candidate_evidence(inside, candidate_source, config)
    settings = config["text_detection"]
    matches: list[tuple[float, str, list[OCRToken]]] = []
    for group in _ocr_phrase_groups(inside):
        display = " ".join(token.text for token in group)
        candidate = normalized_text(display)
        candidate_without_only, has_only_suffix = _strip_fuzzy_only_suffix(
            candidate, settings["only_suffix_max_edits"]
        )
        average_ocr = sum(token.confidence for token in group) / len(group)
        if average_ocr < settings["min_ocr_confidence"]:
            continue
        for canonical, configured_edits in (
            ("ACPAYEE", settings["ac_payee_max_edits"]),
            ("ACCOUNTPAYEE", settings["account_payee_max_edits"]),
        ):
            # Requiring an A-like account marker prevents ordinary "PAYEE
            # ONLY" labels from being promoted by the fuzzy matcher.
            if not candidate_without_only.startswith("A"):
                continue
            allowed_edits = (
                settings["ac_payee_only_max_edits"]
                if canonical == "ACPAYEE" and has_only_suffix
                else configured_edits
            )
            if len(candidate_without_only) > len(canonical) + allowed_edits:
                continue
            distance = phrase_distance(candidate_without_only, canonical)
            if distance > allowed_edits:
                continue
            lexical_score = 1.0 - distance / len(canonical)
            # OCR confidence is the upper bound; fuzzy matching may lower it
            # but must never inflate weak OCR above the configured threshold.
            confidence = average_ocr * lexical_score
            matches.append((confidence, display, group))

        # Compare the account marker and PAYEE wording independently. This is
        # safer than allowing a large edit distance across the whole phrase:
        # both semantic components must still be present.
        component_match = False
        if not candidate_without_only.startswith("A"):
            continue
        for prefix_length in range(1, min(3, len(candidate_without_only) - 2) + 1):
            account_part = candidate_without_only[:prefix_length]
            payee_part = candidate_without_only[prefix_length:]
            if (
                edit_distance(account_part, "AC")
                <= settings["fuzzy_account_max_edits"]
                and edit_distance(payee_part, "PAYEE")
                <= settings["fuzzy_payee_max_edits"]
            ):
                component_match = True
                break
        if component_match:
            # Component-fuzzy matches must always be reviewed. Exact or
            # near-exact whole-phrase evidence above can still win.
            confidence = min(
                average_ocr,
                settings["automatic_accept_threshold"] - 0.001,
            )
            matches.append((confidence, display, group))
    if not matches:
        return AcPayeeEvidence(status="not_found", candidates=candidates)

    confidence, display, group = max(matches, key=lambda item: item[0])
    if confidence < settings["min_ocr_confidence"]:
        return AcPayeeEvidence(status="not_found", candidates=candidates)
    x1 = min(token.box[0] for token in group)
    y1 = min(token.box[1] for token in group)
    x2 = max(token.box[2] for token in group)
    y2 = max(token.box[3] for token in group)
    for candidate in candidates:
        if (
            candidate.text == display
            and candidate.box.x == x1
            and candidate.box.y == y1
            and candidate.box.width == x2 - x1
            and candidate.box.height == y2 - y1
        ):
            candidate.selected = True
            break
    return AcPayeeEvidence(
        status="detected",
        matched_text=display,
        confidence=round(min(1.0, confidence), 3),
        box=BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1),
        candidates=candidates,
    )


def _retry_roi_ocr(
    image: np.ndarray,
    region: tuple[int, int, int, int],
    reader: OCRReader,
    padding: int,
    diagnostic_artifacts: dict[str, np.ndarray] | None = None,
) -> list[OCRToken]:
    """Retry OCR with white padding and map boxes back to cheque coordinates."""
    if padding <= 0:
        return []
    x1, y1, x2, y2 = region
    roi = image[y1:y2, x1:x2]
    padded = cv2.copyMakeBorder(
        roi,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    if diagnostic_artifacts is not None:
        diagnostic_artifacts["padded_roi"] = padded.copy()
    image_height, image_width = image.shape[:2]
    remapped: list[OCRToken] = []
    for token in reader(padded):
        tx1, ty1, tx2, ty2 = token.box
        mapped_x1 = max(0, min(image_width - 1, tx1 - padding + x1))
        mapped_y1 = max(0, min(image_height - 1, ty1 - padding + y1))
        mapped_x2 = max(mapped_x1 + 1, min(image_width, tx2 - padding + x1))
        mapped_y2 = max(mapped_y1 + 1, min(image_height, ty2 - padding + y1))
        remapped.append(
            OCRToken(
                text=token.text,
                confidence=token.confidence,
                box=(mapped_x1, mapped_y1, mapped_x2, mapped_y2),
            )
        )
    return remapped


def analyze_image(
    path: str | Path,
    config: dict[str, Any],
    ocr_reader: OCRReader | None = None,
    diagnostic_artifacts: dict[str, np.ndarray] | None = None,
) -> tuple[AnalysisResult, np.ndarray]:
    """Run deterministic detection and return its evidence and annotation."""
    source = Path(path)
    if not source.is_file():
        raise ImageError(f"Image not found: {source}")
    reader = ocr_reader or read_ocr
    normalized = _load_and_normalize(source, config, reader)
    image = normalized.image
    gray = normalized.gray
    quality_metrics = normalized.metrics
    height, width = gray.shape
    # Searching only the expected crossing zone reduces false matches from
    # payee fields, signatures, and other printed cheque content.
    (x1, y1, x2, y2), region_diagnostics = _resolve_region(gray, config)
    ac_payee = _find_ac_payee(
        normalized.ocr_tokens,
        (x1, y1, x2, y2),
        config,
        ocr_available=normalized.ocr_error is None,
    )
    ocr_retry_used = False
    ocr_retry_error: str | None = None
    retry_tokens: list[OCRToken] = []
    if ac_payee.status == "not_found" and normalized.ocr_error is None:
        try:
            retry_tokens = _retry_roi_ocr(
                image,
                (x1, y1, x2, y2),
                reader,
                config["text_detection"]["ocr_retry_padding_pixels"],
                diagnostic_artifacts,
            )
            retry_evidence = _find_ac_payee(
                retry_tokens,
                (x1, y1, x2, y2),
                config,
                ocr_available=True,
                candidate_source="padded_retry",
            )
            combined_candidates = _merge_ac_payee_candidates(
                ac_payee.candidates,
                retry_evidence.candidates,
            )
            if retry_evidence.status == "detected":
                ac_payee = retry_evidence
                ocr_retry_used = True
            ac_payee.candidates = combined_candidates
        except Exception as exc:
            ocr_retry_error = f"{type(exc).__name__}: {exc}"
    roi = gray[y1:y2, x1:x2]
    detection = config["detection"]
    reference_diagonal = math.hypot(
        width * detection["geometry_reference_width_ratio"],
        height * detection["geometry_reference_height_ratio"],
    )
    segments, _ = _find_segments(roi, config, reference_diagonal)
    all_raw_pairs = _best_pairs(
        segments, roi.shape[1], roi.shape[0], config, reference_diagonal
    )
    non_boundary_pairs = [
        pair
        for pair in all_raw_pairs
        if not _is_top_boundary_artifact(
            pair, roi.shape[0], reference_diagonal, config
        )
    ]
    max_gap_dark_ratio = config["detection"]["validation"]["max_pair_gap_dark_ratio"]
    raw_pairs = [
        pair
        for pair in non_boundary_pairs
        if _pair_gap_dark_ratio(pair, roi, config) <= max_gap_dark_ratio
    ]
    pairs = _distinct_pairs(raw_pairs, reference_diagonal, config)
    best = pairs[0] if pairs else None
    decision = config["detection"]["decision"]
    threshold = decision["positive_threshold"]

    line_detected = bool(best and best.score >= threshold)
    quality, quality_reasons = _quality(quality_metrics, config)

    line_confidence = (
        best.score
        if line_detected and best
        else max(0.0, min(1.0, threshold - (best.score if best else 0.0)) / threshold)
    )
    text_detected = ac_payee.status == "detected"
    crossing = text_detected or line_detected
    if text_detected and line_detected:
        decision_basis = "AC_PAYEE_AND_LINES"
    elif text_detected:
        decision_basis = "AC_PAYEE"
    elif line_detected:
        decision_basis = "LINES"
    else:
        decision_basis = "NONE"
    confidence = (
        max(ac_payee.confidence, line_confidence)
        if text_detected and line_detected
        else ac_payee.confidence
        if text_detected
        else line_confidence
    )

    # Several similarly scored pairs make the winning detection ambiguous.
    competing_pairs = (
        [pair for pair in pairs[1:] if best.score - pair.score < decision["competing_pair_margin"]]
        if line_detected and best
        else []
    )
    conflicting = bool(competing_pairs)
    review_reasons: list[str] = []
    if quality == "low":
        review_reasons.append("LOW_IMAGE_QUALITY")
    if text_detected and ac_payee.confidence < config["text_detection"]["automatic_accept_threshold"]:
        review_reasons.append("BORDERLINE_TEXT")
    elif line_detected and line_confidence < decision["automatic_accept_threshold"]:
        review_reasons.append("BORDERLINE_SCORE")
    if conflicting:
        review_reasons.append("COMPETING_CANDIDATES")
    needs_review = bool(review_reasons)
    decision_status = (
        "NO_CROSSING"
        if not crossing
        else "CROSSING_REVIEW"
        if needs_review
        else "CROSSING_ACCEPTED"
    )
    reasons = list(quality_reasons)
    if text_detected:
        reasons.append(
            f"Found A/C Payee instruction '{ac_payee.matched_text}' with confidence {ac_payee.confidence:.3f}"
        )
    elif ac_payee.status == "unavailable":
        reasons.append("A/C Payee recognition was unavailable; used line evidence only")
    else:
        reasons.append("No A/C Payee instruction was found in the crossing zone")
    if line_detected and best:
        reasons.append(f"Found a paired diagonal crossing mark scoring {best.score:.3f}")
        reasons.append(f"Pair angle difference is {best.angle_difference:.1f} degrees with {best.overlap:.0%} overlap")
    elif best:
        reasons.append(f"Best candidate score {best.score:.3f} is below threshold {threshold:.3f}")
    else:
        reasons.append("No valid parallel-line pair was found in the crossing zone")
    if conflicting:
        reasons.append("Multiple crossing candidates have similar scores")

    displayed_pairs = ([best, *competing_pairs] if best else [])
    displayed_segments: list[Segment] = []
    for pair in displayed_pairs:
        for segment in (pair.first, pair.second):
            if segment not in displayed_segments:
                displayed_segments.append(segment)
    # When no valid pair exists, retain raw diagonal evidence for diagnosis.
    if not displayed_segments:
        displayed_segments = segments

    output_segments = [
        LineEvidence(
            id=index,
            x1=item.x1 + x1,
            y1=item.y1 + y1,
            x2=item.x2 + x1,
            y2=item.y2 + y1,
            angle_degrees=round(item.angle, 2),
            length=round(item.length, 2),
        )
        for index, item in enumerate(displayed_segments)
    ]
    result = AnalysisResult(
        crossing="Y" if crossing else "N",
        decision_basis=decision_basis,
        decision_status=decision_status,
        confidence=round(float(confidence), 3),
        needs_review=needs_review,
        review_reasons=review_reasons,
        image_quality=quality,
        region={"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
        detected_lines=output_segments,
        measurements=Measurements(
            mean_angle_degrees=round(normalize_angle((best.first.angle + best.second.angle) / 2), 2) if best else None,
            angle_difference_degrees=round(best.angle_difference, 2) if best else None,
            line_separation=round(best.separation, 2) if best else None,
            overlap_ratio=round(best.overlap, 3) if best else None,
            pair_score=round(best.score, 3) if best else 0.0,
        ),
        cheque_localization=normalized.localization,
        orientation=normalized.orientation,
        ac_payee=ac_payee,
        line_crossing=LineCrossingEvidence(
            detected=line_detected,
            confidence=round(float(line_confidence), 3),
        ),
        decision_reasons=reasons,
        diagnostics={
            **{key: round(value, 3) for key, value in quality_metrics.items()},
            **region_diagnostics,
            "candidate_pairs": len(pairs),
            "raw_candidate_pairs": len(raw_pairs),
            "boundary_artifact_pairs_rejected": len(all_raw_pairs) - len(non_boundary_pairs),
            "solid_ink_pairs_rejected": len(non_boundary_pairs) - len(raw_pairs),
            "competing_pairs": len(competing_pairs),
            "ocr_token_count": len(normalized.ocr_tokens),
            "ocr_retry_token_count": len(retry_tokens),
            "ocr_retry_used": ocr_retry_used,
            **({"ocr_error": normalized.ocr_error} if normalized.ocr_error else {}),
            **({"ocr_retry_error": ocr_retry_error} if ocr_retry_error else {}),
        },
    )
    annotated = annotate(image, result, best, (x1, y1, x2, y2))
    return result, annotated


def annotate(image: np.ndarray, result: AnalysisResult, best: Pair | None, region: tuple[int, int, int, int]) -> np.ndarray:
    """Overlay the search area, detected strokes, winning pair, and result."""
    annotated = image.copy()
    x1, y1, x2, y2 = region
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 170, 0), 3)
    for line in result.detected_lines:
        cv2.line(annotated, (line.x1, line.y1), (line.x2, line.y2), (170, 170, 170), 2)
    if best:
        for segment in (best.first, best.second):
            cv2.line(
                annotated,
                (segment.x1 + x1, segment.y1 + y1),
                (segment.x2 + x1, segment.y2 + y1),
                (0, 190, 0) if result.line_crossing.detected else (0, 165, 255),
                5,
            )
    if result.ac_payee.box:
        box = result.ac_payee.box
        cv2.rectangle(
            annotated,
            (box.x, box.y),
            (box.x + box.width, box.y + box.height),
            (180, 0, 255),
            4,
        )
        cv2.putText(
            annotated,
            "A/C Payee",
            (box.x, max(20, box.y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (130, 0, 210),
            2,
            cv2.LINE_AA,
        )
    label = (
        f"Crossing: {result.crossing}  A/C Payee: "
        f"{'Y' if result.ac_payee.status == 'detected' else 'N'}  "
        f"Lines: {'Y' if result.line_crossing.detected else 'N'}"
    )
    image_height = annotated.shape[0]
    label_top = max(0, image_height - 62)
    label_bottom = max(label_top + 1, image_height - 10)
    text_baseline = max(20, image_height - 25)
    cv2.rectangle(
        annotated,
        (10, label_top),
        (min(annotated.shape[1] - 10, 920), label_bottom),
        (255, 255, 255),
        -1,
    )
    cv2.putText(
        annotated,
        label,
        (20, text_baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return annotated

from __future__ import annotations

from copy import deepcopy
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Apply nested user overrides while retaining unspecified defaults."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load defaults, merge an optional YAML override, and validate it."""
    default_path = files("crossing_finder").joinpath("config/default.yaml")
    with default_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if path:
        custom_path = Path(path)
        if not custom_path.is_file():
            raise ConfigError(f"Configuration file not found: {custom_path}")
        with custom_path.open("r", encoding="utf-8") as handle:
            override = yaml.safe_load(handle) or {}
        if not isinstance(override, dict):
            raise ConfigError("Configuration root must be a mapping")
        config = _merge(config, override)
    _validate(config)
    return config


def _validate(config: dict[str, Any]) -> None:
    localization = config["localization"]
    if not (
        0 < localization["min_area_ratio"] < localization["max_area_ratio"] <= 1
    ):
        raise ConfigError("localization area ratios must be ordered values in (0, 1]")
    if not (
        1 < localization["min_aspect_ratio"]
        < localization["max_aspect_ratio"]
    ):
        raise ConfigError("localization aspect ratios must be ordered values above 1")
    if localization["max_candidates"] < 1:
        raise ConfigError("localization.max_candidates must be positive")
    for name in ("minimum_score", "ambiguity_margin", "duplicate_iou"):
        if not (0 <= localization[name] <= 1):
            raise ConfigError(f"localization.{name} must be in [0, 1]")
    if not (0 < localization["min_full_width_boundary_support"] <= 1):
        raise ConfigError("localization.min_full_width_boundary_support must be in (0, 1]")
    if not (0 <= localization["paper_max_saturation"] <= 255):
        raise ConfigError("localization.paper_max_saturation must be in [0, 255]")
    if not (0 <= localization["paper_min_brightness"] <= 255):
        raise ConfigError("localization.paper_min_brightness must be in [0, 255]")
    if not (0 < localization["paper_min_axis_support"] <= 1):
        raise ConfigError("localization.paper_min_axis_support must be in (0, 1]")
    if not (0 <= config["orientation"]["minimum_ocr_margin"] <= 1):
        raise ConfigError("orientation.minimum_ocr_margin must be in [0, 1]")
    text_detection = config["text_detection"]
    if not isinstance(text_detection["enabled"], bool):
        raise ConfigError("text_detection.enabled must be a boolean")
    if not (
        0 <= text_detection["min_ocr_confidence"]
        <= text_detection["automatic_accept_threshold"]
        <= 1
    ):
        raise ConfigError("text detection confidence thresholds must be ordered")
    if text_detection["ocr_retry_padding_pixels"] < 0:
        raise ConfigError("text_detection.ocr_retry_padding_pixels must be nonnegative")
    if any(
        text_detection[name] < 0
        for name in (
            "fuzzy_account_max_edits",
            "fuzzy_payee_max_edits",
            "only_suffix_max_edits",
            "ac_payee_max_edits",
            "ac_payee_only_max_edits",
            "account_payee_max_edits",
        )
    ):
        raise ConfigError("text detection edit limits must be nonnegative")

    region = config["region"]
    if not (0 <= region["x_start"] < region["x_end"] <= 1):
        raise ConfigError("region x coordinates must be ordered values in [0, 1]")
    if not (0 <= region["y_start"] < region["y_end"] <= 1):
        raise ConfigError("region y coordinates must be ordered values in [0, 1]")
    dynamic = region["dynamic"]
    if not isinstance(dynamic["enabled"], bool):
        raise ConfigError("region.dynamic.enabled must be a boolean")
    for name in (
        "horizontal_kernel_ratio",
        "min_horizontal_length_ratio",
        "max_horizontal_thickness_ratio",
    ):
        if not (0 < dynamic[name] <= 1):
            raise ConfigError(f"region.dynamic.{name} must be in (0, 1]")
    if not (0 <= dynamic["payee_min_y_ratio"] < dynamic["payee_max_y_ratio"] <= 1):
        raise ConfigError("dynamic payee y limits must be ordered ratios")
    if not (0 <= dynamic["date_min_x_ratio"] <= 1):
        raise ConfigError("region.dynamic.date_min_x_ratio must be in [0, 1]")
    if not (0 <= dynamic["x_margin_ratio"] < 1 and 0 <= dynamic["y_margin_ratio"] < 1):
        raise ConfigError("dynamic ROI margins must be ratios in [0, 1)")
    if not (region["x_start"] < dynamic["min_x_end_ratio"] <= dynamic["max_x_end_ratio"] <= 1):
        raise ConfigError("dynamic ROI x-end limits must be ordered ratios")
    if not (region["y_start"] < dynamic["min_y_end_ratio"] <= dynamic["max_y_end_ratio"] <= 1):
        raise ConfigError("dynamic ROI y-end limits must be ordered ratios")
    detection = config["detection"]
    for name in ("geometry_reference_width_ratio", "geometry_reference_height_ratio"):
        if not (0 < detection[name] <= 1):
            raise ConfigError(f"detection.{name} must be in (0, 1]")
    validation = detection["validation"]
    scoring = detection["scoring"]
    decision = detection["decision"]

    if not (
        0
        <= validation["min_pair_separation_ratio"]
        < validation["max_pair_separation_ratio"]
        <= 1
    ):
        raise ConfigError("pair separation ratios must be ordered values in [0, 1]")
    if not (0 <= validation["min_overlap_ratio"] <= 1):
        raise ConfigError("min_overlap_ratio must be in [0, 1]")
    if not (0 < validation["pair_gap_sample_width_ratio"] <= 1):
        raise ConfigError("pair_gap_sample_width_ratio must be in (0, 1]")
    if not (0 <= validation["pair_gap_dark_pixel_threshold"] <= 255):
        raise ConfigError("pair_gap_dark_pixel_threshold must be in [0, 255]")
    if not (0 <= validation["max_pair_gap_dark_ratio"] <= 1):
        raise ConfigError("max_pair_gap_dark_ratio must be in [0, 1]")
    for name in (
        "boundary_artifact_margin_ratio",
        "boundary_artifact_max_separation_ratio",
        "boundary_artifact_max_length_ratio",
    ):
        if not (0 <= validation[name] <= 1):
            raise ConfigError(f"{name} must be in [0, 1]")
    if not (0 <= validation["min_line_edge_support_ratio"] <= 1):
        raise ConfigError("min_line_edge_support_ratio must be in [0, 1]")
    if validation["max_angle_difference_degrees"] <= 0:
        raise ConfigError("max_angle_difference_degrees must be positive")
    if validation["same_stroke_max_angle_difference_degrees"] <= 0:
        raise ConfigError("same_stroke_max_angle_difference_degrees must be positive")
    if not (
        0
        <= validation["same_stroke_max_separation_ratio"]
        < validation["min_pair_separation_ratio"]
    ):
        raise ConfigError(
            "same_stroke_max_separation_ratio must be below min_pair_separation_ratio"
        )
    if not (0 <= validation["same_stroke_max_gap_ratio"] <= 1):
        raise ConfigError("same_stroke_max_gap_ratio must be in [0, 1]")

    weights = scoring["weights"]
    expected_weights = {"angle", "length", "overlap", "separation"}
    if set(weights) != expected_weights:
        raise ConfigError(f"scoring weights must contain exactly {sorted(expected_weights)}")
    if any(not isinstance(value, (int, float)) or value < 0 for value in weights.values()):
        raise ConfigError("scoring weights must be nonnegative numbers")
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ConfigError("scoring weights must add up to 1.0")
    for key in (
        "angle_zero_score_degrees",
        "full_length_ratio",
        "separation_zero_score_deviation_ratio",
    ):
        if scoring[key] <= 0:
            raise ConfigError(f"{key} must be positive")
    if not (0 <= scoring["ideal_separation_ratio"] <= 1):
        raise ConfigError("ideal_separation_ratio must be in [0, 1]")

    if not (
        0
        <= decision["positive_threshold"]
        <= decision["automatic_accept_threshold"]
        <= 1
    ):
        raise ConfigError("positive_threshold must not exceed automatic_accept_threshold")
    if not (0 <= decision["competing_pair_margin"] <= 1):
        raise ConfigError("competing_pair_margin must be in [0, 1]")
    if not (0 <= decision["same_candidate_center_distance_ratio"] <= 1):
        raise ConfigError("same_candidate_center_distance_ratio must be in [0, 1]")
    if not (0 < decision["same_candidate_max_angle_difference_degrees"] <= 90):
        raise ConfigError("same candidate angle difference must be in (0, 90]")

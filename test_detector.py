from __future__ import annotations

import pytest

from crossing_finder.config import ConfigError, load_config
from crossing_finder.detector import (
    ImageError,
    LayoutLine,
    Segment,
    _deduplicate,
    _edge_support,
    _find_ac_payee,
    _orient_candidate,
    _resolve_region,
    analyze_image,
    is_crossing_angle,
    normalize_angle,
    score_pair,
)
from crossing_finder.ocr import OCRToken


@pytest.mark.parametrize(
    ("angle", "expected"),
    [(45, True), (90, False), (135, True), (5, False), (-45, True), (0, False)],
)
def test_crossing_angle_filter(angle, expected):
    assert is_crossing_angle(angle, load_config()) is expected


def test_angle_normalization():
    assert normalize_angle(-45) == 135
    assert normalize_angle(225) == 45


def test_edge_support_rejects_a_line_bridged_across_large_gaps():
    import cv2
    import numpy as np

    edges = np.zeros((120, 220), dtype=np.uint8)
    cv2.line(edges, (10, 100), (90, 60), 255, 1)
    cv2.line(edges, (150, 30), (210, 0), 255, 1)
    continuous = Segment(10, 100, 90, 60, 26.57, 89.44)
    bridged = Segment(10, 100, 210, 0, 26.57, 223.61)

    assert _edge_support(continuous, edges) > 0.95
    assert _edge_support(bridged, edges) < 0.75


def test_dynamic_region_ends_before_payee_and_date_field_geometry():
    import numpy as np

    gray = np.full((600, 1400), 255, dtype=np.uint8)
    config = load_config()
    config["region"]["dynamic"]["enabled"] = True
    lines = [
        LayoutLine(620, 190, 400),
        LayoutLine(1050, 300, 180),
    ]
    region, diagnostics = _resolve_region(gray, config, lines)

    assert region == (0, 0, 606, 294)
    assert diagnostics["roi_x_source"] == "payee_field_geometry"
    assert diagnostics["roi_y_source"] == "date_field_geometry"


def test_dynamic_region_falls_back_without_anchor_labels():
    import numpy as np

    gray = np.full((600, 1400), 255, dtype=np.uint8)
    config = load_config()
    config["region"]["dynamic"]["enabled"] = True
    region, diagnostics = _resolve_region(
        gray, config, []
    )

    assert region == (0, 0, 1120, 240)
    assert diagnostics["roi_x_source"] == "configured_fallback"
    assert diagnostics["roi_y_source"] == "configured_fallback"


@pytest.mark.parametrize(
    "pattern",
    ["slash", "backslash"],
)
def test_detects_diagonal_crossing_marks(cheque_factory, pattern):
    result, annotated = analyze_image(cheque_factory(f"{pattern}.png", pattern), load_config())
    assert result.crossing == "Y"
    threshold = load_config()["detection"]["decision"]["positive_threshold"]
    assert result.measurements.pair_score >= threshold
    assert annotated.size > 0


@pytest.mark.parametrize("pattern", [None, "single", "nonparallel", "vertical"])
def test_rejects_negative_patterns(cheque_factory, pattern):
    result, _ = analyze_image(cheque_factory(f"negative-{pattern}.png", pattern), load_config())
    assert result.crossing == "N"


@pytest.mark.parametrize(
    ("filename", "expected"),
    [("1.jpg", "Y"), ("3.jpg", "Y"), ("4.jpg", "N"), ("5.jpg", "Y")],
)
def test_real_cheque_examples(filename, expected):
    result, _ = analyze_image(f"data/{filename}", load_config())
    assert result.crossing == expected


def test_overlapping_hough_segments_are_one_physical_stroke():
    result, _ = analyze_image("data/10.jpg", load_config())

    assert result.crossing == "Y"
    assert len(result.detected_lines) >= 2
    assert result.diagnostics["candidate_pairs"] >= 1
    assert "COMPETING_CANDIDATES" not in result.review_reasons


@pytest.mark.parametrize("filename", ["6.jpg", "7.jpg", "10.jpg"])
def test_hough_fragments_are_not_reported_as_competing_crossings(filename):
    result, _ = analyze_image(f"data/{filename}", load_config())

    assert result.crossing == "Y"
    assert "COMPETING_CANDIDATES" not in result.review_reasons
    assert result.diagnostics["competing_pairs"] == 0
    assert len(result.detected_lines) == 2


def test_low_quality_is_flagged(cheque_factory):
    result, _ = analyze_image(cheque_factory("blurred.png", "vertical", blur=True), load_config())
    assert result.image_quality == "low"
    assert result.needs_review is True
    assert "LOW_IMAGE_QUALITY" in result.review_reasons


@pytest.mark.parametrize(
    "text",
    [
        "A/C PAYEE",
        "AC PAYEE ONLY",
        "ACCOUNT PAYEE",
        "AVC PAYEE ONLY",
        "A..y'ayee Only",
    ],
)
def test_ac_payee_common_equivalents_are_detected(text):
    evidence = _find_ac_payee(
        [OCRToken(text=text, confidence=0.95, box=(20, 20, 220, 60))],
        (0, 0, 500, 200),
        load_config(),
        ocr_available=True,
    )

    assert evidence.status == "detected"
    assert evidence.box is not None


def test_plain_payee_label_is_not_crossing_instruction():
    evidence = _find_ac_payee(
        [OCRToken(text="PAYEE", confidence=0.99, box=(20, 20, 120, 60))],
        (0, 0, 500, 200),
        load_config(),
        ocr_available=True,
    )

    assert evidence.status == "not_found"


def test_not_found_ac_payee_exposes_ocr_candidates():
    evidence = _find_ac_payee(
        [OCRToken(text="ADCB", confidence=0.94, box=(20, 20, 120, 60))],
        (0, 0, 500, 200),
        load_config(),
        ocr_available=True,
    )

    assert evidence.status == "not_found"
    assert evidence.candidates[0].text == "ADCB"
    assert evidence.candidates[0].normalized_text == "ADCB"
    assert evidence.candidates[0].source == "primary"
    assert evidence.candidates[0].canonical_text == "ACPAYEE"
    assert evidence.candidates[0].canonical_length == 7
    assert evidence.candidates[0].eligible is False
    assert evidence.candidates[0].rejection_reason is not None


def test_ac_payee_candidate_exposes_match_scoring_math():
    evidence = _find_ac_payee(
        [OCRToken(text="A/C PAYE", confidence=0.95, box=(20, 20, 180, 60))],
        (0, 0, 500, 200),
        load_config(),
        ocr_available=True,
    )

    candidate = evidence.candidates[0]
    assert candidate.canonical_text == "ACPAYEE"
    assert candidate.edit_distance == 1
    assert candidate.allowed_edit_distance == 1
    assert candidate.canonical_length == 7
    assert candidate.lexical_score == 0.857
    assert candidate.match_confidence == 0.814
    assert candidate.match_method == "whole_phrase"
    assert candidate.eligible is True
    assert candidate.selected is True
    assert candidate.rejection_reason is None


def test_ac_payee_recognizes_fuzzy_only_suffix():
    evidence = _find_ac_payee(
        [OCRToken(text="AIC PAYEE DNILY", confidence=0.906, box=(20, 20, 240, 60))],
        (0, 0, 500, 200),
        load_config(),
        ocr_available=True,
    )

    candidate = evidence.candidates[0]
    assert evidence.status == "detected"
    assert candidate.canonical_text == "ACPAYEE"
    assert candidate.edit_distance == 1
    assert candidate.allowed_edit_distance == 2
    assert candidate.lexical_score == 0.857
    assert candidate.match_confidence == 0.777
    assert candidate.eligible is True
    assert candidate.selected is True


def test_payee_only_without_account_marker_is_rejected():
    evidence = _find_ac_payee(
        [OCRToken(text="PAYEE ONLY", confidence=0.99, box=(20, 20, 160, 60))],
        (0, 0, 500, 200),
        load_config(),
        ocr_available=True,
    )

    assert evidence.status == "not_found"


@pytest.mark.parametrize("text", ["AC PATE", "A/P PAEE", "AVC PAYE"])
def test_component_fuzzy_matching_requires_account_and_payee_parts(text):
    config = load_config()
    evidence = _find_ac_payee(
        [OCRToken(text=text, confidence=0.68, box=(20, 20, 190, 60))],
        (0, 0, 500, 200),
        config,
        ocr_available=True,
    )

    assert evidence.status == "detected"
    assert evidence.confidence < config["text_detection"]["automatic_accept_threshold"]


@pytest.mark.parametrize("text", ["PAYEE", "PAYEE ONLY", "ADCB", "ACCOUNT NUMBER"])
def test_component_fuzzy_matching_rejects_incomplete_phrases(text):
    evidence = _find_ac_payee(
        [OCRToken(text=text, confidence=0.99, box=(20, 20, 190, 60))],
        (0, 0, 500, 200),
        load_config(),
        ocr_available=True,
    )

    assert evidence.status == "not_found"


def test_ac_payee_outside_roi_is_ignored():
    evidence = _find_ac_payee(
        [OCRToken(text="A/C PAYEE", confidence=0.99, box=(600, 300, 800, 350))],
        (0, 0, 500, 200),
        load_config(),
        ocr_available=True,
    )

    assert evidence.status == "not_found"


@pytest.mark.parametrize(("confidence", "expected"), [(0.54, "not_found"), (0.55, "detected")])
def test_ac_payee_ocr_confidence_boundary(confidence, expected):
    evidence = _find_ac_payee(
        [OCRToken(text="A/C PAYEE", confidence=confidence, box=(20, 20, 220, 60))],
        (0, 0, 500, 200),
        load_config(),
        ocr_available=True,
    )

    assert evidence.status == expected


def test_text_signal_can_make_crossing_positive_without_lines(cheque_factory):
    def fake_ocr(_image):
        return [OCRToken(text="A/C PAYEE ONLY", confidence=0.96, box=(30, 30, 260, 75))]

    result, _ = analyze_image(
        cheque_factory("text-only.png"),
        load_config(),
        ocr_reader=fake_ocr,
    )

    assert result.crossing == "Y"
    assert result.decision_basis == "AC_PAYEE"
    assert result.ac_payee.status == "detected"
    assert result.line_crossing.detected is False


def test_text_and_line_signals_are_reported_independently(cheque_factory):
    calls = 0

    def fake_ocr(_image):
        nonlocal calls
        is_first_orientation = calls % 2 == 0
        calls += 1
        if is_first_orientation:
            return [OCRToken(text="A/C PAYEE ONLY", confidence=0.96, box=(30, 30, 260, 75))]
        return []

    result, _ = analyze_image(
        cheque_factory("text-and-lines.png", "slash"),
        load_config(),
        ocr_reader=fake_ocr,
    )

    assert result.crossing == "Y"
    assert result.decision_basis == "AC_PAYEE_AND_LINES"
    assert result.ac_payee.status == "detected"
    assert result.line_crossing.detected is True


def test_strong_lines_keep_overall_confidence_when_text_is_borderline(cheque_factory):
    calls = 0

    def fake_ocr(_image):
        nonlocal calls
        is_first_orientation = calls % 2 == 0
        calls += 1
        if is_first_orientation:
            return [OCRToken(text="A/C PAYEE", confidence=0.70, box=(30, 30, 230, 75))]
        return []

    result, _ = analyze_image(
        cheque_factory("borderline-text-with-lines.png", "slash"),
        load_config(),
        ocr_reader=fake_ocr,
    )

    assert result.decision_basis == "AC_PAYEE_AND_LINES"
    assert result.confidence == result.line_crossing.confidence
    assert "BORDERLINE_TEXT" in result.review_reasons


def test_locates_cheque_at_bottom_of_a4_page(cheque_factory, tmp_path):
    import cv2
    import numpy as np

    cheque = cv2.imread(str(cheque_factory("source.png", "slash")))
    cheque = cv2.resize(cheque, (1200, 514))
    page = np.full((2400, 1800, 3), 245, dtype=np.uint8)
    page[1800:2314, 300:1500] = cheque
    path = tmp_path / "a4-bottom.jpg"
    assert cv2.imwrite(str(path), page)

    result, annotated = analyze_image(path, load_config())

    assert result.cheque_localization.status == "located"
    assert result.cheque_localization.source_quad[0][1] > 1500
    assert result.cheque_localization.output_width == 1600
    assert annotated.shape[1] == 1600
    assert annotated.shape[0] < 900


@pytest.mark.parametrize(
    ("rotation_code", "expected_clockwise"),
    [
        ("counterclockwise", 90),
        ("clockwise", 270),
    ],
)
def test_portrait_orientation_chooses_readable_side(rotation_code, expected_clockwise):
    import cv2
    import numpy as np

    upright = np.full((240, 560, 3), 255, dtype=np.uint8)
    cv2.putText(upright, "PAY DATE", (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    code = (
        cv2.ROTATE_90_COUNTERCLOCKWISE
        if rotation_code == "counterclockwise"
        else cv2.ROTATE_90_CLOCKWISE
    )
    portrait = cv2.rotate(upright, code)

    def readable_only_when_text_is_at_top(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        midpoint = gray.shape[0] // 2
        if np.mean(gray[:midpoint] < 100) > np.mean(gray[midpoint:] < 100):
            return [OCRToken(text="PAY DATE", confidence=0.98, box=(20, 20, 180, 65))]
        return []

    _, _, orientation, _, _ = _orient_candidate(
        portrait,
        load_config(),
        readable_only_when_text_is_at_top,
    )

    assert orientation.method == "ocr"
    assert orientation.rotation_degrees_clockwise == expected_clockwise


def test_orientation_uses_expected_cheque_field_positions_before_ink_density():
    import cv2
    import numpy as np

    # A dense header at the top deliberately makes the old raw-density
    # fallback prefer the upside-down image, reproducing image 12's failure.
    upright = np.full((400, 900, 3), 255, dtype=np.uint8)
    upright[:100] = 80
    calls = 0

    def same_text_but_position_sensitive(_image):
        nonlocal calls
        first_orientation = calls == 0
        calls += 1
        if first_orientation:
            return [
                OCRToken(text="DATE", confidence=0.92, box=(700, 35, 820, 70)),
                OCRToken(text="SIGNATURE", confidence=0.90, box=(650, 305, 850, 345)),
            ]
        return [
            OCRToken(text="SIGNATURE", confidence=0.90, box=(50, 55, 250, 95)),
            OCRToken(text="DATE", confidence=0.92, box=(80, 330, 200, 365)),
        ]

    _, _, orientation, _, _ = _orient_candidate(
        upright,
        load_config(),
        same_text_but_position_sensitive,
    )

    assert orientation.method == "ocr"
    assert orientation.rotation_degrees_clockwise == 0


def test_rejects_page_without_a_cheque(tmp_path):
    import cv2
    import numpy as np

    page = np.full((1600, 1130, 3), 255, dtype=np.uint8)
    path = tmp_path / "blank-a4.png"
    assert cv2.imwrite(str(path), page)

    with pytest.raises(ImageError, match="cheque could not be located"):
        analyze_image(path, load_config())


def test_proposes_full_width_cheque_above_photo_background():
    import cv2
    import numpy as np

    from crossing_finder.detector import _candidate_quads, _quad_bbox

    image = np.full((450, 600, 3), 235, dtype=np.uint8)
    image[:280] = 245
    cv2.rectangle(image, (0, 0), (599, 279), (80, 80, 80), 3)
    for y in (60, 120, 190, 240):
        cv2.line(image, (20, y), (575, y), (120, 120, 120), 2)
    # Non-cheque photographic clutter below the long lower cheque edge.
    rng = np.random.default_rng(7)
    image[285:] = rng.integers(20, 190, size=image[285:].shape, dtype=np.uint8)

    candidates = _candidate_quads(image, load_config())
    boxes = [_quad_bbox(candidate.quad) for candidate in candidates]

    assert any(x1 <= 5 and x2 >= 597 and 250 <= y2 <= 300 for x1, y1, x2, y2 in boxes)


@pytest.mark.parametrize("filename", ["a.png", "c.png", "d.png"])
def test_localizes_faint_skewed_and_dark_page_scans(filename):
    result, annotated = analyze_image(f"data/{filename}", load_config())

    assert result.cheque_localization.status == "located"
    assert result.cheque_localization.output_width == 1600
    assert annotated.shape[1] > annotated.shape[0]


def test_faint_crossing_lines_survive_page_localization():
    result, _ = analyze_image("data/b.png", load_config())

    assert result.cheque_localization.status == "located"
    assert result.crossing == "Y"
    assert result.decision_basis == "LINES"
    assert result.line_crossing.detected is True


def test_skewed_crop_boundary_is_not_a_crossing():
    result, _ = analyze_image("data/30.jpeg", load_config())

    assert result.cheque_localization.status == "located"
    assert result.crossing == "N"


def test_real_crossing_near_skewed_crop_boundary_survives():
    result, _ = analyze_image("data/43.jpeg", load_config())

    assert result.crossing == "Y"
    assert result.line_crossing.detected is True


def test_applies_exif_orientation_before_localization(cheque_factory, tmp_path):
    from PIL import Image

    source_path = cheque_factory("exif-source.png", "slash")
    output_path = tmp_path / "exif-rotated.jpg"
    with Image.open(source_path) as image:
        exif = Image.Exif()
        exif[274] = 6  # Display the stored landscape pixels 90 degrees clockwise.
        image.convert("RGB").save(output_path, exif=exif)

    result, annotated = analyze_image(output_path, load_config())

    assert result.orientation.rotation_degrees_clockwise == 90
    assert max(point[1] for point in result.cheque_localization.source_quad) > 1300
    assert annotated.shape[1] > annotated.shape[0]


def test_angle_validation_limit_does_not_change_valid_pair_score():
    config = load_config()
    first = Segment(20, 100, 120, 0, 45.0, 141.42)
    second = Segment(45, 105, 145, 5, 50.0, 141.42)
    original = score_pair(first, second, 400, 400, config)
    assert original is not None

    config["detection"]["validation"]["max_angle_difference_degrees"] = 30.0
    permissive = score_pair(first, second, 400, 400, config)
    assert permissive is not None
    assert permissive.score == pytest.approx(original.score)


def test_scoring_weights_must_add_up_to_one(tmp_path):
    override = tmp_path / "invalid.yaml"
    override.write_text("detection:\n  scoring:\n    weights:\n      angle: 0.9\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="add up to 1.0"):
        load_config(override)


def test_deduplicate_collapses_fragments_of_one_physical_stroke():
    config = load_config()
    segments = [
        Segment(0, 200, 200, 0, 45.0, 282.84),
        # Same stroke, but its midpoint is far enough away that the previous
        # midpoint-only deduplication treated it as another physical line.
        Segment(100, 100, 250, -50, 45.5, 212.13),
        # A genuinely separate parallel stroke must remain available for pairing.
        Segment(0, 240, 200, 40, 45.0, 282.84),
    ]

    retained = _deduplicate(segments, roi_diagonal=500.0, config=config)

    assert len(retained) == 2
    assert segments[0] in retained
    assert segments[2] in retained


def test_rejects_unsupported_and_small_images(tmp_path, cheque_factory):
    wrong = tmp_path / "image.gif"
    wrong.write_bytes(b"not an image")
    with pytest.raises(ImageError, match="Unsupported"):
        analyze_image(wrong, load_config())

    import cv2
    import numpy as np

    small = tmp_path / "small.png"
    cv2.imwrite(str(small), np.zeros((50, 100, 3), dtype=np.uint8))
    with pytest.raises(ImageError, match="too small"):
        analyze_image(small, load_config())

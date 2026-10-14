Inspect my existing cheque crossing detection project and implement an evaluation pipeline using pandas and scikit-learn. My detector has three outputs, OCR text detection, OpenCV crossing line detection, and final crossed or not crossed decision. Create a CSV with seven columns: file name, actual crossed, actual text, actual lines, predicted crossed, predicted text, predicted lines. The three actual columns are manually labeled by ground truths. Implement three evaluations: first, OCR by comparing actual text versus predicted text, second line detection by comparing actual lines versus predicted lines, and third, combined by comparing actual crossed versus predicted crossed. For each, compute accuracy, precision, recall, F1, specificity, and confusion matrix, and output a report comparing all three. Require Codex to inspect existing code, reuse your OCR and OpenCV functions, preserve current behavior, handle errors, export results, generate matrix visualizations, add tests, and document uv run instructions.

default:
orientation:
  minimum_ocr_margin: 0.06 # Required score gap before OCR controls 0/180 orientation.
18:
  paper_max_saturation: 50 # Upper HSV saturation used to distinguish cheque paper from its surroundings.
  paper_min_brightness: 100 # Lower HSV brightness used for the paper-support proposal.
  paper_min_axis_support: 0.25 # Required paper fraction along a candidate row or column.


config 62
    if not (0 <= localization["paper_max_saturation"] <= 255):
        raise ConfigError("localization.paper_max_saturation must be in [0, 255]")
    if not (0 <= localization["paper_min_brightness"] <= 255):
        raise ConfigError("localization.paper_min_brightness must be in [0, 255]")
    if not (0 < localization["paper_min_axis_support"] <= 1):
        raise ConfigError("localization.paper_min_axis_support must be in (0, 1]")

detector 316


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


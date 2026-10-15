Refactor this Python cheque-crossing detection repository into a clean, scalable structure.

Work autonomously from inspection through implementation and verification. Do not stop after producing a plan.

Project context:
- The system analyzes cheque images.
- It normalizes the image and identifies the cheque ROI.
- It detects pairs of crossing lines using OpenCV.
- It uses OCR to detect phrases such as “A/C Payee”.
- It produces Y/N crossing decisions, confidence scores, review decisions, annotations, and diagnostic details.
- It includes a CLI, Streamlit UI, YAML configuration, optional OpenRouter summarization, and an evaluation system.
- This is deterministic image-processing and OCR logic; there is currently no trained ML model.

First inspect:
- AGENTS.md or other repository instructions
- git status
- pyproject.toml
- the current crossing_finder package
- tests
- configuration files
- evaluation files
- .gitignore

Important constraints:
- Preserve all existing user changes in the dirty worktree.
- Never discard, reset, or overwrite unrelated changes.
- Do not commit anything unless explicitly asked.
- Do not access, move, delete, or commit private cheque images.
- Keep data/, output/, evaluation/dataset/, evaluation/reports/, evaluation/labels.csv, .env, and generated artifacts ignored by Git.
- Do not change detection thresholds, scoring formulas, OCR rules, decision logic, output schemas, or user-visible behavior as part of this refactor.
- Do not add a machine-learning training system.
- Do not attempt to fix existing detector accuracy failures unless a structural change causes a new failure.
- Keep comments short and useful. Explain reasoning, not obvious syntax.
- Avoid splitting code into tiny modules without a clear responsibility.

Target structure:

crossing_finder/
├── pyproject.toml
├── README.md
├── PRODUCT.md
├── .gitignore
├── src/
│   └── crossing_finder/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── models.py
│       ├── exceptions.py
│       │
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── analyzer.py
│       │   ├── normalization.py
│       │   ├── localization.py
│       │   ├── orientation.py
│       │   ├── quality.py
│       │   └── annotation.py
│       │
│       ├── crossing/
│       │   ├── __init__.py
│       │   ├── segments.py
│       │   ├── deduplication.py
│       │   ├── pairing.py
│       │   ├── scoring.py
│       │   └── decision.py
│       │
│       ├── text_detection/
│       │   ├── __init__.py
│       │   ├── engine.py
│       │   ├── normalization.py
│       │   ├── candidates.py
│       │   ├── matching.py
│       │   └── decision.py
│       │
│       ├── summarization/
│       │   ├── __init__.py
│       │   ├── openrouter.py
│       │   └── fallback.py
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── cli.py
│       │   ├── runner.py
│       │   ├── labels.py
│       │   ├── metrics.py
│       │   └── reports.py
│       │
│       ├── interfaces/
│       │   ├── __init__.py
│       │   └── streamlit_app.py
│       │
│       └── config/
│           ├── __init__.py
│           ├── loader.py
│           ├── schema.py
│           └── default.yaml
│
├── configs/
│   ├── development.yaml
│   ├── validation.yaml
│   └── production.yaml
│
├── evaluation/
│   ├── README.md
│   ├── labels.example.csv
│   ├── labels.csv
│   ├── dataset/
│   │   ├── tuning/
│   │   ├── validation/
│   │   └── test/
│   └── reports/
│
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_normalization.py
│   │   ├── test_line_scoring.py
│   │   ├── test_line_pairing.py
│   │   ├── test_text_matching.py
│   │   └── test_metrics.py
│   ├── integration/
│   │   ├── test_analyzer.py
│   │   ├── test_cli.py
│   │   └── test_evaluation.py
│   └── regression/
│       └── test_known_cases.py
│
├── scripts/
└── docs/
    ├── architecture.md
    ├── configuration.md
    └── evaluation.md

Treat this as a target architecture, not permission to create empty or artificial modules. Only create a module when current code provides a coherent responsibility for it.

Responsibilities:
- pipeline/analyzer.py: coordinate the complete analysis workflow.
- pipeline/normalization.py: resizing, grayscale conversion, contrast, and blur preparation.
- pipeline/localization.py: cheque localization and ROI extraction.
- pipeline/orientation.py: orientation detection and correction.
- pipeline/quality.py: image-quality measurements and warnings.
- pipeline/annotation.py: diagnostic overlays and annotated output.
- crossing/segments.py: Canny/Hough segment detection.
- crossing/deduplication.py: merge or suppress duplicate representations of the same physical line.
- crossing/pairing.py: generate and validate candidate line pairs.
- crossing/scoring.py: angle, length, overlap, separation, and final weighted score.
- crossing/decision.py: crossing Y/N and review decisions based on line evidence.
- text_detection/: OCR execution, text normalization, candidate extraction, phrase matching, and OCR-based decisions.
- evaluation/: label loading, evaluation execution, metrics, and report generation.
- models.py: shared public data models without pipeline implementation.
- config/: typed schema, loading, validation, defaults, and clear error messages.

Implementation order:
1. Run the current test suite and record the baseline.
2. Introduce the src layout and update packaging.
3. Move and validate configuration loading.
4. Extract OCR into text_detection.
5. Extract line processing into crossing.
6. Extract pipeline stages and reduce the old detector module to orchestration or a compatibility facade.
7. Move evaluation implementation into the package.
8. Reorganize tests without losing coverage.
9. Add architecture, configuration, and evaluation documentation.
10. Run the complete verification suite again.

Compatibility requirements:
- Preserve public models and serialized JSON fields.
- Preserve configuration key names and meanings.
- Preserve existing imports where practical using compatibility re-exports.
- Preserve `python -m crossing_finder`.
- Preserve the `crossing-finder` command.
- Preserve the `crossing-evaluate` command.
- Add a stable way to launch Streamlit, preferably a `crossing-finder-ui` console command.
- Ensure default.yaml is included as package data in built distributions.
- Ensure editable and normal package installations work.
- Avoid circular imports by keeping shared data models independent from implementations.

Evaluation requirements:
- Continue evaluating final crossing classification.
- Continue evaluating line-crossing classification separately.
- Continue evaluating A/C Payee classification separately.
- Retain OCR exact-match and character-error-rate measurements.
- Retain decision-basis and human-review metrics.
- Keep the existing labels schema compatible:
  file_name,crossing,ac_payee,line_crossing,ac_payee_text,bank,difficulty,quality,group_id,split,notes
- Do not upload datasets or require Hugging Face access to run local evaluation.
- Dataset loading must support tuning, validation, and test splits.
- Prevent leakage by documenting that related images should share group_id and remain in one split.

Testing requirements:
- Add focused unit tests around extracted pure functions.
- Preserve current integration and regression tests.
- Compare the final test result with the recorded baseline.
- Do not hide failures with broad exception handling, skipped tests, or weakened assertions.
- The repository previously had known image-level failures involving 5.jpg and 43.jpeg. Confirm the current baseline yourself. The refactor must not introduce additional failures or silently alter those outcomes.
- Test CLI entry points and configuration resource loading after installation.
- Run an evaluation smoke test with temporary synthetic data if private evaluation images are unavailable.

Documentation:
- Update README.md with installation and run commands.
- Document how the pipeline flows from image input to normalization, ROI, OCR/line evidence, scoring, decision, human review, and reporting.
- Document every configuration group and distinguish:
  - candidate filtering thresholds,
  - pair-scoring weights,
  - final decision thresholds,
  - human-review thresholds.
- Document local evaluation without exposing private images or labels.

Acceptance criteria:
- The package imports successfully from the src layout.
- Existing public commands continue to work.
- Streamlit starts through a documented stable command.
- Configuration works both from source and from an installed wheel.
- Evaluation works locally and produces machine-readable and human-readable reports.
- No private data or generated output becomes tracked.
- No detection behavior changes unless required to repair a regression introduced by the refactor.
- The final test results have no new failures compared with baseline.
- There are no empty placeholder modules.

At completion, report:
- The resulting structure.
- The main responsibility moved into each package.
- Compatibility shims retained.
- Commands run and their results.
- Baseline versus final test results.
- Any unresolved risks or intentionally deferred cleanup.

Start by inspecting the repository and establishing the test baseline, then implement the refactor incrementally.
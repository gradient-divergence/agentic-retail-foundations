# Shelf fixture contract

Both shelf implementations use an explicit fixture interface: float32 RGB NHWC
input `[1, 640, 640, 3]`, values in `[0,1]`, and dictionary outputs
`detection_boxes [1,N,4]`, `detection_classes [1,N]`, and `detection_scores [1,N]`.
Boxes are normalized `[ymin,xmin,ymax,xmax]`. Class 1 maps to S4 and class 2 to S5
in the dummy planogram catalog. Scores are uncalibrated detector scores.
Same-SKU boxes with intersection over union above 0.5 are suppressed, retaining
higher scores. These fixture thresholds are illustrative.

The executable TorchScript fixture is built by
`tests/agents/test_cv_regressions.py::test_torchscript_fixture_runs_frame_to_catalog_and_planogram`.
It exercises one frame through preprocessing, artifact loading, decoding, SKU
mapping, and comparison. It is a deterministic fixture, not a trained detector.
A deployment must supply a compatible detector and validate it on its own images.

Coordinates assume a rectified, current, section-specific frame whose entire
image covers the same normalized shelf region as the planogram. The demo reads
one frame from the configured camera; it has no capture timestamp verification.
Validate capture freshness and camera calibration upstream before using it for
operational decisions. An absent accepted detection is an observation gap, not
proof that a shelf or authoritative inventory is empty.

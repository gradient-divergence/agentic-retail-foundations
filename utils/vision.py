"""One-to-one placement accounting for a rectified shelf image."""

from math import hypot


def compare_placements(detections: list[dict], planogram: dict, tolerance: float = 0.15) -> list[dict]:
    """Assign each visible instance once to its nearest placement of the same SKU."""
    expected = planogram["products"]
    counts = [0] * len(expected)
    issues = []
    for product in detections:
        candidates = [
            (
                hypot(
                    p["position"]["x"] - product["shelf_position"]["x"],
                    p["position"]["y"] - product["shelf_position"]["y"],
                ),
                i,
            )
            for i, p in enumerate(expected)
            if p["product_id"] == product["product_id"]
        ]
        if not candidates:
            issues.append(
                dict(
                    type="UNEXPECTED_PRODUCT",
                    product_id=product["product_id"],
                    count=1,
                    positions=[product["shelf_position"]],
                )
            )
            continue
        distance, index = min(candidates)
        counts[index] += 1
        if distance > tolerance:
            issues.append(
                dict(
                    type="MISPLACED_PRODUCT",
                    product_id=product["product_id"],
                    expected_position=expected[index]["position"],
                    actual_position=product["shelf_position"],
                    distance=distance,
                )
            )
    for p, count in zip(expected, counts, strict=True):
        if count < p["expected_count"]:
            issues.append(
                dict(
                    type="OUT_OF_STOCK" if count == 0 else "LOW_STOCK",
                    product_id=p["product_id"],
                    expected_count=p["expected_count"],
                    actual_count=count,
                    position=p["position"],
                    gap_percentage=(p["expected_count"] - count) / p["expected_count"],
                )
            )
    return issues


def box_overlap(left: list[int], right: list[int]) -> float:
    """Intersection over union of [ymin, xmin, ymax, xmax] pixel boxes."""
    intersection = max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0, min(left[3], right[3]) - max(left[1], right[1])
    )
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in (left, right)]
    union = sum(areas) - intersection
    return intersection / union if union > 0 else 0.0

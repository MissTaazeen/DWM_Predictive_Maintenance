"""
Predictive Maintenance using Gower-style Similarity and k-Nearest Neighbors.
Dataset is synthetic and intended only for educational demonstration.
"""
from __future__ import annotations
import csv
import math
from collections import Counter
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "cnc_machines.csv"

# Attribute metadata: type and, for ordinal features, ordered categories.
FEATURES = {
    "machine_type": {"type": "nominal"},
    "coolant_leak": {"type": "binary"},
    "vibration_level": {"type": "ordinal", "order": ["Low", "Medium", "High"]},
    "temperature_c": {"type": "numeric"},
    "spindle_rpm": {"type": "numeric"},
    "hours_since_service": {"type": "numeric"},
}
TARGET = "condition"


def load_dataset(path: Path = DATA_PATH):
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, meta in FEATURES.items():
            if meta["type"] == "numeric":
                row[key] = float(row[key])
            elif meta["type"] == "binary":
                row[key] = int(row[key])
    return rows


def numeric_ranges(rows):
    ranges = {}
    for feature, meta in FEATURES.items():
        if meta["type"] == "numeric":
            values = [float(r[feature]) for r in rows]
            ranges[feature] = max(values) - min(values)
    return ranges


def attribute_similarity(a, b, feature, meta, ranges):
    kind = meta["type"]
    if kind == "nominal":
        return 1.0 if a == b else 0.0
    if kind == "binary":
        return 1.0 if int(a) == int(b) else 0.0
    if kind == "ordinal":
        order = meta["order"]
        span = max(1, len(order) - 1)
        return 1.0 - abs(order.index(a) - order.index(b)) / span
    if kind == "numeric":
        value_range = ranges[feature]
        if value_range == 0:
            return 1.0
        return max(0.0, 1.0 - abs(float(a) - float(b)) / value_range)
    raise ValueError(f"Unsupported attribute type: {kind}")


def gower_similarity(a, b, rows):
    """Mean attribute similarity across the configured mixed-type features."""
    ranges = numeric_ranges(rows)
    similarities = [
        attribute_similarity(a[f], b[f], f, meta, ranges)
        for f, meta in FEATURES.items()
    ]
    return sum(similarities) / len(similarities)


def predict_knn(query, training_rows, k=3):
    """Predict class by majority vote among k most similar training records."""
    if not training_rows:
        raise ValueError("Training dataset cannot be empty.")
    if k < 1:
        raise ValueError("k must be at least 1.")
    k = min(k, len(training_rows))
    ranked = sorted(
        ((gower_similarity(query, row, training_rows), row) for row in training_rows),
        key=lambda item: item[0],
        reverse=True,
    )
    neighbours = ranked[:k]
    votes = Counter(row[TARGET] for _, row in neighbours)
    # Deterministic tie-break: highest vote, then highest mean neighbour similarity.
    class_similarity = {}
    for sim, row in neighbours:
        class_similarity[row[TARGET]] = class_similarity.get(row[TARGET], 0.0) + sim
    predicted = sorted(votes, key=lambda c: (votes[c], class_similarity[c], c), reverse=True)[0]
    return predicted, neighbours


def train_test_split(rows, test_fraction=0.25):
    """Deterministic split preserving row order for reproducible demonstration."""
    split = max(1, int(len(rows) * (1 - test_fraction)))
    return rows[:split], rows[split:]


def evaluate(training_rows, test_rows, k=3):
    if not test_rows:
        return {"accuracy": 0.0, "correct": 0, "total": 0, "predictions": []}
    predictions = []
    correct = 0
    for row in test_rows:
        query = {f: row[f] for f in FEATURES}
        predicted, _ = predict_knn(query, training_rows, k)
        actual = row[TARGET]
        is_correct = predicted == actual
        correct += int(is_correct)
        predictions.append((actual, predicted, is_correct))
    return {
        "accuracy": correct / len(test_rows),
        "correct": correct,
        "total": len(test_rows),
        "predictions": predictions,
    }


def main():
    rows = load_dataset()
    training, test = train_test_split(rows, test_fraction=0.25)
    results = evaluate(training, test, k=3)

    print("=" * 62)
    print("PREDICTIVE MAINTENANCE: GOWER-STYLE SIMILARITY + k-NN")
    print("=" * 62)
    print(f"Dataset records: {len(rows)}")
    print(f"Training records: {len(training)} | Test records: {len(test)}")
    print("Feature types: nominal, binary, ordinal, numeric")
    print(f"k = 3")
    print(f"Test accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")
    print("\nTest predictions:")
    for i, (actual, predicted, ok) in enumerate(results["predictions"], start=1):
        print(f"{i:02d}. Actual={actual:<7} Predicted={predicted:<7} {'PASS' if ok else 'MISS'}")

    new_machine = {
        "machine_type": "CNC_Mill",
        "coolant_leak": 1,
        "vibration_level": "High",
        "temperature_c": 91.0,
        "spindle_rpm": 1200.0,
        "hours_since_service": 680.0,
    }
    prediction, neighbours = predict_knn(new_machine, rows, k=3)
    print("\nNew machine assessment:")
    for key, value in new_machine.items():
        print(f"  {key}: {value}")
    print(f"Predicted condition: {prediction}")
    print("Top 3 similar machines:")
    for rank, (similarity, row) in enumerate(neighbours, start=1):
        print(f"  {rank}. ID={row['machine_id']} | condition={row[TARGET]} | similarity={similarity:.3f}")

    print("\nNote: Results use a small synthetic dataset for academic demonstration.")
    print("This prototype is not validated for real industrial maintenance decisions.")


if __name__ == "__main__":
    main()

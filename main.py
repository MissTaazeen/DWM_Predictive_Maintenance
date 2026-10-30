
"""
Predictive Maintenance using Gower-style Similarity and k-Nearest Neighbors.

Dataset: AI4I 2020 Predictive Maintenance Dataset
Target: Machine failure (0 = Normal, 1 = Failure)

AI4I is a synthetic dataset intended for educational and
research purposes. This model is not validated for real
industrial maintenance decisions.
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_PATH = Path(__file__).parent / "data" / "ai4i2020.csv"

TARGET = "Machine failure"

NOMINAL_FEATURES = ["Type"]

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

FEATURES = NOMINAL_FEATURES + NUMERIC_FEATURES

K = 5
TEST_FRACTION = 0.25
RANDOM_SEED = 42


# --------------------------------------------------
# 1. LOAD AND PREPROCESS DATA
# --------------------------------------------------

def load_dataset(path=DATA_PATH):
    """Load AI4I data and retain only valid model columns."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    required_columns = ["UDI", "Product ID"] + FEATURES + [TARGET]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Remove rows with missing model values, if any.
    df = df[required_columns].dropna().copy()

    # Convert target to integer labels.
    df[TARGET] = df[TARGET].astype(int)

    # Validate binary target.
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")

    return df.to_dict(orient="records")


# --------------------------------------------------
# 2. STRATIFIED TRAIN-TEST SPLIT
# --------------------------------------------------

def train_test_split(rows, test_fraction=TEST_FRACTION):
    """
    Split each target class separately to preserve
    approximately the same class proportions.
    """

    rng = random.Random(RANDOM_SEED)

    normal = [r for r in rows if r[TARGET] == 0]
    failure = [r for r in rows if r[TARGET] == 1]

    rng.shuffle(normal)
    rng.shuffle(failure)

    normal_test_count = round(len(normal) * test_fraction)
    failure_test_count = round(len(failure) * test_fraction)

    test_rows = (
        normal[:normal_test_count]
        + failure[:failure_test_count]
    )

    training_rows = (
        normal[normal_test_count:]
        + failure[failure_test_count:]
    )

    rng.shuffle(training_rows)
    rng.shuffle(test_rows)

    return training_rows, test_rows


# --------------------------------------------------
# 3. PREPARE TRAINING DATA
# --------------------------------------------------

def prepare_training_data(training_rows):
    """
    Convert training features into NumPy arrays.
    Numeric ranges are calculated from training data only.
    """

    numeric_data = np.array(
        [
            [float(row[f]) for f in NUMERIC_FEATURES]
            for row in training_rows
        ],
        dtype=float,
    )

    numeric_min = numeric_data.min(axis=0)
    numeric_max = numeric_data.max(axis=0)
    numeric_ranges = numeric_max - numeric_min

    nominal_data = np.array(
        [row["Type"] for row in training_rows],
        dtype=str,
    )

    labels = np.array(
        [row[TARGET] for row in training_rows],
        dtype=int,
    )

    return {
        "numeric_data": numeric_data,
        "numeric_ranges": numeric_ranges,
        "nominal_data": nominal_data,
        "labels": labels,
        "rows": training_rows,
    }


# --------------------------------------------------
# 4. GOWER-STYLE SIMILARITY
# --------------------------------------------------

def calculate_similarities(query, prepared):
    """
    Calculate similarity between one query machine
    and every training record.

    Nominal feature:
        Similarity = 1 if equal, otherwise 0.

    Numeric feature:
        Similarity = 1 - absolute difference / range.

    Final similarity:
        Mean of all feature similarities.
    """

    training_numeric = prepared["numeric_data"]
    numeric_ranges = prepared["numeric_ranges"]

    query_numeric = np.array(
        [float(query[f]) for f in NUMERIC_FEATURES],
        dtype=float,
    )

    # Numeric similarities
    differences = np.abs(training_numeric - query_numeric)

    numeric_similarities = np.ones_like(differences)

    valid_ranges = numeric_ranges > 0

    numeric_similarities[:, valid_ranges] = (
        1 - differences[:, valid_ranges]
        / numeric_ranges[valid_ranges]
    )

    numeric_similarities = np.clip(
        numeric_similarities, 0, 1
    )

    # Nominal similarity: Type
    query_type = str(query["Type"])

    nominal_similarity = (
        prepared["nominal_data"] == query_type
    ).astype(float)

    # Combine nominal and numeric similarities.
    all_similarities = np.column_stack(
        [nominal_similarity, numeric_similarities]
    )

    return all_similarities.mean(axis=1)


# --------------------------------------------------
# 5. k-NEAREST NEIGHBOUR CLASSIFICATION
# --------------------------------------------------


def predict_knn(query, prepared, k=K, voting="majority"):
    """
    Predict machine failure using one of three voting methods:

    majority:
        Each neighbour gets one vote.

    weighted:
        Each neighbour's vote is weighted by similarity.

    balanced:
        Similarity-weighted voting with inverse-frequency
        class weights to compensate for class imbalance.
    """

    if k < 1:
        raise ValueError("k must be at least 1.")

    if voting not in {"majority", "weighted", "balanced"}:
        raise ValueError("Unknown voting method.")

    similarities = calculate_similarities(query, prepared)

    k = min(k, len(similarities))

    candidate_indices = np.argpartition(
        similarities, -k
    )[-k:]

    neighbour_indices = sorted(
        candidate_indices,
        key=lambda i: similarities[i],
        reverse=True,
    )

    neighbour_labels = prepared["labels"][neighbour_indices]
    neighbour_similarities = similarities[neighbour_indices]

    scores = {0: 0.0, 1: 0.0}

    # Class frequencies in training data
    class_counts = {
        0: int(np.sum(prepared["labels"] == 0)),
        1: int(np.sum(prepared["labels"] == 1)),
    }

    total = len(prepared["labels"])

    for label, similarity in zip(
        neighbour_labels, neighbour_similarities
    ):
        label = int(label)

        if voting == "majority":
            weight = 1.0

        elif voting == "weighted":
            weight = float(similarity)

        else:
            # Inverse-frequency class weighting
            class_weight = total / (2 * class_counts[label])
            weight = float(similarity) * class_weight

        scores[label] += weight

    prediction = max(
        [0, 1],
        key=lambda label: (scores[label], label),
    )

    neighbours = [
        {
            "UDI": prepared["rows"][i].get("UDI"),
            "Product ID": prepared["rows"][i].get("Product ID"),
            "actual": int(prepared["labels"][i]),
            "similarity": float(similarities[i]),
        }
        for i in neighbour_indices
    ]

    return prediction, neighbours


# --------------------------------------------------
# 6. MODEL EVALUATION
# --------------------------------------------------


def evaluate(training_rows, test_rows, k=K, voting="majority"):
    """Evaluate one voting method using binary classification metrics."""

    prepared = prepare_training_data(training_rows)

    actual_values = []
    predicted_values = []

    for row in test_rows:
        query = {f: row[f] for f in FEATURES}

        prediction, _ = predict_knn(
            query, prepared, k=k, voting=voting
        )

        actual_values.append(int(row[TARGET]))
        predicted_values.append(int(prediction))

    actual = np.array(actual_values)
    predicted = np.array(predicted_values)

    tn = int(np.sum((actual == 0) & (predicted == 0)))
    fp = int(np.sum((actual == 0) & (predicted == 1)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))
    tp = int(np.sum((actual == 1) & (predicted == 1)))

    accuracy = (
        (tp + tn) / len(actual) if len(actual) else 0
    )

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0
    )

    normal_recall = (
        tn / (tn + fp) if (tn + fp) else 0
    )

    balanced_accuracy = (normal_recall + recall) / 2

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }



# --------------------------------------------------
# 7. DISPLAY RESULTS
# --------------------------------------------------


def main():

    print("=" * 75)
    print("PREDICTIVE MAINTENANCE: GOWER SIMILARITY + k-NN")
    print("=" * 75)

    rows = load_dataset()
    training_rows, test_rows = train_test_split(rows)

    print("\nDATASET INFORMATION")
    print("-" * 75)
    print(f"Total records    : {len(rows)}")
    print(f"Training records : {len(training_rows)}")
    print(f"Testing records  : {len(test_rows)}")

    print("\nFEATURES USED")
    print("-" * 75)
    for feature in FEATURES:
        print(f"- {feature}")

    experiments = [
        ("A", "Standard Majority Voting", "majority"),
        ("B", "Similarity-Weighted Voting", "weighted"),
        ("C", "Class-Balanced Weighted Voting", "balanced"),
    ]

    results = {}

    print("\nEXPERIMENTAL COMPARISON")
    print("-" * 75)

    for code, name, method in experiments:
        result = evaluate(
            training_rows,
            test_rows,
            k=K,
            voting=method,
        )

        results[method] = result

        print(f"\nExperiment {code}: {name}")
        print(f"Accuracy          : {result['accuracy']:.2%}")
        print(f"Precision         : {result['precision']:.2%}")
        print(f"Recall            : {result['recall']:.2%}")
        print(f"F1-score          : {result['f1']:.2%}")
        print(f"Balanced Accuracy : {result['balanced_accuracy']:.2%}")

        print("\nConfusion Matrix:")
        print("                 Predicted")
        print("                 Normal  Failure")
        print(
            f"Actual Normal    {result['tn']:>6}  "
            f"{result['fp']:>7}"
        )
        print(
            f"Actual Failure   {result['fn']:>6}  "
            f"{result['tp']:>7}"
        )

    print("\nFINAL COMPARISON")
    print("-" * 75)
    print(
        f"{'Method':<20} {'Accuracy':>10} "
        f"{'Precision':>11} {'Recall':>10} {'F1':>10}"
    )
    print("-" * 75)

    for code, name, method in experiments:
        r = results[method]

        print(
            f"{name:<20} "
            f"{r['accuracy']:>9.2%} "
            f"{r['precision']:>10.2%} "
            f"{r['recall']:>9.2%} "
            f"{r['f1']:>9.2%}"
        )

    # Use Experiment A for the example prediction.
    prepared = prepare_training_data(training_rows)
    example = test_rows[0]
    query = {f: example[f] for f in FEATURES}

    prediction, neighbours = predict_knn(
        query, prepared, k=K, voting="majority"
    )

    print("\nEXAMPLE MACHINE PREDICTION (EXPERIMENT A)")
    print("-" * 75)

    for feature, value in query.items():
        print(f"{feature:<30}: {value}")

    print(
        "Actual condition    : "
        + ("Failure" if example[TARGET] == 1 else "Normal")
    )
    print(
        "Predicted condition : "
        + ("Failure" if prediction == 1 else "Normal")
    )

    print("\nTOP SIMILAR MACHINES")
    print("-" * 75)

    for rank, neighbour in enumerate(neighbours, start=1):
        condition = (
            "Failure" if neighbour["actual"] == 1 else "Normal"
        )

        print(
            f"{rank}. UDI={neighbour['UDI']} | "
            f"Product={neighbour['Product ID']} | "
            f"Condition={condition} | "
            f"Similarity={neighbour['similarity']:.4f}"
        )

    print("\nNOTE")
    print("-" * 75)
    print("AI4I 2020 is a synthetic predictive-maintenance dataset.")
    print("This model is an academic demonstration, not a")
    print("validated industrial maintenance system.")

    print("\n" + "=" * 75)
    print("EXECUTION COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()

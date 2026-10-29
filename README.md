# Predictive Maintenance Using Gower-Style Similarity and k-NN

**Course:** Data Warehousing and Mining (DWM)  
**Student:** Taazeen Ansari, Anusha Shetty, Aarya Yewale, Pranali Raut
**Roll Number:** 124A1166, 124A1166, 124A1152, 124A1188, 122A1071


## Problem statement
Build a small demonstration system that compares CNC machines with mixed-type attributes and predicts whether a machine is **Normal**, **Warning**, or **Critical**. The implementation combines a Gower-style similarity score with a k-Nearest Neighbors (k-NN) classifier.

## Dataset
`data/cnc_machines.csv` contains 20 **synthetic** machine records for demonstration. It includes:
- `machine_type` — nominal
- `coolant_leak` — binary (0/1)
- `vibration_level` — ordinal (Low, Medium, High)
- `temperature_c`, `spindle_rpm`, `hours_since_service` — numeric
- `condition` — target class

The dataset is fabricated for coursework and must not be treated as real industrial sensor data.

## Algorithm
1. Load the CSV records.
2. Define each attribute's data type and ordinal ordering.
3. For nominal/binary attributes, assign similarity 1 for a match and 0 otherwise.
4. For ordinal attributes, map values to ranks and calculate normalized similarity.
5. For numeric attributes, calculate `1 - abs(x-y)/range`, bounded to [0, 1].
6. Average attribute-level similarities to obtain the Gower-style similarity.
7. Rank training records by similarity to a query and select the top `k` records.
8. Predict the majority class among the neighbours.
9. Evaluate the classifier on a deterministic held-out split and demonstrate a new-machine prediction.

## Implementation details
- `main.py`: similarity calculation, k-NN, evaluation, and demonstration.
- `data/cnc_machines.csv`: synthetic dataset.
- `requirements.txt`: notes that no external packages are required.

This is an educational Gower-style implementation. It uses equal weights for all features and a simple deterministic split; it is not a production-grade predictive-maintenance system.

## Run instructions
Requires Python 3.10 or later.

```bash
python main.py
```

No third-party dependencies are needed.

## Example output
The program prints the train/test sizes, held-out predictions, test accuracy, predicted condition for a sample machine, and its three most similar records. Run the program to obtain the current output.

## Limitations and future improvements
- Replace synthetic data with approved, real and properly labelled machine records.
- Use stratified cross-validation and tune `k` on validation data.
- Add missing-value handling and configurable feature weights.
- Report precision, recall, and F1-score per condition.
- Integrate sensor streams and maintenance records only after industrial validation.
- Consult maintenance engineers before using predictions for real-world actions.

## Author
Taazeen Ansari — Roll No. 124A1166 — Group 4

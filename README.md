# Predictive Maintenance Using Gower-Style Similarity and k-NN

A Data Warehousing and Mining (DWM) lab project that uses mixed-attribute similarity and the k-Nearest Neighbors (k-NN) algorithm to predict machine failures.

## Project Details

- **Institution:** SIES Graduate School of Technology
- **Department:** Computer Engineering
- **Subject:** Data Warehousing and Mining (DWM)
- **Group Number:** 1

### Team Members

| Student Name | Roll Number |
|---|---|
| Taazeen Ansari | 124A1166 |
| Anusha Shetty | 124A1152 |
| Aarya Yewale | 124A1188 |
| Pranali Raut | 122A1071 |

## 1. Problem Statement

Industrial machines generate operating measurements such as temperature, rotational speed, torque and tool wear. Analysing these attributes can help identify patterns associated with machine failure.

The objective of this project is to calculate similarity between machine records using a Gower-style similarity measure and classify a machine into one of two conditions:

- **Normal (0):** No machine failure.
- **Failure (1):** Machine failure recorded.

This implementation uses the AI4I 2020 Predictive Maintenance Dataset and a k-NN classifier.

> **Scope note:** The original assignment describes Normal, Warning and Critical classes. This implementation uses the dataset's original binary `Machine failure` target instead.

## 2. Dataset

**Dataset:** AI4I 2020 Predictive Maintenance Dataset  
**Source:** UCI Machine Learning Repository  
**Format:** CSV  
**Records:** 10,000  
**Original columns:** 14

Dataset source: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

The AI4I dataset is synthetic and designed to reflect industrial predictive-maintenance data. It is not directly collected from operating factory machines.

### Features Used

| Attribute | Type | Description |
|---|---|---|
| Type | Nominal | Product quality type (L, M, H) |
| Air temperature [K] | Numeric | Air temperature |
| Process temperature [K] | Numeric | Process temperature |
| Rotational speed [rpm] | Numeric | Machine rotational speed |
| Torque [Nm] | Numeric | Machine torque |
| Tool wear [min] | Numeric | Tool wear duration |

**Target attribute:** `Machine failure`

- `0` = Normal
- `1` = Failure

The dataset contains 9,661 Normal records (96.61%) and 339 Failure records (3.39%).

`UDI` and `Product ID` are retained for identifying similar records. The failure-mode indicators `TWF`, `HDF`, `PWF`, `OSF` and `RNF` are excluded from model inputs to avoid target leakage.

## 3. Methodology

### Gower-Style Similarity

The implementation calculates similarity between a query machine and training records.

**Nominal attributes:**

Similarity is 1 when the values match and 0 otherwise.

**Numeric attributes:**

\[
S_{numeric}(x,y)=\max\left(0,1-\frac{|x-y|}{R}\right)
\]

where \(R\) is the feature range calculated from the training data.

The final similarity is the average of the six individual feature similarities.

### k-Nearest Neighbors

The classifier uses:

- **Algorithm:** k-Nearest Neighbors
- **k:** 5
- **Train-test split:** 75:25
- **Split strategy:** Stratified
- **Random seed:** 42

The five most similar training records are selected to predict the class of a machine.

### Voting Strategies Evaluated

1. **Standard Majority Voting:** Each neighbour contributes one vote.
2. **Similarity-Weighted Voting:** Each neighbour's vote is weighted by its similarity.
3. **Class-Balanced Weighted Voting:** Similarity weights are adjusted using inverse class frequency to give greater influence to the minority Failure class.

## 4. Project Structure

```text
DWM_Predictive_Maintenance/
│
├── data/
│   ├── ai4i2020.csv
│   └── cnc_machines.csv
│
├── main.py
├── inspect_dataset.py
├── main_backup.py
├── requirements.txt
├── README.md
├── sample_output.txt
└── DWM_Group_1_Predictive_Maintenance_Report.docx
```

`main_backup.py` is an optional local backup of the earlier implementation. `cnc_machines.csv` is retained from the initial prototype; the upgraded program uses `ai4i2020.csv`.

## 5. Technologies Used

- Python 3
- Pandas
- NumPy
- Visual Studio Code
- Git and GitHub

## 6. Installation and Execution

### Step 1: Clone the repository

```bash
git clone https://github.com/MissTaazeen/DWM_Predictive_Maintenance.git
```

### Step 2: Navigate to the project directory

```bash
cd DWM_Predictive_Maintenance
```

### Step 3: Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 4: Run the program

```bash
python main.py
```

### Step 5: Inspect the dataset (optional)

```bash
python inspect_dataset.py
```

Make sure `data/ai4i2020.csv` is present before executing the main program.

## 7. Experimental Results

The model was evaluated on 2,500 test records after training on 7,500 records.

| Metric | Experiment A: Majority | Experiment B: Similarity Weighted | Experiment C: Class-Balanced Weighted |
|---|---:|---:|---:|
| Accuracy | 97.40% | 97.40% | 93.80% |
| Precision | 81.25% | 81.25% | 31.77% |
| Recall | 30.59% | 30.59% | 71.76% |
| F1-score | 44.44% | 44.44% | 44.04% |
| Balanced Accuracy | 65.17% | 65.17% | 83.17% |

### Discussion

The dataset is imbalanced, with only 3.39% Failure records.

Standard majority voting achieved 97.40% accuracy but detected only 30.59% of actual failures.

Class-balanced weighted voting increased failure recall to 71.76% and balanced accuracy to 83.17%. However, its precision decreased to 31.77%, indicating more false-positive predictions.

The results demonstrate the trade-off between detecting more failures and avoiding false alarms.

## 8. Limitations and Future Scope

- The AI4I dataset is synthetic rather than real factory-collected data.
- The dataset has a highly imbalanced target distribution.
- Results are based on one stratified train-test split.
- The current implementation uses six selected input attributes and a fixed k value of 5.
- The model is an academic prototype and is not validated for real industrial maintenance decisions.

Possible future improvements include:

- Hyperparameter tuning and stratified cross-validation.
- Experimenting with different values of k and class weights.
- Comparing k-NN with other classification algorithms.
- Adding confusion-matrix and precision-recall visualizations.
- Evaluating the approach on real industrial datasets.

## 9. GitHub Repository

**Repository:** https://github.com/MissTaazeen/DWM_Predictive_Maintenance

The repository contains the source code, dataset, dependencies, project documentation and lab report.

## Disclaimer

This project is developed for academic purposes as part of the DWM laboratory assignment. Its predictions should not be used as the sole basis for real-world industrial maintenance decisions.
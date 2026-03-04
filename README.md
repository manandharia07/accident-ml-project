# 🚦 Accident Severity Prediction & Hotspot Detection

An end-to-end Machine Learning project that predicts UK road accident severity and detects high-risk geographical hotspots using 1.78 million real-world records from the UK Department for Transport (2005–2015).

---

## 📌 Project Overview

Road accidents cause significant human and economic losses worldwide. This project tackles two related problems:

1. **Severity Prediction** — Predict how severe an accident will be (Fatal / Serious / Slight) based on road, environment, and driver features using Random Forest classifiers.
2. **Hotspot Detection** — Identify high-risk geographic zones and provide a real-time risk checker for any given location using geospatial analysis and interactive maps.

---

## 📁 Repository Structure

```
accident-ml-project/
│
├── Severity_Prediction.ipynb          # ML pipeline: merging, preprocessing, training, evaluation, inference
├── Accident_Hotspot_Detection.ipynb   # Geospatial hotspot detection & interactive Folium maps
├── Project_Documentation.pdf          # Full project report with results, methodology, and visuals
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Ignores large CSV/pkl/html files
└── README.md
```

---

## 📊 Dataset

**Source:** [UK DfT Accident Data — Kaggle](https://www.kaggle.com/datasets/silicon99/dft-accident-data)

| File | Records | Description |
|---|---|---|
| `Accidents0515.csv` | ~1,780,515 | Accident info: location, severity, time, road & environment |
| `Casualties0515.csv` | — | Casualty details per accident |
| `Vehicles0515.csv` | — | Vehicle and driver details per accident |

After merging all 3 files → **857,519 records** with 100+ features.

> ⚠️ Dataset files are NOT included due to size. Auto-downloaded via `kagglehub` when notebooks run.

---

## ⚙️ ML Pipeline — Severity Prediction

### Step 1: Data Loading & Merging
```python
merged_data = pd.merge(Accidents, Vehicles, left_index=True, right_index=True, how='inner')
merged_data = pd.merge(merged_data, Casualties, left_index=True, right_index=True, how='inner')
# Result: 857,519 records with 100+ features
```

### Step 2: Feature Selection
Features selected via **correlation analysis** with `Accident_Severity` (threshold |corr| > 0.005).

**Final 10 Pre-Accident Features:**

| Feature | Type | Domain |
|---|---|---|
| Speed Limit | Numeric | Road |
| Urban or Rural Area | Categorical | Location |
| Light Conditions | Categorical | Environment |
| Road Type | Categorical | Infrastructure |
| 1st Road Class | Categorical | Infrastructure |
| Local Authority (District) | Categorical | Location |
| Age of Driver | Numeric | Human |
| Age Band of Driver | Categorical | Human |
| Sex of Driver | Categorical | Human |
| Journey Purpose of Driver | Categorical | Human |

### Step 3: Preprocessing
- Missing values → **Median imputation** (`SimpleImputer`)
- Categorical codes → **Bidirectional mapping dictionaries**
- Train/Test split → **80/20 stratified**

### Step 4: Model — Random Forest Classifier
**Why Random Forest?**
- Handles mixed feature types (numeric + categorical)
- Robust to class imbalance using `class_weight='balanced'`
- No feature scaling required

---

## 📈 Three Model Variants & Results

| Option | Task | Accuracy |
|---|---|---|
| **A** | 3-Class: Fatal / Serious / Slight | 58.75% |
| **B ✅ Recommended** | Binary: Fatal vs Non-Fatal | **72.72%** |
| **C** | Binary: Slight vs Serious+Fatal | 67.11% |

### Best Model: Option B — Fatal vs Non-Fatal

```
              precision    recall    f1-score    support
   Non-Fatal       0.99      0.73      0.84      840798
       Fatal       0.05      0.75      0.10       16721

    accuracy                           0.73      857519
```

> **Key insight:** Fatal accidents are rare (~2% of records). The model achieves **0.75 recall for Fatal class** — correctly catching most life-threatening accidents. This is the critical metric for a real-world safety system.

---

## 🗺️ Hotspot Detection

### Methodology
```
1. Load 1.78M records in memory-safe chunks (chunksize=40000)
2. Balance classes → 22,998 samples each (Fatal / Serious / Slight)
3. Plot all accidents on interactive Folium map
      🔴 Red = Fatal  |  🟠 Orange = Serious  |  🟢 Green = Slight
4. User inputs latitude & longitude
5. Count accidents in 3km radius using geodesic distance
6. Return risk verdict + save updated HTML map
```

### Sample Output
```
=============================================
          🚦 ACCIDENT RISK REPORT
=============================================
📌 User Location: (52.4862, -0.1585)
🔴 Fatal Accidents   : 2
🟠 Serious Accidents : 1
🟢 Slight Accidents  : 1
📢 FINAL RISK VERDICT: FATAL
🗺️  Map saved as: USER_RISK_RESULT_MAP.html
=============================================
```

---

## 📦 Deployment Artifacts

Generated after running `Severity_Prediction.ipynb`:

| File | Purpose |
|---|---|
| `model.pkl` | Trained Random Forest (Option B) |
| `input_mapper.pkl` | Categorical encoding/decoding dictionaries |
| `selected_features.pkl` | Feature list used during training |
| `imputer.pkl` | Fitted median imputer for inference |

---

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/manandharia07/accident-ml-project.git
cd accident-ml-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run severity prediction
jupyter notebook Severity_Prediction.ipynb

# 4. Run hotspot detection
jupyter notebook Accident_Hotspot_Detection.ipynb
```

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `Matplotlib` · `Seaborn` · `Folium` · `Geopy` · `KaggleHub`

---

## 📄 Full Documentation

See [`Project_Documentation.pdf`](./Project_Documentation.pdf) for complete methodology, classification reports, and map screenshots.

---

## 👥 Authors

**Manan Dharia** (25MCE005) & **Milan Korat** (25MCE010)
M.Tech CSE — Advanced Machine Learning, Nirma University, December 2025

---

## 📜 License

Academic use. Dataset: [UK DfT Road Safety Data](https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data) via Kaggle.

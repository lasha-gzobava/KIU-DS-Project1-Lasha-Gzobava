
# 📊 Multi-Module Data Analytics Suite

This repository contains three self-contained Python analytics modules — **Weather & Sales Analytics**, **Student Grading System**, and **Fitness Analytics** — combined under a single entry point (`main.py`).
Each module demonstrates realistic data generation, cleaning, and statistical analysis using **NumPy** and Python OOP principles.

---

## 🧩 Project Structure

```
├── main.py
├── weather_sales_analytics.py
├── student_grading.py
├── fitness_analytics.py
└── README.md
```

---

## 🚀 How to Run

### 1️⃣ Requirements

* Python 3.10+
* NumPy (only external dependency)

Install via pip:

```bash
pip install numpy
```

### 2️⃣ Execute the entire suite

```bash
python main.py
```

This will sequentially execute:

1. **Weather & Sales Analytics** (Part A)
2. **Student Grading System** (Part B)
3. **Fitness Analytics** (Part C)

Each module prints detailed analysis reports directly to the console.

---

## 🌤️ Part A — Weather & Sales Analytics

**File:** `weather_sales_analytics.py`

This module simulates climate data across 5 locations for 365 days and sales data for 12 months × 4 categories.
It demonstrates NumPy operations such as:

* Array creation, slicing, and indexing
* Boolean and fancy indexing
* Rolling averages & normalization
* Correlation and descriptive statistics

### Example Output

```
=== PART A: ARRAYS, INDEXING & STATS ===
Generating climate and sales data…
Shape: (5, 365)
Days with at least one city >35°C: 52
Top month: Jul (total $13,824)
Best category overall: Category 3 ($48,772)
```

---

## 🎓 Part B — Student Grading System

**File:** `student_grading.py`

A class-based grading engine for a small roster of students.
Each record includes name, marks, and attendance count.
It computes:

* Average scores
* Letter grades (A–F)
* Pass/fail eligibility (based on attendance ≥ 75 % and grade ≥ 60 %)
* Leaderboard & summary statistics

### Example Output

```
=== PART B: STUDENT EVALUATION SYSTEM ===
Top 5 performers:
1. S010 — Lasha Gzobava: 100.0 (A)
...
Class summary:
Passed: 8 (80.0%)
Avg Attendance Rate: 89.3%
```

---

## 🏃 Part C — Fitness Analytics

**File:** `fitness_analytics.py`

A full-scale data pipeline that models fitness tracking metrics for 100 users across 90 days.

### Key Features

* Synthetic 3D data cube generation: users × days × metrics
* Data cleaning: noise injection, NaN imputation, IQR-based outlier winsorization
* Statistical analyses:

  * User & population averages
  * Consistency and ranking analysis
  * Temporal & weekday trends
  * Correlations and demographic insights
  * Health scoring and goal-tracking system

### Example Output

```
=== PART C: FITNESS DATA PIPELINE ===
Simulating 100 users × 90 days × 4 metrics…
Cleaned data summary (per metric):
Steps mean=8470.22, std=1520.44
Top 10 users by combined z-score:
...
Users with >80% all-goals days: 7
```

---

## 🧠 Concepts Demonstrated

* NumPy data manipulation & statistical operations
* Object-oriented modular design
* Data cleaning (imputation, winsorization)
* Aggregation and summarization
* Console-based analytical reporting

---

## 🛠️ Customization

You can modify:

* Random seed values for reproducible outputs (`seed` parameter)
* Number of users, days, and metrics in `FitnessAnalytics`
* Student data and grading thresholds in `StudentGradingSystem`

---

## 🧾 License

This project is provided for **educational and academic** purposes under the MIT License.


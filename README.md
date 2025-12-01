# 🔮 SISU Preview: A Machine Learning Cutoff Score Predictor

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io)
[![Model](https://img.shields.io/badge/Model-LightGBM-purple.svg)](https://lightgbm.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is a complete, end-to-end machine learning solution that predicts university cutoff scores for Brazil's SISU. It showcases the full development lifecycle, from data engineering and analysis to deploying an optimized model as a user-friendly web application.

---

## Live Demo

You can access and interact with the live application deployed on Streamlit Community Cloud:

**➡️ [Launch SISU Preview App](https://sisu-preview.streamlit.app/)**

![SISU Preview App Screenshot 1](https://i.imgur.com/wt99gIf.png)
![SISU Preview App Screenshot 2](https://i.imgur.com/Wna1eAY.png)
![SISU Preview App Screenshot 3](https://i.imgur.com/j6yNWEW.png)

---

## Project Overview

Predicting cutoff scores for Brazil's highly competitive university selection process (SISU) is a complex challenge due to volatility and distinct regional factors. This project aims to provide a data-driven estimate to help prospective students gauge their chances across all admission categories.

The core is a Global LightGBM regression model trained on a rich historical dataset (2019–2025). The project evolved from a simple linear predictor into a robust ML pipeline, utilizing DuckDB for heavy data transformations and Window Functions to capture temporal trends, ensuring accurate predictions for both General Admission (Ampla Concorrência) and Affirmative Action Quotas (Cotas) through and iterative process of analysis, hypothesis testing, and refinement.

---

## Key Features

* **Universal Prediction:** Unlike simple average-based tools, this model predicts scores for all modalities (Racial, PwD, Income-based), adapting to the specific competitive context of each category.
* **Interactive Simulation:** Users can filter by University, Course, Level, Campus, Shift and Modality to get real-time estimates of multiple courses.
* **Historical Trends:** Displays interactive Plotly charts showing the score evolution over the years.
* **Exportable Results:** Generates an exportable table at the bottom of the page with detailed results for all predictions.
* **Robust Engineering:** Powered by a DuckDB backend that handles complex joins between Vacancy data (weights/minimums) and historical scores without memory overhead.

---

## Tech Stack

* **Core:** Python 3.10+
* **Data Engineering:** DuckDB (OLAP SQL), Pandas, Parquet
* **Machine Learning:** LightGBM, Scikit-learn
* **Web App:** Matplotlib, Seaborn, Streamlit
* **Visualization:** Plotly Express
* **CI/CD & Quality:** GitHub Actions, Pytest

---

## The Model Optimization Process

This project's value lies in its iterative development cycle.

### 1. The Baseline

The initial approach was to train a single LightGBM model on the entire dataset, including all admission categories (affirmative action quotas, etc.).

* **Result:** This model was impractical, with a **Mean Absolute Error (MAE) of ~25 points**. An error this large makes the predictions unreliable.

### 2. The MVP (Focus on General Admission)

To validate feasibility, the problem was simplified. We isolated "Ampla Concorrência" (General Admission) data and removed outliers (cutoff scores of 0.0).

* **Action:** Trained a specialist model on this cleaner segment with a quick hyperparemeter tuning to find the optimal settings for `n_estimators` and `learning_rate`.
* **Result:** Success. The MAE dropped to  **~17 points**, proving the model could learn effectively when variance was controlled. However, it lacked historical context and support for quotas.

### 3. The Engineering Overhaul (DuckDB & Data Integrity)

To generalize the model for all modalities, we needed a robust data architecture.

* **Action:**
  * Migrated the pipeline to DuckDB to handle complex joins between Cutoff and Vacancies datasets, enabling efficient feature creation directly in SQL.
  * Implemented SQL Window Functions to create Lag Features (previous years' scores) and historical trends.

### 4. Experimenting with Delta Targets (Failed Hypothesis)

With clean data, we attempted to predict year-over-year changes (delta) instead of absolute scores to reduce location bias.

* **Hypothesis:** Predicting the change would better capture shocks in demand.
* **Outcome:** The model collapsed to tiny adjustments near zero or exploded for sparse quotas.
* **Decision:** Abandoned Delta targets. Reverted to predicting Absolute Scores, but kept the new features derived during this phase (rolling trends, demand ratios).

### 5. Refining the Signal: Leakage Removal & The Global Model

Deep diagnostics revealed that the model was "cheating" by relying too heavily on the direct previous year's score (`nota_edicao_anterior`).

* **Action:**
  * **Leakage Fix:** Removed the direct lag variable from training, replacing it with derived features (relative deltas, rolling means) to force the model to learn patterns rather than just copying values.
  * **Consolidation:** Instead of separate models for each quota, we trained a single, robust Global LightGBM Model**.**
* **Why Global Worked:** By encoding modality as a categorical feature, the global model leveraged the massive dataset to stabilize predictions for sparse quotas.

### 6. Final Production State

The system is now a production-ready forecasting tool.

* **Performance:** **~6 points** **MAE** with **0.990 R²**.
* **Reliability:** Explicit column selection prevents silent leaks, and a deterministic SQL pipeline ensures data consistency.
* **Inference:** The system builds plausible synthetic "future rows" for 2026 using the latest available context, ready for deployment.

---

## 📂 Project Structure

```
sisu-preview/
├── .github/
│   └── workflows/      # CI Pipeline (tests)
├── .streamlit/         # Streamlit configuration
├── data/
│   ├── raw/            # Source XLSX files (not committed)
│   ├── processed/      # Cleaned Parquet checkpoints
│   └── database/       # DuckDB database file (sisu_preview.db)
├── scripts/
│   └── build_database.py  # Orchestrator: builds the DB from Parquets
├── src/
│   ├── data_processing.py # Raw data cleaning logic
│   └── model_training.py  # Model training & evaluation logic
├── tests/                 # Pytest suite
├── saved_models/
│   └── lgbm_sisu_predictor.joblib
├── app.py                 # Streamlit frontend
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run Locally

To run this project on your local machine, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone [https://github.com/cauafsantosdev/sisu-preview.git](https://github.com/cauafsantosdev/sisu-preview.git)
   cd sisu-preview
   ```
2. **Create and activate a virtual environment:**

   ```bash
   # For Windows
   python -m venv .venv
   .venv\Scripts\activate

   # For macOS/Linux
   python -m venv .venv
   source .venv/bin/activate
   ```
3. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
4. **Build the Database:**

```bash
  python -m scripts.build_database
```

5. **Run the Streamlit application:**

```bash
   streamlit run app.py
```

   The application should open in your web browser.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

Cauã Santos – [My LinkedIn Profile](https://www.linkedin.com/in/cauafsantosdev/) – cauafsantosdev@gmail.com

Project Link: [https://github.com/cauafsantosdev/sisu-preview](https://github.com/cauafsantosdev/sisu-preview)

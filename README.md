# 🔮 SISU Preview: A Machine Learning Cutoff Score Predictor

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io)
[![Model](https://img.shields.io/badge/Model-LightGBM-purple.svg)](https://lightgbm.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is a complete, end-to-end machine learning solution that predicts university cutoff scores for Brazil's SISU. It showcases the full development lifecycle, from data engineering and analysis to deploying an optimized model as a user-friendly web application.

---

## Live Demo

You can access and interact with the live application deployed on Streamlit Community Cloud:

**➡️ [Launch SISU Preview App](https://your-streamlit-app-url.streamlit.app/)** *(Note: Replace the URL above with your actual Streamlit app link after deployment.)*

![SISU Preview App Screenshot](https://i.imgur.com/your-screenshot-url.png)
*(Note: It's highly recommended to add a screenshot of your app here. You can upload an image to a service like Imgur and paste the link.)*

---

## Project Overview

Predicting the cutoff scores for Brazil's highly competitive university selection process (SISU) is a major challenge for prospective students. These scores are volatile and depend on a lot of factors. This project aims to provide a data-driven estimate to help students gauge their chances and make informed decisions.

The core of the project is a LightGBM regression model trained on a historical dataset spanning from 2019 to 2025. However, the most significant part of this project was not just training a model, but the iterative process of analysis, hypothesis testing, and refinement to achieve a useful level of accuracy.

---

## Key Features

* **Interactive Prediction:** Users can select a university, course, degree, and shift to get a score estimate.
* **Historical Context:** The app displays a Plotly chart showing the historical trend of the selected course's cutoff score.
* **Specialist Model:** The final model is an expert, trained specifically on the most competitive and data-rich segment: "Ampla Concorrência" (General Admission).
* **Transparent Performance:** The application clearly communicates the model's average error margin, promoting a realistic understanding of the prediction.

---

## Tech Stack

* **Programming Language:** Python
* **Data Manipulation & Analysis:** Pandas
* **Machine Learning:** Scikit-learn, LightGBM
* **Web Application:** Streamlit
* **Data Visualization:** Matplotlib, Seaborn, Plotly Express
* **Data Storage:** Parquet

---

## The Model Optimization Process

This project's value lies in its iterative development cycle.

### 1. The Baseline

The initial approach was to train a single LightGBM model on the entire dataset, including all admission categories (affirmative action quotas, etc.).

* **Result:** This model was impractical, with a **Mean Absolute Error (MAE) of ~64 points**. An error this large makes the predictions unreliable.

### 2. Focus on General Admission

After analyzing the baseline's poor performance, the hypothesis was that the high variance and noise from sparse data in affirmative action categories were polluting the model's learning process.

* **Action:** A new strategy was adopted. The problem was simplified by training a **specialist model** focused only on "Ampla Concorrência" (General Admission) for courses with 10 or more spots.
* **Result:** This was a major success. The MAE dropped by over 50% to **~30 points**, validating the hypothesis that isolating a cleaner, more homogeneous data segment was the correct approach.

### 3. Error Analysis & Hyperparameter Tuning

To further improve, a deep-dive error analysis was conducted.

* **Insight:** The analysis revealed that the model's largest errors were on data points where the actual cutoff score was `0.0`, which likely represents data anomalies or non-competitive scenarios. These outliers were removed in a final data-cleaning step.
* **Action:** A quick hyperparameter tuning round was performed to find the optimal settings for `n_estimators` and `learning_rate`.
* **Final Result:** The final, optimized specialist model achieved a **MAE of ~16 points** and an **R² of 0.89** on the test set, transforming it into a genuinely useful predictive tool.

---

## 📂 Project Structure

```
sisu-preview/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   └── 02_model_evolution.ipynb
├── src/
│   ├── data_processing.py
│   └── model_training.py
├── saved_models/
│   └── lgbm_sisu_predictor.joblib
├── app.py
├── requirements.txt
├── LICENSE
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
4. **Run the Streamlit application:**

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

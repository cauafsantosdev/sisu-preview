import os
import joblib
import duckdb
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load data
print("Loading DB sisu_data...")
db_path = "data/database/sisu_preview.db"
conn = duckdb.connect(database=str(db_path), read_only=True)
df = conn.sql("SELECT * FROM sisu_data").df()

# Noise removal
df = df[df["nu_notacorte"] != 0].copy()

# Removing rows with missing values in "vagas_edicao_anterior" and "inscritos_edicao_anterior"
df = df.dropna(subset=["vagas_edicao_anterior", "inscritos_edicao_anterior"]).copy()

# Removing previous grade data leakage
if "nota_edicao_anterior" in df.columns:
    df = df.drop(columns=["nota_edicao_anterior"])

# Features and Target
TARGET = "nu_notacorte"
DROP_COLS = [
    "edicao", "co_ies", "no_ies", "co_curso", "qt_inscricao", "ano", "__index_level_0__",
    "chave_curso", "no_municipio_campus", "nu_notacorte"
]
features = [c for c in df.columns if c not in DROP_COLS]

categorical_features = [
    "sg_ies", "no_curso", "no_campus", "ds_grau", "ds_turno", "sg_uf_campus", "regiao", "ds_mod_concorrencia"
]
for col in categorical_features:
    if col in df.columns:
        df[col] = df[col].astype("category")

X = df[features].copy()
y = df[TARGET]

# Model Training
print("Training LightGBM model...")

params = {
    "learning_rate": 0.03,
    "num_leaves": 512,
    "max_depth": 8,
    "n_estimators": 5000,
    "min_child_samples": 25,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "feature_fraction": 0.9,
    "reg_alpha": 0.6,
    "reg_lambda": 1.2,
    "min_split_gain": 0.2,
    "random_state": 42,
    "verbose": -1,
}

model = lgb.LGBMRegressor(**params)
model.fit(X, y, categorical_feature=categorical_features)

# Performance Check
test_mask = df["ano"] == df["ano"].max()
X_test, y_test = X[test_mask], y[test_mask]
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print(f"\nResults (last year):")
print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}")

# Save Model
os.makedirs("saved_models", exist_ok=True)
path_out = "saved_models/lgbm_sisu_predictor.joblib"
joblib.dump(model, path_out)
print(f"\nModel saved in: {path_out}")
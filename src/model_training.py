import pandas as pd
import lightgbm as lgb
import joblib
import os


PROCESSED_DATA_PATH = 'data/processed/final_data.parquet'
MODEL_OUTPUT_PATH = 'saved_models/lgbm_sisu_predictor.joblib'

print("Loading and preparing final data...")
df = pd.read_parquet(PROCESSED_DATA_PATH)

# Applying the filters
df_cleaned = df[df['nu_notacorte'] != 0].copy()
df_model = df_cleaned.query(
    "`ds_mod_concorrencia` == 'AMPLA CONCORRÊNCIA' and `qt_vagas_concorrencia` >= 10"
).copy()

df_model = df_model.dropna(subset=['nota_edicao_anterior', 'vagas_edicao_anterior'])

print(f"Final dataset for training with {len(df_model)} rows.")

# Feature and Target definition
TARGET = 'nu_notacorte'
COLUMNS_TO_EXCLUDE = [
    'edicao', 'co_ies', 'no_ies', 'no_campus', 'co_curso',
    'chave_curso', 'qt_inscricao', TARGET, 'ds_mod_concorrencia'
]
features = [col for col in df_model.columns if col not in COLUMNS_TO_EXCLUDE]

X = df_model[features].copy()
y = df_model[TARGET]

categorical_features = ['sg_ies', 'no_curso', 'ds_grau', 'ds_turno']
for col in categorical_features:
    X[col] = X[col].astype('category')

# Final Model Training
print("\nStarting the training of the final optimized model...")

# Using the best parameters discovered during the testing phase
best_params = {
    'n_estimators': 5000, 
    'learning_rate': 0.01,
    'random_state': 42
}

lgbm_final = lgb.LGBMRegressor(**best_params)

# Training the model with ALL available and clean data
lgbm_final.fit(X, y)

print("Final training complete!")

# Saving the Model
os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
joblib.dump(lgbm_final, MODEL_OUTPUT_PATH)

print(f"\nFinal optimized model saved successfully at: {MODEL_OUTPUT_PATH}")
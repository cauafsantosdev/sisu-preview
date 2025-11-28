import duckdb
from pathlib import Path
from src.data_processing import process_data


BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "data" / "database"
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

DB_DIR.mkdir(parents=True, exist_ok=True)
db_path = DB_DIR / "sisu_preview.db"

sql_transform_query = f"""
/* Load raw parquet files (cutoffs and vacancies) */
WITH all_cutoffs_raw AS (
    SELECT *
    FROM read_parquet('{PROCESSED_DATA_DIR}/*_notasdecorte.parquet', union_by_name = TRUE)
), 

all_vacancies_raw AS (
    SELECT *
    FROM read_parquet('{PROCESSED_DATA_DIR}/*_vagas.parquet', union_by_name = TRUE)
),

/* Deduplication using ROW_NUMBER */
all_cutoffs_deduped AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY edicao, chave_curso, ds_mod_concorrencia
            ORDER BY edicao
        ) AS rn
    FROM all_cutoffs_raw
),

all_vacancies_deduped AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY edicao, chave_curso, ds_mod_concorrencia
            ORDER BY edicao
        ) AS rn
    FROM all_vacancies_raw
),

/* Merge cutoff data with vacancies information */
merged_data AS (
    SELECT
        c.* EXCLUDE (rn),  -- Remove temporary dedup column

        -- Add vacancies rule parameters
        v.peso_redacao, v.nota_minima_redacao,
        v.peso_linguagens, v.nota_minima_linguagens,
        v.peso_matematica, v.nota_minima_matematica,
        v.peso_ciencias_humanas, v.nota_minima_ciencias_humanas,
        v.peso_ciencias_natureza, v.nota_minima_ciencias_natureza,
        v.nu_media_minima_enem

    FROM 
        (SELECT * FROM all_cutoffs_deduped WHERE rn = 1) c
    LEFT JOIN 
        (SELECT * FROM all_vacancies_deduped WHERE rn = 1) v
        ON c.edicao = v.edicao
        AND c.chave_curso = v.chave_curso
        AND c.ds_mod_concorrencia = v.ds_mod_concorrencia
),

/* Lag Features (1 and 2 editions back) */
lag_features AS (
    SELECT
        *,
        /* 1-year lag features */
        LAG(nu_notacorte, 1) OVER w AS lag1_nota,
        LAG(qt_vagas_concorrencia, 1) OVER w AS lag1_vagas,
        LAG(qt_inscricao, 1) OVER w AS lag1_inscritos,

        /* 2-year lag features */
        LAG(nu_notacorte, 2) OVER w AS lag2_nota,
        LAG(qt_vagas_concorrencia, 2) OVER w AS lag2_vagas,
        LAG(qt_inscricao, 2) OVER w AS lag2_inscritos,

    FROM merged_data
    WINDOW w AS (PARTITION BY chave_curso, ds_mod_concorrencia ORDER BY edicao)
),

/* Extra domain-specific features */
extra_features AS (
    SELECT
        *,
        /* Binary flag: technological degree */
        CASE WHEN UPPER(ds_grau) LIKE '%TECNOL%' THEN 1 ELSE 0 END AS is_tecnologico,

        /* Region mapping based on campus state */
        CASE
            WHEN sg_uf_campus IN ('AC','AM','AP','PA','RO','RR','TO') THEN 'Norte'
            WHEN sg_uf_campus IN ('AL','BA','CE','MA','PB','PE','PI','RN','SE') THEN 'Nordeste'
            WHEN sg_uf_campus IN ('DF','GO','MT','MS') THEN 'Centro-Oeste'
            WHEN sg_uf_campus IN ('ES','MG','RJ','SP') THEN 'Sudeste'
            WHEN sg_uf_campus IN ('PR','RS','SC') THEN 'Sul'
            ELSE 'Outro'
        END AS regiao
    FROM lag_features
),

/* rolling windows, deltas, trends, regional aggregates, growth rates */
final_features AS (
    SELECT
        *,

        /* Cleaned lag names */
        lag1_nota AS nota_edicao_anterior,
        lag1_vagas AS vagas_edicao_anterior,
        lag1_inscritos AS inscritos_edicao_anterior,

        /* Trend based on last 2 editions */
        CASE WHEN lag1_nota IS NOT NULL AND lag2_nota IS NOT NULL
             THEN lag1_nota - lag2_nota
             ELSE 0 END AS tendencia_nota,

        /* Previous demand */
        CASE WHEN lag1_inscritos IS NOT NULL AND lag1_vagas IS NOT NULL
             THEN (lag1_inscritos / (lag1_vagas + 1))
             ELSE 0 END AS demanda_anterior,

        /* Delta features */
        CASE WHEN lag1_vagas IS NOT NULL AND lag2_vagas IS NOT NULL
             THEN lag1_vagas - lag2_vagas
             ELSE 0 END AS delta_vagas,

        CASE WHEN lag1_inscritos IS NOT NULL AND lag2_inscritos IS NOT NULL
             THEN lag1_inscritos - lag2_inscritos
             ELSE 0 END AS delta_inscritos,

        /* Rolling averages */
        AVG(nu_notacorte) OVER (
            PARTITION BY chave_curso, ds_mod_concorrencia
            ORDER BY edicao
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) AS rolling_media_nota_3anos,

        AVG(nu_notacorte) OVER (
            PARTITION BY sg_uf_campus
            ORDER BY edicao
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) AS rolling_media_uf_3anos,

        /* Growth rate (%) */
        CASE WHEN lag2_nota > 0 AND lag1_nota IS NOT NULL AND lag2_nota IS NOT NULL
             THEN (lag1_nota - lag2_nota) / lag2_nota
             ELSE 0 END AS taxa_crescimento_nota,

        /* National mean by degree */
        AVG(nu_notacorte) OVER (PARTITION BY ds_grau) AS media_nacional_grau,

        /* Regional aggregates */
        AVG(nu_notacorte) OVER (PARTITION BY regiao) AS media_regiao,
        STDDEV(nu_notacorte) OVER (PARTITION BY regiao) AS desvio_regiao,

        /* Relative deltas */
        CASE WHEN lag1_vagas IS NOT NULL
            THEN (qt_vagas_concorrencia - lag1_vagas) / (lag1_vagas + 1)
            ELSE 0 END AS delta_vagas_rel,

        CASE WHEN lag1_inscritos IS NOT NULL
            THEN (qt_inscricao - lag1_inscritos) / (lag1_inscritos + 1)
            ELSE 0 END AS delta_inscritos_rel,

        qt_inscricao * 1.0 / (qt_vagas_concorrencia + 1) AS demanda_ratio,

        /* Normalized year */
        (ano - MIN(ano) OVER ()) * 1.0 / (MAX(ano) OVER () - MIN(ano) OVER ()) AS ano_norm

    FROM extra_features
)

/* Final Output */
SELECT * EXCLUDE (
        lag1_nota,
        lag2_nota,
        lag1_inscritos,
        lag2_inscritos,
        lag1_vagas,
        lag2_vagas
    )
FROM final_features
"""

def build_database():
    """
    Orchestration script that uses DuckDB to build the database from processed Parquet files.
    """
    print("Starting DuckDB orchestration...")

    # Calls the pre-processor to update Parquet files
    process_data(RAW_DATA_DIR, PROCESSED_DATA_DIR)

    # Connect to DuckDB and run the transformation
    print(f"Connecting and building database at: {db_path}")

    # Connects to the database file (it will be created if it doesn't exist)
    conn = duckdb.connect(str(db_path))

    # Executes the entire query and saves the result in the 'sisu_data' table
    conn.execute(f"CREATE OR REPLACE TABLE sisu_data AS ({sql_transform_query})")

    conn.close()
    print("DuckDB built successfully!")

if __name__ == "__main__":
    build_database()
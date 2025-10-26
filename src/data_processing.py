import pandas as pd
import os


# List of columns to drop from the cutoff dataset
DROP_CUTOFF_COLUMNS = ['NU_ANO', 'TP_MODALIDADE', 'DS_REGIAO_CAMPUS', 'NU_PERCENTUAL_BONUS', 
                       'DS_ORGANIZACAO_ACADEMICA', 'TP_MOD_CONCORRENCIA', 'CO_CAMPUS', 
                       'TIPO_CONCORRENCIA', "NU_EDICAO", "DS_CATEGORIA_ADM"]

# List of columns to drop from the vacancies dataset
DROP_VACANCIES_COLUMNS = ['co_ies', 'co_curso', 'ds_categoria_adm', 'ds_grau', 'no_municipio_campus', 
                   'ds_organizacao_academica', 'ds_periodicidade', 'ds_regiao', 'ds_turno', 'no_campus',
                   'no_curso', 'no_ies', 'nu_ano', 'nu_edicao', 'nu_percentual_bonus', 'nu_perc_i', 'edicao',
                   'nu_perc_lei', 'nu_perc_pcd', 'nu_perc_ppi', 'nu_perc_ppi_def', 'nu_perc_pp', 'sg_uf_campus',
                   'nu_perc_q', 'nu_vagas_autorizadas', 'perc_uf_i', 'perc_uf_ibge_i', 'perc_uf_ibge_pcd', 
                   'perc_uf_ibge_ppi', 'perc_uf_ibge_pp', 'perc_uf_ibge_q', 'perc_uf_pcd', 'perc_uf_pp', 
                   'perc_uf_ppid', 'perc_uf_pre_ppi', 'perc_uf_q', 'qt_semestre', 'qt_vagas_concorrencia', 
                   'qt_vagas_ofertadas', 'sg_ies', 'tp_cota', "tp_modalidade", "tp_mod_concorrencia"]

import re

def normalize_campus_name(name: str) -> str:
    if not isinstance(name, str):
        return name
    
    # Converts to upper
    name = name.upper()
    # Removes common ponctuations 
    name = re.sub(r'[-.,_]', '', name)
    # Removes extra whitespaces at start/end and multiple whitespaces in the middle
    name = " ".join(name.strip().split())
    # Normalizes common terms
    name = name.replace('CAMPUS UNIVERSITARIO', 'CAMPUS')
    name = name.replace('CAMPUS DE', 'CAMPUS')
    name = name.replace('UNIDADE SEDE', 'SEDE')
    
    return name

def normalize_concurrency(concurrency: str) -> str:
    if "renda familiar bruta per capita igual ou inferior" in concurrency:
        if "pretos" in concurrency:
            return "LB_PPI"
        elif "quilombolas" in concurrency:
            return "LB_Q"
        elif "deficiência" in concurrency:
            return "LB_PCD"
        return "LB_EP"
        
    elif "independentemente da renda" in concurrency:
        if "pretos" in concurrency:
            return "LI_PPI"
        elif "quilombolas" in concurrency:
            return "LI_Q"
        elif "deficiência" in concurrency:
            return "LI_PCD"
        return "LI_EP"
    
    elif "Ampla" in concurrency:
        return "AC"
    
    return ""

def create_course_key(df):
    return (df['edicao'].astype(str) + '_' +
            df['co_ies'].astype(str) + '_' +
            df['co_curso'].astype(str) + '_' +
            df['no_campus'].astype(str) + '_' +
            df['ds_grau'].astype(str) + '_' +
            df['ds_turno'].astype(str) + '_' +
            df['ds_mod_concorrencia'].astype(str))

def process_cutoff_file(file_path):
    """
    Processa um único arquivo XLSX do SISU, limpando e padronizando os dados.
    Esta é a sua função original, levemente ajustada.
    """
    df = pd.read_excel(file_path, sheet_name=1)

    # Creates 'EDICAO' column if it doesn't exist
    if 'EDICAO' not in df.columns:
        df["EDICAO"] = df["NU_ANO"].astype(str) + "/" + df["NU_EDICAO"].astype(str)
        edition_column = df.pop('EDICAO')
        df.insert(0, 'edicao', edition_column)
        year_column = df['NU_ANO']
        df.insert(1, 'ano', year_column)
    else:
        year_column = df['EDICAO']
        df.insert(1, 'ano', year_column)

    # Renames 'QT_VAGAS_OFERTADAS' to 'qt_vagas_concorrencia' if it exists
    if 'QT_VAGAS_OFERTADAS' in df.columns:
        df.rename(columns={'QT_VAGAS_OFERTADAS': 'qt_vagas_concorrencia'}, inplace=True)

    df['DS_MOD_CONCORRENCIA'] = df['DS_MOD_CONCORRENCIA'].apply(normalize_concurrency)
    df = df[df['DS_MOD_CONCORRENCIA'] != '']

    # Aplicando a função no seu pipeline
    df['NO_CAMPUS'] = df['NO_CAMPUS'].apply(normalize_campus_name)

    # Removes unnecessary columns if they exist
    df.drop(columns=[col for col in DROP_CUTOFF_COLUMNS if col in df.columns], inplace=True)
    
    # Converts all column names to lowercase and strips whitespace
    df.columns = [col.strip().lower() for col in df.columns]

    # Renames columns for consistency
    df.rename(columns={'co_ies_curso': 'co_curso'}, inplace=True)
    
    # Converts all text columns to uppercase and strips whitespace
    text_columns = ['no_ies', 'sg_ies', 'no_campus', 'no_municipio_campus', 'sg_uf_campus', 
                    'no_curso', 'ds_grau', 'ds_turno', 'ds_mod_concorrencia']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].str.strip().str.upper()
    
    # Ensures 'nu_notacorte' is numeric
    df['nu_notacorte'] = pd.to_numeric(df['nu_notacorte'], errors='coerce')
    df.dropna(subset=['nu_notacorte'], inplace=True)

    # Creates a key for each course
    df['chave_curso'] = create_course_key(df)
    key_column = df.pop('chave_curso')
    df.insert(1, 'chave_curso', key_column)

    return df

def process_vacancy_file(file_path):
    """
    Processa um único arquivo XLSX do SISU, limpando e padronizando os dados.
    Esta é a sua função original, levemente ajustada.
    """
    df = pd.read_excel(file_path, sheet_name=1)

    # Creates 'EDICAO' column if it doesn't exist
    if 'EDICAO' not in df.columns:
        df["EDICAO"] = df["NU_ANO"].astype(str) + "/" + df["NU_EDICAO"].astype(str)
    
    # Aplicando a função no seu pipeline
    df['NO_CAMPUS'] = df['NO_CAMPUS'].apply(normalize_campus_name)

    df['DS_MOD_CONCORRENCIA'] = df['DS_MOD_CONCORRENCIA'].apply(normalize_concurrency)
    df = df[df['DS_MOD_CONCORRENCIA'] != '']

    df.rename(columns={'CO_IES_CURSO': 'co_curso'}, inplace=True)

    # Converts all column names to lowercase and strips whitespace
    df.columns = [col.strip().lower() for col in df.columns]

    # Converts all text columns to uppercase and strips whitespace
    for col in ['no_campus', 'ds_grau', 'ds_turno']:
        if col in df.columns:
            df[col] = df[col].str.strip().str.upper()

    df['chave_curso'] = create_course_key(df)
    key_column = df.pop('chave_curso')
    df.insert(0, 'chave_curso', key_column)

    # Removes unnecessary columns if they exist
    df.drop(columns=[col for col in DROP_VACANCIES_COLUMNS if col in df.columns], inplace=True)

    return df

def process_data(raw_data_dir, output_dir):
    """
    Lê todos os arquivos XLSX de um diretório, processa cada um,
    junta todos em um único DataFrame e salva como Parquet.
    """
    all_cutoff_dfs = []
    all_vacancies_dfs = []
    
    # Process each file in the directory
    for filename in os.listdir(raw_data_dir):
        if filename.endswith('_notasdecorte.xlsx'):
            file_path = os.path.join(raw_data_dir, filename)
            print(f'- Processing {filename}...')
            df_cutoff = process_cutoff_file(file_path)
            all_cutoff_dfs.append(df_cutoff)
        if filename.endswith('_vagas.xlsx'):
            file_path = os.path.join(raw_data_dir, filename)
            print(f'- Processing {filename}...')
            df_vacancies = process_vacancy_file(file_path)
            all_vacancies_dfs.append(df_vacancies)
            
    if not all_cutoff_dfs or not all_vacancies_dfs:
        print('No .xlsx files found.')
        return None
    
    if len(all_cutoff_dfs) != len(all_vacancies_dfs):
        print('Unable to merge cutoff and vacancies DFs.')
        return None
    
    merged_dfs = []

    # Merges and concatenates all DFs
    for i in range(len(all_cutoff_dfs)):
        merged = pd.merge(
            left=all_cutoff_dfs[i],
            right=all_vacancies_dfs[i],
            on=['chave_curso', 'ds_mod_concorrencia'],
            how='left'
        )
        merged_dfs.append(merged)

    final_df = pd.concat(merged_dfs, ignore_index=True)
    
    # Ensures that 'edicao' column is string type
    final_df['edicao'] = final_df['edicao'].astype(str)
    
    # Creates lag features
    final_df.sort_values(by=['chave_curso', 'ds_mod_concorrencia', 'edicao'], inplace=True)
    final_df['nota_edicao_anterior'] = final_df.groupby(['chave_curso', 'ds_mod_concorrencia'])['nu_notacorte'].shift(1)
    final_df['vagas_edicao_anterior'] = final_df.groupby(['chave_curso', 'ds_mod_concorrencia'])['qt_vagas_concorrencia'].shift(1)
    
    # Creates tendency features
    # Create a lag of 2 periods to know the score from two years ago
    nota_t_menos_2 = final_df.groupby(['chave_curso', 'ds_mod_concorrencia'])['nu_notacorte'].shift(2)
    # Tendency is the difference between last year and two years ago
    final_df['tendencia_nota'] = final_df['nota_edicao_anterior'] - nota_t_menos_2

    # Creates demand features
    # Creates lag of the number of subscribers
    final_df['inscritos_edicao_anterior'] = final_df.groupby(['chave_curso', 'ds_mod_concorrencia'])['qt_inscricao'].shift(1)
    # Calculate proportion
    final_df['demanda_anterior'] = final_df['inscritos_edicao_anterior'] / (final_df['vagas_edicao_anterior'] + 1)

    # Fills NaNs created on new features with 0 (0 means stability)
    final_df.fillna({'tendencia_nota': 0, 'demanda_anterior': 0}, inplace=True)

    # Saves to Parquet
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    output_path = os.path.join(output_dir, 'final_data.parquet')
    final_df.to_parquet(output_path, index=False)
    
    print(f'\nProcessing finished and Parquet saved to: {output_dir}')
    return final_df
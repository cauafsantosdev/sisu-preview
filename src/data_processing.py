import os
import re
import pandas as pd


# List of columns to drop from the cutoff dataset
DROP_CUTOFF_COLUMNS = ['NU_ANO', 'TP_MODALIDADE', 'DS_REGIAO_CAMPUS', 'NU_PERCENTUAL_BONUS', 
                       'DS_ORGANIZACAO_ACADEMICA', 'TP_MOD_CONCORRENCIA', 'CO_CAMPUS', 
                       'TIPO_CONCORRENCIA', "NU_EDICAO", "DS_CATEGORIA_ADM"]

# List of columns to drop from the vacancies dataset
DROP_VACANCIES_COLUMNS = ['co_ies', 'co_curso', 'ds_categoria_adm', 'ds_grau', 'no_municipio_campus', 
                   'ds_organizacao_academica', 'ds_periodicidade', 'ds_regiao', 'ds_turno', 'no_campus',
                   'no_curso', 'no_ies', 'nu_ano', 'nu_edicao', 'nu_percentual_bonus', 'nu_perc_i',
                   'nu_perc_lei', 'nu_perc_pcd', 'nu_perc_ppi', 'nu_perc_ppi_def', 'nu_perc_pp', 'sg_uf_campus',
                   'nu_perc_q', 'nu_vagas_autorizadas', 'perc_uf_i', 'perc_uf_ibge_i', 'perc_uf_ibge_pcd', 
                   'perc_uf_ibge_ppi', 'perc_uf_ibge_pp', 'perc_uf_ibge_q', 'perc_uf_pcd', 'perc_uf_pp', 
                   'perc_uf_ppid', 'perc_uf_pre_ppi', 'perc_uf_q', 'qt_semestre', 'qt_vagas_concorrencia', 
                   'qt_vagas_ofertadas', 'sg_ies', 'tp_cota', "tp_modalidade", "tp_mod_concorrencia"]

def normalize_campus_name(name: str) -> str:
    """
    Cleans and standardizes campus names.
    """
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

def normalize_modality(modality: str) -> str:
    """
    Normalizes a modality description string to a standardized code.
    """
    if "renda familiar bruta per capita igual ou inferior" in modality:
        if "pretos" in modality:
            return "LB_PPI"
        elif "quilombolas" in modality:
            return "LB_Q"
        elif "deficiência" in modality:
            return "LB_PCD"
        return "LB_EP"
        
    elif "independentemente da renda" in modality:
        if "pretos" in modality:
            return "LI_PPI"
        elif "quilombolas" in modality:
            return "LI_Q"
        elif "deficiência" in modality:
            return "LI_PCD"
        return "LI_EP"
    
    elif "Ampla" in modality:
        return "AC"
    
    return ""

def create_course_key(df):
    """
    Creates a unique identifier key for each course.
    """
    return (df['co_ies'].astype(str) + '_' +
            df['co_curso'].astype(str) + '_' +
            df['no_campus'].astype(str) + '_' +
            df['ds_grau'].astype(str) + '_' +
            df['ds_turno'].astype(str) + '_' +
            df['ds_mod_concorrencia'].astype(str))

def process_cutoff_file(file_path: str, filename: str, output_dir: str):
    """
    Processes a single SISU cutoff XLSX file, cleaning and standardizing the data.
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
        # Ensure everything is string
        edicao_str = df['EDICAO'].astype(str)
        
        # Get the year part
        ano_str = edicao_str.str.split('/').str[0]
        
        # Convert to number, forcing "nan" and other errors to become NaN
        ano_num = pd.to_numeric(ano_str, errors='coerce')
        
        # Handle NaNs.
        if ano_num.isnull().any():
            print(f"    -> WARNING: Found NaN values in EDICAO column. Filling with 0.")
            ano_num = ano_num.dropna(0)
            
        # Convert to int
        year_column = ano_num.astype(int)
        df.insert(1, 'ano', year_column)

    # Renames 'QT_VAGAS_OFERTADAS' to 'qt_vagas_concorrencia' if it exists
    if 'QT_VAGAS_OFERTADAS' in df.columns:
        df.rename(columns={'QT_VAGAS_OFERTADAS': 'qt_vagas_concorrencia'}, inplace=True)

    # Normalizing modality names
    df['DS_MOD_CONCORRENCIA'] = df['DS_MOD_CONCORRENCIA'].apply(normalize_modality)
    df = df[df['DS_MOD_CONCORRENCIA'] != '']

    # Normalizing campus names
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

    output_name = filename.replace('.xlsx', '.parquet')
    df.to_parquet(os.path.join(output_dir, output_name))

def process_vacancy_file(file_path: str, filename: str, output_dir: str):
    """
    Processes a single SISU vacancies XLSX file, cleaning and standardizing the data.
    """
    df = pd.read_excel(file_path, sheet_name=1)

    # Creates 'EDICAO' column if it doesn't exist
    if 'EDICAO' not in df.columns:
        df["EDICAO"] = df["NU_ANO"].astype(str) + "/" + df["NU_EDICAO"].astype(str)
        edition_column = df.pop('EDICAO')
        df.insert(0, 'edicao', edition_column)
    
    # Normalizing campus names
    df['NO_CAMPUS'] = df['NO_CAMPUS'].apply(normalize_campus_name)

    # Normalizing modality names
    df['DS_MOD_CONCORRENCIA'] = df['DS_MOD_CONCORRENCIA'].apply(normalize_modality)
    df = df[df['DS_MOD_CONCORRENCIA'] != '']

    # Renames columns for consistency
    df.rename(columns={'CO_IES_CURSO': 'co_curso'}, inplace=True)

    # Converts all column names to lowercase and strips whitespace
    df.columns = [col.strip().lower() for col in df.columns]

    # Edge case column name normalization
    if 'nota_minima_redacao' in df.columns:
        df = df.rename(columns={'nota_minima_redacao': 'nota_minima_redacao'})

    # Converts all text columns to uppercase and strips whitespace
    for col in ['no_campus', 'ds_grau', 'ds_turno']:
        if col in df.columns:
            df[col] = df[col].str.strip().str.upper()

    # Creates a key for each course
    df['chave_curso'] = create_course_key(df)
    key_column = df.pop('chave_curso')
    df.insert(1, 'chave_curso', key_column)

    # Removes unnecessary columns if they exist
    df.drop(columns=[col for col in DROP_VACANCIES_COLUMNS if col in df.columns], inplace=True)

    output_name = filename.replace('.xlsx', '.parquet')
    df.to_parquet(os.path.join(output_dir, output_name))

def process_data(raw_data_dir, output_dir):
    """
    Processes raw XLSX files and saves individual cleaned Parquet files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(raw_data_dir):
        file_path = os.path.join(raw_data_dir, filename)
        
        if filename.endswith('_notasdecorte.xlsx'):
            print(f'- Processing {filename}...')
            process_cutoff_file(file_path, filename, output_dir)
            
        elif filename.endswith('_vagas.xlsx'):
            print(f'- Processing {filename}...')
            process_vacancy_file(file_path, filename, output_dir)
            
    print(f'\nCheckpoint processing completed in: {output_dir}')
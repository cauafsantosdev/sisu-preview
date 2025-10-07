import pandas as pd
import os


# List of columns to drop from the dataset
COLUMNS_TO_DROP = ['NO_MUNICIPIO_CAMPUS', 'NU_ANO', 'TP_MODALIDADE', 'DS_REGIAO_CAMPUS', 
                    'NU_PERCENTUAL_BONUS', 'DS_ORGANIZACAO_ACADEMICA', 'TP_MOD_CONCORRENCIA', 
                    'CO_CAMPUS', 'TIPO_CONCORRENCIA', "NU_EDICAO", "SG_UF_CAMPUS", "DS_CATEGORIA_ADM"]

def process_single_file(file_path):
    """
    Processa um único arquivo XLSX do SISU, limpando e padronizando os dados.
    Esta é a sua função original, levemente ajustada.
    """
    df = pd.read_excel(file_path, sheet_name=1)

    # Creates 'EDICAO' column if it doesn't exist
    if 'EDICAO' not in df.columns:
        df["EDICAO"] = df["NU_ANO"].astype(str) + "_" + df["NU_EDICAO"].astype(str)
        edition_column = df.pop('EDICAO')
        df.insert(0, 'EDICAO', edition_column)
        df["EDICAO"] = df['EDICAO'].str.replace('/', '_')

    # Renames 'QT_VAGAS_OFERTADAS' to 'qt_vagas_concorrencia' if it exists
    if 'QT_VAGAS_OFERTADAS' in df.columns:
        df.rename(columns={'QT_VAGAS_OFERTADAS': 'qt_vagas_concorrencia'}, inplace=True)

    # Removes unnecessary columns if they exist
    df.drop(columns=[col for col in COLUMNS_TO_DROP if col in df.columns], inplace=True)
    
    # Converts all column names to lowercase and strips whitespace
    df.columns = [col.strip().lower() for col in df.columns]

    # Renames columns for consistency
    df.rename(columns={'co_ies_curso': 'co_curso'}, inplace=True)
    
    # Converts all text columns to uppercase and strips whitespace
    text_columns = ['no_ies', 'sg_ies', 'no_campus', 'no_curso', 'ds_grau', 'ds_turno', 'ds_mod_concorrencia']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].str.strip().str.upper()
    
    # Ensures 'nu_notacorte' is numeric
    df['nu_notacorte'] = pd.to_numeric(df['nu_notacorte'], errors='coerce')
    df.dropna(subset=['nu_notacorte'], inplace=True)

    return df

def consolidate_data(raw_data_dir, output_dir):
    """
    Lê todos os arquivos XLSX de um diretório, processa cada um,
    junta todos em um único DataFrame e salva como Parquet.
    """
    all_dfs = []
    
    # Process each file in the directory
    for filename in os.listdir(raw_data_dir):
        if filename.endswith('_notasdecorte.xlsx'):
            file_path = os.path.join(raw_data_dir, filename)
            print(f"- Processing {filename}...")
            df_year = process_single_file(file_path)
            all_dfs.append(df_year)
            
    if not all_dfs:
        print("No .xlsx files found.")
        return None

    print("\nCreating final Parquet...")
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    # Ensures that 'edicao' column is string type
    final_df['edicao'] = final_df['edicao'].astype(str)
    
    # Creates a key for each course
    final_df['chave_curso'] = (
        final_df['co_ies'].astype(str) + '_' +
        final_df['co_curso'].astype(str) + '_' +
        final_df['ds_grau'] + '_' +
        final_df['ds_turno']
    )
    
    # Creates lag features
    final_df.sort_values(by=['chave_curso', 'ds_mod_concorrencia', 'edicao'], inplace=True)
    final_df['nota_edicao_anterior'] = final_df.groupby(['chave_curso', 'ds_mod_concorrencia'])['nu_notacorte'].shift(1)
    final_df['vagas_edicao_anterior'] = final_df.groupby(['chave_curso', 'ds_mod_concorrencia'])['qt_vagas_concorrencia'].shift(1)
    
    # 2. Feature de Tendência
    # Criamos o lag de 2 períodos para saber a nota do ano retrasado
    nota_t_menos_2 = final_df.groupby(['chave_curso', 'ds_mod_concorrencia'])['nu_notacorte'].shift(2)
    # A tendência é a diferença entre o ano passado e o ano retrasado
    final_df['tendencia_nota'] = final_df['nota_edicao_anterior'] - nota_t_menos_2

    # 3. Feature de Demanda
    # Criamos o lag do número de inscritos
    final_df['inscritos_edicao_anterior'] = final_df.groupby(['chave_curso', 'ds_mod_concorrencia'])['qt_inscricao'].shift(1)
    # Calculamos a proporção (adicionamos 1 no denominador para evitar divisão por zero)
    final_df['demanda_anterior'] = final_df['inscritos_edicao_anterior'] / (final_df['vagas_edicao_anterior'] + 1)

    # Preencher NaNs criados nas novas features com 0 (uma tendência 0 significa estabilidade)
    final_df.fillna({'tendencia_nota': 0, 'demanda_anterior': 0}, inplace=True)

    # Saves to Parquet
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    output_path = os.path.join(output_dir, 'final_data.parquet')
    final_df.to_parquet(output_path, index=False)
    
    print(f"\nProcessing finished and Parquet saved to: {output_dir}")
    return final_df
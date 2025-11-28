import pytest
import pandas as pd
from src.data_processing import normalize_modality, normalize_campus_name, create_course_key


@pytest.mark.parametrize("raw_string, expected_output", [
    ("Ampla Concorrência", "AC"),

    ("Candidatos... renda familiar bruta per capita igual ou inferior... pretos...", "LB_PPI"),

    ("Candidatos... independentemente da renda...", "LI_EP"),

    ("Bônus regional de 10%", ""),
])
def test_normalize_concurrency(raw_string, expected_output):
    """
    Testing normalize_concurreny function with multiple cases
    """
    result = normalize_modality(raw_string)
    assert result == expected_output

@pytest.mark.parametrize("raw_string, expected_output", [
    ("Unidade Sede", "SEDE"),

    ("Campus de Rio Grande", "CAMPUS RIO GRANDE"),

    ("Campus Universitario - CARREIROS", "CAMPUS CARREIROS"),

    (" Carreiros ", "CARREIROS"),
])
def test_normalize_campus_name(raw_string, expected_output):
    """
    Testing normalize_campus_name function with multiple cases
    """
    result = normalize_campus_name(raw_string)
    assert result == expected_output

@pytest.fixture
def fake_df():
    """
    Fake DataFrame for testing create_course_key function
    """
    return pd.DataFrame({
        'edicao': ['2023/1'],
        'co_ies': [12],
        'co_curso': [666], 
        'no_campus': ['CARREIROS'],
        'ds_grau': ['BACHARELADO'],
        'ds_turno': ['INTEGRAL'],
        'ds_mod_concorrencia': ['AC']
    })

def test_create_course_key(fake_df):
    """
    Testing create_course_key function success
    """
    fake_df['chave_curso'] = create_course_key(fake_df)

    assert fake_df['chave_curso'].loc[0] == "2023/1_12_666_CARREIROS_BACHARELADO_INTEGRAL_AC"
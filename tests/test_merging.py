import pytest
import pandas as pd


@pytest.fixture
def df_cutoff_fake():
    """
    Fake cutoff DataFrame
    """
    return pd.DataFrame({
        "chave_curso": ["curso_A_AC", "curso_B_LB_PPI"],
        "ds_mod_concorrencia": ["AC", "LB_PPI"],
        "nu_notacorte": [700, 600]
    })

@pytest.fixture
def df_vacancy_fake():
    """
    Fake vacancy DataFrame that matches fake cutoff DataFrame
    """
    return pd.DataFrame({
        "chave_curso": ["curso_A_AC", "curso_B_LB_PPI"],
        "ds_mod_concorrencia": ["AC", "LB_PPI"],
        "peso_redacao": [4, 3]
    })

@pytest.fixture
def df_vacancy_wrong_fake():
    """
    Fake vacancy DataFrame that don't matches fake cutoff DataFrame on both merge fail cases
    """
    return pd.DataFrame({
        "chave_curso": ["curso_A_LI_EP", "curso_B_LI_PPI"],
        "ds_mod_concorrencia": ["AC", "LB_PPI"],
        "peso_redacao": [4, 3]
    })

def test_merge_succeeds(df_cutoff_fake, df_vacancy_fake):
    """
    Testing success case of merging
    """
    df_final = pd.merge(
        left=df_cutoff_fake,
        right=df_vacancy_fake,
        on=["chave_curso", "ds_mod_concorrencia"],
        how="left"
    )

    assert len(df_final) == 2
    assert df_final['peso_redacao'].isnull().sum() == 0

def test_merge_fails(df_cutoff_fake, df_vacancy_wrong_fake):
    """
    Testing fail case of merging
    """
    df_final = pd.merge(
        left=df_cutoff_fake,
        right=df_vacancy_wrong_fake,
        on=["chave_curso", "ds_mod_concorrencia"],
        how="left"
    )

    assert len(df_final) == 2
    assert df_final['peso_redacao'].isnull().sum() == 2
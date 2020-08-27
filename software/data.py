"""Módulo para leitura dos dados entrada e
separação nos conjuntos de treinamento, validação e testes."""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def read_data(arquivo):
    """ Método para ler um arquivo e separar os conjuntos de dados.

    Params:
        arquivo (str): Nome do arquivo com os conjuntos de dados.

    Returns:
        DataFrame: Objeto com o conjunto de dados de treinamento.
        DataFrame: Objeto com o conjunto de dados de validação.
        DataFrame: Objeto com o conjunto de dados de testes.
    """
    # Leitura do arquivo com os dados de entrada.
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df_data = pd.read_csv(dir_path + "/" + arquivo)
    # Separando valores das notas falsas (1) e verdadeiras (0).
    df_f = df_data[df_data["class"] == 1]
    df_v = df_data[df_data["class"] == 0]
    # Criação dos conjuntos de treinamento, validação e de testes.
    # Separação 80% e 20% entre treinamento/validação e testes.
    array_train_val_f, array_test_f = train_test_split(
        df_f.to_numpy(), test_size=0.2)
    array_train_val_v, array_test_v = train_test_split(
        df_v.to_numpy(), test_size=0.2)
    # Separação 80% e 20% entre treinamento e validação.
    df_train_val_f = pd.DataFrame(
        data=array_train_val_f, columns=df_data.columns)
    array_train_f, array_validate_f = train_test_split(
        df_train_val_f.to_numpy(), test_size=0.2)
    df_train_val_v = pd.DataFrame(
        data=array_train_val_v, columns=df_data.columns)
    array_train_v, array_validate_v = train_test_split(
        df_train_val_v.to_numpy(), test_size=0.2)
    # Junção dos dados de V e F para criação dos DataFrames de treinamento, validação e testes.
    array_train = np.concatenate((array_train_f, array_train_v), axis=0)
    array_validate = np.concatenate(
        (array_validate_f, array_validate_v), axis=0)
    array_test = np.concatenate((array_test_f, array_test_v), axis=0)
    df_train = pd.DataFrame(data=array_train, columns=df_data.columns)
    df_validate = pd.DataFrame(data=array_validate, columns=df_data.columns)
    df_test = pd.DataFrame(data=array_test, columns=df_data.columns)
    return df_train, df_validate, df_test

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import EngFormatter
from sklearn.metrics import PredictionErrorDisplay

from .models import RANDOM_STATE

sns.set_theme(palette="bright")

PALETTE = "coolwarm"
SCATTER_ALPHA = 0.2


def plot_coeficientes(df_coefs, tituto="Coeficientes"):
    df_coefs.plot.barh()
    plt.title(tituto)
    plt.axvline(x=0, color=".5")
    plt.xlabel("Coeficientes")
    plt.gca().get_legend().remove()
    plt.show()


def plot_residuos(y_true, y_pred):
    residuos = y_true - y_pred

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    sns.histplot(residuos, kde=True, ax=axs[0])

    error_display_01 = PredictionErrorDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, kind="residual_vs_predicted", ax=axs[1]
    )

    error_display_02 = PredictionErrorDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, kind="actual_vs_predicted", ax=axs[2]
    )

    plt.tight_layout()

    plt.show()


def plot_residuos_estimador(estimator, X, y, eng_formatter=False, fracao_amostra=0.25):

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    error_display_01 = PredictionErrorDisplay.from_estimator(
        estimator,
        X,
        y,
        kind="residual_vs_predicted",
        ax=axs[1],
        random_state=RANDOM_STATE,
        scatter_kwargs={"alpha": SCATTER_ALPHA},
        subsample=fracao_amostra,
    )

    error_display_02 = PredictionErrorDisplay.from_estimator(
        estimator,
        X,
        y,
        kind="actual_vs_predicted",
        ax=axs[2],
        random_state=RANDOM_STATE,
        scatter_kwargs={"alpha": SCATTER_ALPHA},
        subsample=fracao_amostra,
    )

    residuos = error_display_01.y_true - error_display_01.y_pred

    sns.histplot(residuos, kde=True, ax=axs[0])

    if eng_formatter:
        for ax in axs:
            ax.yaxis.set_major_formatter(EngFormatter())
            ax.xaxis.set_major_formatter(EngFormatter())

    plt.tight_layout()

    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

def plot_comparar_metricas_modelos(df_resultados):
    fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True)

    comparar_metricas = [
        "time_seconds",
        "test_r2",
        "test_neg_mean_absolute_error",
        "test_neg_root_mean_squared_error",
    ]

    nomes_metricas = [
        "Tempo (s)",
        "R²",
        "MAE",
        "RMSE",
    ]

    for ax, metrica, nome in zip(axs.flatten(), comparar_metricas, nomes_metricas):
        sns.stripplot(
            x="model",
            y=metrica,
            data=df_resultados,
            ax=ax,
            jitter=True,  # Adjust to spread points to avoid overlap
            dodge=True,   # Separates points by 'model'
            color='black',  # Color of the points
            alpha=0.7,     # Transparency of points
        )
        ax.set_title(nome)
        ax.set_ylabel(nome)
        ax.tick_params(axis="x", rotation=90)

    plt.tight_layout()

    plt.show()




import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_grid_search_results(grid_search, top_n=10):
    """
    Process grid search results, extract relevant columns, and generate strip plots.
    
    Parameters:
        grid_search: GridSearchCV object containing the cv_results_ attribute.
        top_n: Number of top results to include in the final DataFrame.
    
    Returns:
        Processed DataFrame with selected columns.
    """
    # Step 1: Convert grid search results to DataFrame
    df_resultados = pd.DataFrame(grid_search.cv_results_)
    
    # Step 2: Extract 'preprocessor' and 'transformer' information
    df_resultados['preprocessor'] = df_resultados['params'].apply(
        lambda params: ', '.join(
            name for name, _, _ in params['regressor__preprocessor'].transformers
        )
    )
    df_resultados['transformer'] = df_resultados['params'].apply(
        lambda params: params['transformer']
    )
    
    # Step 3: Drop irrelevant columns
    cols_to_drop = [
        'mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
        'param_regressor__preprocessor', 'param_transformer', 'params',
        'mean_test_r2', 'std_test_r2', 'rank_test_r2',
        'mean_test_neg_mean_absolute_error', 'std_test_neg_mean_absolute_error',
        'rank_test_neg_mean_absolute_error', 'mean_test_neg_root_mean_squared_error',
        'std_test_neg_root_mean_squared_error'
    ]
    df_resultados = df_resultados.drop(columns=cols_to_drop)
    
    # Step 4: Rename columns for clarity
    df_resultados = df_resultados.rename(columns={'rank_test_neg_root_mean_squared_error': 'rank_rmse'})
    
    # Step 5: Standardize column names
    df_resultados.columns = df_resultados.columns.str.replace(r'^.*?_test_', 'test_', regex=True)
    
    # Step 6: Sort and select top N rows
    df_resultados = df_resultados.sort_values(by='rank_rmse').iloc[:top_n]
    
    # Step 7: Transform to long format for plotting
    df_long = df_resultados.T.reset_index().melt(id_vars='index', var_name='column', value_name='value')
    
    # Step 8: Generate plots
    unique_indices = df_long['index'].unique()[:-3]
    for index_name in unique_indices:
        filtered_data = df_long[df_long['index'] == index_name]
        cleaned_index_name = index_name.replace('test_', '')
        column_order = filtered_data['column'].unique()
        
        plt.figure(figsize=(6, 6))
        sns.stripplot(data=filtered_data, x='column', y='value', order=column_order)
        plt.title(f'{cleaned_index_name} cross validation results')
        plt.xlabel('10 best preprocessor + transformer combinations')
        plt.ylabel('')
        plt.xticks(rotation=45)
        plt.show()
    
    # Step 9: Return the processed DataFrame
    return df_resultados.loc[:, ['rank_rmse', 'preprocessor', 'transformer']].T

# #use the following strukture to have the Transpose effect
# plot_show = plot_grid_search_results(grid_search)
# plot_show.T

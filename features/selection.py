import logging
from typing import List

import cudf
import cupy as cp

from utils.logging_utils import get_logger

logger = get_logger(__name__, "features.selection")


def create_target_variable(df: cudf.DataFrame, lookahead: int = 5) -> cudf.Series:
    """
    Cria a variável alvo (y) para o modelo de classificação.
    Exemplo: Retorno futuro de 'lookahead' períodos com base na coluna 'close'.
    Retorna 1 para retorno positivo, 0 para negativo/zero.

    Notas:
    - Mantém o índice alinhado para facilitar o corte do X.
    - Remove valores NaN resultantes do shift negativo no fim da série.
    """
    if 'close' not in df.columns:
        raise ValueError("Coluna 'close' ausente para construção do alvo")

    future_returns = df['close'].shift(-lookahead) - df['close']
    target = (future_returns > 0).astype('int8')
    return target.dropna()


essential_drop_cols = {'close'}


def select_features_with_catboost(
    features_df: cudf.DataFrame,
    target_series: cudf.Series,
    num_features_to_select: int = 50
) -> List[str]:
    """
    Treina um modelo CatBoost (GPU) para encontrar as features mais importantes.

    Mantém os dados em GPU (cuDF/CuPy) para evitar cópias CPU↔GPU.
    Em fallback (GPU indisponível), tenta CPU automaticamente.
    """
    logger.info(
        f"Iniciando seleção de features com CatBoost (GPU). Candidatas: {len(features_df.columns)}"
    )

    # Alinhar X ao índice do y e remover colunas essenciais (ex.: close)
    X = features_df.loc[target_series.index]
    cols_to_use = [c for c in X.columns if c not in essential_drop_cols]
    if len(cols_to_use) != len(X.columns):
        logger.info(
            f"Removendo colunas não permitidas do conjunto preditor: {sorted(set(X.columns) - set(cols_to_use))}"
        )
    X = X[cols_to_use]
    y = target_series

    # Extrair arrays CuPy diretamente (sem cópia para CPU)
    X_gpu = X.values
    y_gpu = y.values

    # Lazy import to avoid hard dependency at module import time
    try:
        from catboost import CatBoostClassifier
    except ModuleNotFoundError as e:
        logger.error("CatBoost não está instalado. Instale 'catboost' no ambiente antes de usar esta função.")
        raise

    # CatBoost GPU classifier
    cat = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        random_seed=42,
        verbose=0,
        task_type='GPU'
    )

    logger.info("Treinando o modelo CatBoost na GPU para avaliação de features...")
    try:
        cat.fit(X_gpu, y_gpu)
    except Exception as gpu_err:
        # CPU fallback explicitamente desabilitado neste projeto
        logger.error(
            f"CatBoost GPU training failed and CPU fallback is disabled: {gpu_err}"
        )
        raise

    # Extrair importâncias e ordenar
    feature_importances = cat.get_feature_importance()
    importance_df = cudf.DataFrame({
        'feature': X.columns,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)

    logger.info("Ranking de Importância de Features (Top 10):")
    try:
        logger.info("\n" + importance_df.head(10).to_pandas().to_string(index=False))
    except Exception:
        logger.info(str(importance_df.head(10)))

    best_features = importance_df['feature'].head(num_features_to_select).to_arrow().to_pylist()
    logger.info(f"Seleção finalizada. {len(best_features)} features selecionadas.")
    return best_features

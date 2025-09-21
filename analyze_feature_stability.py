#!/usr/bin/env python3
"""
[IMPLEMENTAÇÃO FINAL] Análise de Estabilidade de Features Pós-Otimização

Este script analisa os resultados da otimização multi-objetivo e gera
relatórios detalhados sobre:
- Estabilidade das features selecionadas
- Frequência de seleção na fronteira de Pareto
- Robustez e co-seleção de features
- Visualizações interativas dos resultados

Usage:
    python analyze_feature_stability.py --study-path results/study.pkl
    python analyze_feature_stability.py --study-path results/study.pkl --output-dir analysis_results
"""

import argparse
import pickle
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import optuna

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuração de visualização
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_study(study_path: str) -> optuna.Study:
    """Carregar estudo Optuna salvo"""
    logger.info(f"Carregando estudo de: {study_path}")
    
    with open(study_path, 'rb') as f:
        study = pickle.load(f)
    
    logger.info(f"Estudo carregado: {len(study.trials)} trials")
    logger.info(f"Trials da fronteira de Pareto: {len(study.best_trials)}")
    
    return study


def analyze_feature_stability(study: optuna.Study, feature_names: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analisar estabilidade e robustez das features selecionadas
    baseado nos trials da fronteira de Pareto
    
    Returns:
        Tuple: (rank_df, trial_metrics_df)
    """
    logger.info("Iniciando análise de estabilidade das features...")
    
    # Usar apenas trials da fronteira de Pareto
    pareto_trials = study.best_trials
    
    if len(pareto_trials) == 0:
        logger.warning("Nenhum trial da fronteira de Pareto encontrado")
        return pd.DataFrame(), pd.DataFrame()
    
    logger.info(f"Analisando {len(pareto_trials)} trials da fronteira de Pareto")
    
    # Coletar features selecionadas de cada trial
    feature_selections = []
    trial_metrics = []
    
    for trial in pareto_trials:
        if 'selected_features' in trial.user_attrs:
            # Features selecionadas em cada fold do trial
            selected_features_folds = trial.user_attrs['selected_features']
            
            # Flatten das features de todos os folds
            all_selected = set()
            for fold_features in selected_features_folds:
                all_selected.update(fold_features)
            
            feature_selections.append(list(all_selected))
            
            # Métricas do trial
            trial_metrics.append({
                'trial_number': trial.number,
                'sharpe': trial.values[0],
                'turnover': trial.values[1], 
                'max_drawdown': trial.values[2],
                'ic': trial.user_attrs.get('avg_ic', 0.0),
                'feature_stability': trial.user_attrs.get('feature_stability', 0.0),
                'n_features': trial.user_attrs.get('n_features_avg', 0)
            })
    
    if len(feature_selections) == 0:
        logger.warning("Nenhuma feature selecionada encontrada nos trials")
        return pd.DataFrame(), pd.DataFrame()
    
    # Calcular frequência de seleção para cada feature
    feature_freq = defaultdict(int)
    for selected in feature_selections:
        for feature_idx in selected:
            if feature_idx < len(feature_names):  # Verificação de segurança
                feature_freq[feature_idx] += 1
    
    # Calcular porcentagem de frequência
    n_trials = len(feature_selections)
    feature_freq_pct = {idx: count/n_trials for idx, count in feature_freq.items()}
    
    # Calcular stability score (co-seleção)
    feature_stability_scores = {}
    for feature_idx in feature_freq.keys():
        # Calcular quantas vezes esta feature foi selecionada junto com outras
        co_selection_scores = []
        
        for selected in feature_selections:
            if feature_idx in selected:
                # Features co-selecionadas
                co_selected = [f for f in selected if f != feature_idx]
                
                if len(co_selected) > 0:
                    # Score baseado na consistência das co-seleções
                    stability = sum(feature_freq_pct.get(f, 0) for f in co_selected) / len(co_selected)
                    co_selection_scores.append(stability)
        
        feature_stability_scores[feature_idx] = np.mean(co_selection_scores) if co_selection_scores else 0.0
    
    # Criar DataFrame final
    results = []
    for feature_idx, freq_pct in feature_freq_pct.items():
        if feature_idx < len(feature_names):  # Verificação de segurança
            results.append({
                'feature_name': feature_names[feature_idx],
                'feature_idx': feature_idx,
                'selection_frequency': freq_pct,
                'stability_score': feature_stability_scores[feature_idx],
                'combined_score': freq_pct * 0.7 + feature_stability_scores[feature_idx] * 0.3,
                'absolute_count': feature_freq[feature_idx]
            })
    
    rank_df = pd.DataFrame(results)
    rank_df = rank_df.sort_values('combined_score', ascending=False).reset_index(drop=True)
    rank_df['rank'] = range(1, len(rank_df) + 1)
    
    trial_metrics_df = pd.DataFrame(trial_metrics)
    
    logger.info(f"Análise concluída: {len(rank_df)} features analisadas")
    
    return rank_df, trial_metrics_df


def create_visualizations(rank_df: pd.DataFrame, trial_metrics_df: pd.DataFrame, output_dir: Path):
    """Criar visualizações dos resultados"""
    logger.info("Gerando visualizações...")
    
    # 1. Top Features - Combined Score
    plt.figure(figsize=(12, 8))
    top_20 = rank_df.head(20)
    
    plt.subplot(2, 2, 1)
    plt.barh(range(len(top_20)), top_20['combined_score'])
    plt.yticks(range(len(top_20)), top_20['feature_name'], fontsize=8)
    plt.xlabel('Combined Score')
    plt.title('Top 20 Features - Combined Score')
    plt.gca().invert_yaxis()
    
    # 2. Scatter: Frequência vs Estabilidade
    plt.subplot(2, 2, 2)
    scatter = plt.scatter(rank_df['selection_frequency'], rank_df['stability_score'], 
                         c=rank_df['combined_score'], cmap='viridis', alpha=0.7)
    plt.xlabel('Selection Frequency')
    plt.ylabel('Stability Score')
    plt.title('Feature Frequency vs Stability')
    plt.colorbar(scatter, label='Combined Score')
    plt.grid(True, alpha=0.3)
    
    # 3. Distribuição dos Combined Scores
    plt.subplot(2, 2, 3)
    plt.hist(rank_df['combined_score'], bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(rank_df['combined_score'].mean(), color='red', linestyle='--', 
               label=f'Média: {rank_df["combined_score"].mean():.3f}')
    plt.xlabel('Combined Score')
    plt.ylabel('Frequência')
    plt.title('Distribuição dos Combined Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Métricas dos Trials
    if len(trial_metrics_df) > 0:
        plt.subplot(2, 2, 4)
        plt.scatter(trial_metrics_df['sharpe'], trial_metrics_df['turnover'], 
                   c=trial_metrics_df['max_drawdown'], cmap='RdYlBu_r', alpha=0.7)
        plt.xlabel('Sharpe Ratio')
        plt.ylabel('Turnover')
        plt.title('Fronteira de Pareto: Sharpe vs Turnover')
        plt.colorbar(label='Max Drawdown')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_analysis_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Gráfico interativo 3D da fronteira de Pareto
    if len(trial_metrics_df) > 0:
        fig = go.Figure(data=go.Scatter3d(
            x=trial_metrics_df['sharpe'],
            y=trial_metrics_df['turnover'],
            z=trial_metrics_df['max_drawdown'],
            mode='markers',
            marker=dict(
                size=8,
                color=trial_metrics_df['ic'],
                colorscale='Viridis',
                colorbar=dict(title='Information Coefficient'),
                showscale=True
            ),
            text=[f'Trial {t}' for t in trial_metrics_df['trial_number']],
            hovertemplate='<b>%{text}</b><br>' +
                         'Sharpe: %{x:.3f}<br>' +
                         'Turnover: %{y:.3f}<br>' +
                         'Max DD: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Fronteira de Pareto - Otimização Multi-Objetivo',
            scene=dict(
                xaxis_title='Sharpe Ratio',
                yaxis_title='Turnover',
                zaxis_title='Max Drawdown'
            ),
            width=800,
            height=600
        )
        
        fig.write_html(output_dir / 'pareto_frontier_3d.html')
    
    logger.info(f"Visualizações salvas em: {output_dir}")


def generate_report(rank_df: pd.DataFrame, trial_metrics_df: pd.DataFrame, output_dir: Path):
    """Gerar relatório detalhado em texto"""
    logger.info("Gerando relatório detalhado...")
    
    report_path = output_dir / 'feature_stability_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RELATÓRIO DE ANÁLISE DE ESTABILIDADE DE FEATURES\n")
        f.write("=" * 80 + "\n\n")
        
        # Sumário executivo
        f.write("SUMÁRIO EXECUTIVO\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total de features analisadas: {len(rank_df)}\n")
        f.write(f"Trials da fronteira de Pareto: {len(trial_metrics_df)}\n")
        
        if len(rank_df) > 0:
            f.write(f"Combined Score médio: {rank_df['combined_score'].mean():.4f}\n")
            f.write(f"Frequência de seleção média: {rank_df['selection_frequency'].mean():.4f}\n")
            f.write(f"Score de estabilidade médio: {rank_df['stability_score'].mean():.4f}\n")
        
        f.write("\n")
        
        # Top 20 features
        if len(rank_df) > 0:
            f.write("TOP 20 FEATURES MAIS ROBUSTAS\n")
            f.write("-" * 30 + "\n")
            top_20 = rank_df.head(20)
            
            for _, row in top_20.iterrows():
                f.write(f"{row['rank']:2d}. {row['feature_name']:30s} "
                       f"| Score: {row['combined_score']:.4f} "
                       f"| Freq: {row['selection_frequency']:.3f} "
                       f"| Estab: {row['stability_score']:.3f}\n")
        
        f.write("\n")
        
        # Estatísticas dos trials
        if len(trial_metrics_df) > 0:
            f.write("ESTATÍSTICAS DOS TRIALS (FRONTEIRA DE PARETO)\n")
            f.write("-" * 45 + "\n")
            f.write(f"Sharpe Ratio   - Médio: {trial_metrics_df['sharpe'].mean():.4f}, "
                   f"Melhor: {trial_metrics_df['sharpe'].max():.4f}\n")
            f.write(f"Turnover       - Médio: {trial_metrics_df['turnover'].mean():.4f}, "
                   f"Menor: {trial_metrics_df['turnover'].min():.4f}\n")
            f.write(f"Max Drawdown   - Médio: {trial_metrics_df['max_drawdown'].mean():.4f}, "
                   f"Menor: {trial_metrics_df['max_drawdown'].min():.4f}\n")
            f.write(f"Info Coeff     - Médio: {trial_metrics_df['ic'].mean():.4f}, "
                   f"Melhor: {trial_metrics_df['ic'].max():.4f}\n")
            f.write(f"N° Features    - Médio: {trial_metrics_df['n_features'].mean():.1f}\n")
        
        f.write("\n")
        
        # Recomendações
        f.write("RECOMENDAÇÕES\n")
        f.write("-" * 15 + "\n")
        
        if len(rank_df) > 0:
            high_quality = rank_df[rank_df['combined_score'] > 0.7]
            medium_quality = rank_df[(rank_df['combined_score'] > 0.5) & 
                                   (rank_df['combined_score'] <= 0.7)]
            
            f.write(f"Features de ALTA qualidade (score > 0.7): {len(high_quality)}\n")
            f.write(f"Features de MÉDIA qualidade (0.5 < score <= 0.7): {len(medium_quality)}\n")
            f.write("\n")
            
            f.write("Próximos passos recomendados:\n")
            f.write("1. Implementar as top 15-20 features no modelo de produção\n")
            f.write("2. Validar performance no período holdout\n")
            f.write("3. Monitorar estabilidade out-of-sample\n")
            f.write("4. Considerar re-otimização se performance degradar\n")
    
    logger.info(f"Relatório salvo em: {report_path}")


def export_results(rank_df: pd.DataFrame, trial_metrics_df: pd.DataFrame, output_dir: Path):
    """Exportar resultados em formatos estruturados"""
    logger.info("Exportando resultados...")
    
    # CSV das features ranqueadas
    rank_df.to_csv(output_dir / 'feature_ranking.csv', index=False)
    
    # CSV das métricas dos trials
    if len(trial_metrics_df) > 0:
        trial_metrics_df.to_csv(output_dir / 'trial_metrics.csv', index=False)
    
    # JSON com sumário
    summary = {
        'total_features_analyzed': len(rank_df),
        'pareto_trials_count': len(trial_metrics_df),
        'top_10_features': rank_df.head(10)[['feature_name', 'combined_score']].to_dict('records') if len(rank_df) > 0 else [],
        'best_metrics': {
            'best_sharpe': trial_metrics_df['sharpe'].max() if len(trial_metrics_df) > 0 else None,
            'min_turnover': trial_metrics_df['turnover'].min() if len(trial_metrics_df) > 0 else None,
            'min_max_drawdown': trial_metrics_df['max_drawdown'].min() if len(trial_metrics_df) > 0 else None
        }
    }
    
    import json
    with open(output_dir / 'analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Resultados exportados para: {output_dir}")


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description="Análise de Estabilidade de Features")
    parser.add_argument('--study-path', type=str, required=True,
                        help='Caminho para o arquivo do estudo Optuna (.pkl)')
    parser.add_argument('--output-dir', type=str, default='analysis_results',
                        help='Diretório para salvar resultados da análise')
    parser.add_argument('--feature-names-file', type=str, default=None,
                        help='Arquivo CSV com nomes das features (opcional)')
    
    args = parser.parse_args()
    
    # Criar diretório de output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Carregar estudo
    try:
        study = load_study(args.study_path)
    except Exception as e:
        logger.error(f"Erro ao carregar estudo: {e}")
        return 1
    
    # Obter nomes das features
    feature_names = []
    if args.feature_names_file:
        try:
            features_df = pd.read_csv(args.feature_names_file)
            feature_names = features_df['feature_name'].tolist()
        except Exception as e:
            logger.warning(f"Erro ao carregar nomes das features: {e}")
    
    # Se não conseguiu carregar nomes, criar genéricos
    if not feature_names and len(study.best_trials) > 0:
        # Tentar inferir número de features dos trials
        max_feature_idx = 0
        for trial in study.best_trials:
            if 'selected_features' in trial.user_attrs:
                for fold_features in trial.user_attrs['selected_features']:
                    if len(fold_features) > 0:
                        max_feature_idx = max(max_feature_idx, max(fold_features))
        
        feature_names = [f'feature_{i:03d}' for i in range(max_feature_idx + 1)]
        logger.info(f"Criados {len(feature_names)} nomes genéricos de features")
    
    if not feature_names:
        logger.error("Não foi possível determinar os nomes das features")
        return 1
    
    # Analisar estabilidade
    try:
        rank_df, trial_metrics_df = analyze_feature_stability(study, feature_names)
    except Exception as e:
        logger.error(f"Erro durante análise: {e}")
        return 1
    
    if len(rank_df) == 0:
        logger.error("Nenhum resultado de análise gerado")
        return 1
    
    # Gerar visualizações
    try:
        create_visualizations(rank_df, trial_metrics_df, output_dir)
    except Exception as e:
        logger.warning(f"Erro ao gerar visualizações: {e}")
    
    # Gerar relatório
    try:
        generate_report(rank_df, trial_metrics_df, output_dir)
    except Exception as e:
        logger.warning(f"Erro ao gerar relatório: {e}")
    
    # Exportar resultados
    try:
        export_results(rank_df, trial_metrics_df, output_dir)
    except Exception as e:
        logger.warning(f"Erro ao exportar resultados: {e}")
    
    # Sumário final
    logger.info("=" * 60)
    logger.info("ANÁLISE CONCLUÍDA")
    logger.info("=" * 60)
    logger.info(f"Features analisadas: {len(rank_df)}")
    logger.info(f"Trials da fronteira de Pareto: {len(trial_metrics_df)}")
    
    if len(rank_df) > 0:
        logger.info(f"\nTop 5 Features:")
        for i, row in rank_df.head(5).iterrows():
            logger.info(f"  {row['rank']}. {row['feature_name']} (score: {row['combined_score']:.4f})")
    
    logger.info(f"\nResultados salvos em: {output_dir}")
    logger.info("Arquivos gerados:")
    logger.info("  - feature_ranking.csv")
    logger.info("  - trial_metrics.csv")
    logger.info("  - feature_analysis_summary.png")
    logger.info("  - pareto_frontier_3d.html")
    logger.info("  - feature_stability_report.txt")
    logger.info("  - analysis_summary.json")
    
    return 0


if __name__ == "__main__":
    exit(main())

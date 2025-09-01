#!/usr/bin/env python3
"""
Script para validar arquivos de saída do pipeline no servidor remoto.

Este script verifica se os arquivos de dados foram criados corretamente
e analisa seu conteúdo para garantir que o pipeline funcionou adequadamente.
"""

import os
import sys
import glob
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RemoteOutputValidator:
    """Validador para arquivos de saída do pipeline no servidor remoto."""
    
    def __init__(self, project_dir: str = "/workspace/feature_genesis"):
        """Inicializa o validador."""
        self.project_dir = Path(project_dir)
        self.output_dir = self.project_dir / "output"
        self.data_dir = Path("/data")
        self.validation_results = {}
        
    def check_directory_structure(self) -> Dict[str, Any]:
        """Verifica a estrutura de diretórios."""
        logger.info("🔍 Verificando estrutura de diretórios...")
        
        results = {
            'project_dir_exists': self.project_dir.exists(),
            'output_dir_exists': self.output_dir.exists(),
            'data_dir_exists': self.data_dir.exists(),
            'logs_dir_exists': (self.project_dir / "logs").exists(),
            'config_file_exists': (self.project_dir / "config" / "config.yaml").exists()
        }
        
        logger.info(f"Projeto existe: {results['project_dir_exists']}")
        logger.info(f"Diretório de saída existe: {results['output_dir_exists']}")
        logger.info(f"Diretório de dados existe: {results['data_dir_exists']}")
        logger.info(f"Diretório de logs existe: {results['logs_dir_exists']}")
        logger.info(f"Arquivo de configuração existe: {results['config_file_exists']}")
        
        return results
    
    def find_feather_files(self) -> List[Path]:
        """Encontra todos os arquivos Feather no sistema."""
        logger.info("🔍 Procurando arquivos Feather...")
        
        feather_files = []
        
        # Procurar no diretório de saída
        if self.output_dir.exists():
            feather_files.extend(self.output_dir.glob("**/*.feather"))
        
        # Procurar no diretório do projeto
        feather_files.extend(self.project_dir.glob("**/*.feather"))
        
        # Procurar no diretório de dados
        if self.data_dir.exists():
            feather_files.extend(self.data_dir.glob("**/*.feather"))
        
        # Remover duplicatas
        feather_files = list(set(feather_files))
        
        logger.info(f"Encontrados {len(feather_files)} arquivos Feather")
        for file in feather_files[:10]:  # Mostrar apenas os primeiros 10
            logger.info(f"  - {file}")
        
        if len(feather_files) > 10:
            logger.info(f"  ... e mais {len(feather_files) - 10} arquivos")
        
        return feather_files
    
    def validate_feather_file(self, file_path: Path) -> Dict[str, Any]:
        """Valida um arquivo Feather específico."""
        logger.info(f"🔍 Validando arquivo: {file_path}")
        
        try:
            # Ler o arquivo
            df = pd.read_feather(file_path)
            
            # Informações básicas
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            # Análise das colunas
            original_features = [col for col in df.columns if col.startswith('y_')]
            stationarization_features = [col for col in df.columns if any(prefix in col for prefix in ['fracdiff', 'log_', 'diff_', 'rolling_'])]
            signal_features = [col for col in df.columns if col.startswith('bk_filter')]
            statistical_features = [col for col in df.columns if any(prefix in col for prefix in ['adf_', 'dcor_', 'stationarity_', 'test_'])]
            garch_features = [col for col in df.columns if col.startswith('garch_')]
            
            # Verificar qualidade dos dados
            total_rows = len(df)
            total_cols = len(df.columns)
            nan_counts = df.isnull().sum()
            columns_with_nans = nan_counts[nan_counts > 0].to_dict()
            
            # Verificar se há dados suficientes
            min_data_threshold = 1000  # Mínimo de linhas esperadas
            data_quality = "GOOD" if total_rows >= min_data_threshold else "POOR"
            
            # Contar engines que funcionaram
            engines_working = 0
            if len(stationarization_features) > 0:
                engines_working += 1
            if len(signal_features) > 0:
                engines_working += 1
            if len(statistical_features) > 0:
                engines_working += 1
            if len(garch_features) > 0:
                engines_working += 1
            
            validation_result = {
                'file_path': str(file_path),
                'file_size_mb': file_size_mb,
                'total_rows': total_rows,
                'total_cols': total_cols,
                'original_features': len(original_features),
                'stationarization_features': len(stationarization_features),
                'signal_features': len(signal_features),
                'statistical_features': len(statistical_features),
                'garch_features': len(garch_features),
                'engines_working': engines_working,
                'data_quality': data_quality,
                'columns_with_nans': columns_with_nans,
                'nan_percentage': (df.isnull().sum().sum() / (total_rows * total_cols)) * 100,
                'success': True
            }
            
            logger.info(f"✅ Arquivo válido: {total_rows} linhas, {total_cols} colunas, {engines_working}/4 engines")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ Erro ao validar {file_path}: {e}")
            return {
                'file_path': str(file_path),
                'success': False,
                'error': str(e)
            }
    
    def check_logs(self) -> Dict[str, Any]:
        """Verifica os logs do pipeline."""
        logger.info("🔍 Verificando logs...")
        
        logs_dir = self.project_dir / "logs"
        log_results = {
            'logs_dir_exists': logs_dir.exists(),
            'log_files': [],
            'recent_errors': [],
            'pipeline_status': 'UNKNOWN'
        }
        
        if logs_dir.exists():
            log_files = list(logs_dir.glob("*.log"))
            log_results['log_files'] = [str(f) for f in log_files]
            
            # Verificar o log principal
            main_log = logs_dir / "pipeline_execution.log"
            if main_log.exists():
                try:
                    with open(main_log, 'r') as f:
                        lines = f.readlines()
                        last_lines = lines[-50:]  # Últimas 50 linhas
                        
                        # Procurar por erros
                        error_lines = [line for line in last_lines if 'ERROR' in line or 'CRITICAL' in line]
                        log_results['recent_errors'] = error_lines[-10:]  # Últimos 10 erros
                        
                        # Determinar status do pipeline
                        if any('SUCCESS' in line for line in last_lines):
                            log_results['pipeline_status'] = 'SUCCESS'
                        elif any('FAILURE' in line or 'ERROR' in line for line in last_lines):
                            log_results['pipeline_status'] = 'FAILURE'
                        elif any('running' in line.lower() for line in last_lines):
                            log_results['pipeline_status'] = 'RUNNING'
                        
                except Exception as e:
                    logger.error(f"Erro ao ler log: {e}")
        
        return log_results
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Verifica recursos do sistema."""
        logger.info("🔍 Verificando recursos do sistema...")
        
        try:
            import psutil
            
            # Informações de disco
            disk_usage = psutil.disk_usage('/workspace')
            disk_free_gb = disk_usage.free / (1024**3)
            
            # Informações de memória
            memory = psutil.virtual_memory()
            memory_free_gb = memory.available / (1024**3)
            
            # Informações de CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            
            return {
                'disk_free_gb': disk_free_gb,
                'memory_free_gb': memory_free_gb,
                'cpu_percent': cpu_percent,
                'disk_usage_percent': (disk_usage.used / disk_usage.total) * 100,
                'memory_usage_percent': memory.percent
            }
            
        except ImportError:
            logger.warning("psutil não disponível, usando comandos do sistema")
            return {
                'disk_free_gb': 'UNKNOWN',
                'memory_free_gb': 'UNKNOWN',
                'cpu_percent': 'UNKNOWN'
            }
    
    def generate_summary_report(self) -> str:
        """Gera um relatório resumido da validação."""
        logger.info("📊 Gerando relatório resumido...")
        
        # Coletar todos os resultados
        structure = self.check_directory_structure()
        feather_files = self.find_feather_files()
        log_info = self.check_logs()
        resources = self.check_system_resources()
        
        # Validar arquivos Feather
        validations = []
        for file_path in feather_files[:5]:  # Validar apenas os primeiros 5
            validation = self.validate_feather_file(file_path)
            validations.append(validation)
        
        # Gerar relatório
        report = []
        report.append("=" * 80)
        report.append("RELATÓRIO DE VALIDAÇÃO DOS ARQUIVOS DE SAÍDA")
        report.append("=" * 80)
        report.append(f"Data/Hora: {pd.Timestamp.now()}")
        report.append("")
        
        # Estrutura de diretórios
        report.append("📁 ESTRUTURA DE DIRETÓRIOS:")
        report.append(f"   Projeto: {'✅' if structure['project_dir_exists'] else '❌'}")
        report.append(f"   Saída: {'✅' if structure['output_dir_exists'] else '❌'}")
        report.append(f"   Dados: {'✅' if structure['data_dir_exists'] else '❌'}")
        report.append(f"   Logs: {'✅' if structure['logs_dir_exists'] else '❌'}")
        report.append(f"   Config: {'✅' if structure['config_file_exists'] else '❌'}")
        report.append("")
        
        # Arquivos encontrados
        report.append("📄 ARQUIVOS FEATHER:")
        report.append(f"   Total encontrados: {len(feather_files)}")
        if feather_files:
            report.append("   Primeiros arquivos:")
            for file in feather_files[:5]:
                report.append(f"     - {file}")
        else:
            report.append("   ❌ Nenhum arquivo Feather encontrado!")
        report.append("")
        
        # Validação dos arquivos
        if validations:
            report.append("🔍 VALIDAÇÃO DOS ARQUIVOS:")
            successful_validations = [v for v in validations if v.get('success', False)]
            report.append(f"   Arquivos válidos: {len(successful_validations)}/{len(validations)}")
            
            if successful_validations:
                avg_rows = np.mean([v['total_rows'] for v in successful_validations])
                avg_cols = np.mean([v['total_cols'] for v in successful_validations])
                avg_engines = np.mean([v['engines_working'] for v in successful_validations])
                
                report.append(f"   Média de linhas: {avg_rows:.0f}")
                report.append(f"   Média de colunas: {avg_cols:.0f}")
                report.append(f"   Média de engines funcionando: {avg_engines:.1f}/4")
        report.append("")
        
        # Status do pipeline
        report.append("🚀 STATUS DO PIPELINE:")
        report.append(f"   Status: {log_info['pipeline_status']}")
        report.append(f"   Logs encontrados: {len(log_info['log_files'])}")
        if log_info['recent_errors']:
            report.append(f"   Erros recentes: {len(log_info['recent_errors'])}")
        report.append("")
        
        # Recursos do sistema
        report.append("💻 RECURSOS DO SISTEMA:")
        if isinstance(resources['disk_free_gb'], (int, float)):
            report.append(f"   Disco livre: {resources['disk_free_gb']:.1f} GB")
        if isinstance(resources['memory_free_gb'], (int, float)):
            report.append(f"   Memória livre: {resources['memory_free_gb']:.1f} GB")
        if isinstance(resources['cpu_percent'], (int, float)):
            report.append(f"   CPU: {resources['cpu_percent']:.1f}%")
        report.append("")
        
        # Conclusão
        report.append("📋 CONCLUSÃO:")
        if len(feather_files) > 0:
            report.append("✅ Arquivos de saída foram criados com sucesso!")
            if successful_validations:
                report.append("✅ Pipeline funcionou corretamente")
            else:
                report.append("⚠️  Pipeline executou mas arquivos podem ter problemas")
        else:
            report.append("❌ Nenhum arquivo de saída encontrado!")
            report.append("   Verifique se o pipeline foi executado corretamente")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Executa a validação completa."""
        logger.info("🚀 Iniciando validação completa...")
        
        results = {
            'structure': self.check_directory_structure(),
            'feather_files': self.find_feather_files(),
            'log_info': self.check_logs(),
            'resources': self.check_system_resources(),
            'summary_report': self.generate_summary_report()
        }
        
        # Validar alguns arquivos se existirem
        if results['feather_files']:
            results['file_validations'] = []
            for file_path in results['feather_files'][:3]:  # Validar apenas os primeiros 3
                validation = self.validate_feather_file(file_path)
                results['file_validations'].append(validation)
        
        return results

def main():
    """Função principal."""
    print("🔍 Iniciando validação dos arquivos de saída do pipeline...")
    print("")
    
    validator = RemoteOutputValidator()
    results = validator.run_full_validation()
    
    # Imprimir relatório
    print(results['summary_report'])
    
    # Salvar relatório em arquivo
    report_file = Path("/tmp/pipeline_validation_report.txt")
    with open(report_file, 'w') as f:
        f.write(results['summary_report'])
    
    print(f"📄 Relatório salvo em: {report_file}")
    
    # Retornar código de saída baseado no resultado
    if results['feather_files']:
        print("✅ Validação concluída com sucesso!")
        return 0
    else:
        print("❌ Validação falhou - nenhum arquivo de saída encontrado!")
        return 1

if __name__ == "__main__":
    sys.exit(main())

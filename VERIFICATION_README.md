# Scripts de Verificação de Arquivos de Saída

Este documento explica como verificar se os arquivos de dados foram criados corretamente no servidor remoto após a execução do pipeline.

## 📋 Visão Geral

O pipeline de feature engineering gera arquivos Feather v2 no diretório `/workspace/feature_genesis/output/` no servidor remoto. Estes scripts ajudam a verificar se:

1. ✅ Os arquivos foram criados com sucesso
2. ✅ O conteúdo dos arquivos está correto
3. ✅ Todos os engines do pipeline funcionaram
4. ✅ A qualidade dos dados está adequada

## 🚀 Scripts Disponíveis

### 1. `verify_pipeline_output.sh` (Recomendado)
**Script completo que faz tudo automaticamente**

```bash
./verify_pipeline_output.sh
```

**O que faz:**
- Conecta ao servidor remoto via vast.ai
- Sincroniza scripts de validação
- Executa verificação básica (bash)
- Executa validação detalhada (Python)
- Gera relatório completo
- Mostra estatísticas dos arquivos

### 2. `check_remote_files.sh`
**Verificação remota via SSH**

```bash
./check_remote_files.sh
```

**O que faz:**
- Conecta ao servidor remoto
- Verifica estrutura de diretórios
- Procura por arquivos Feather
- Verifica logs e processos
- Mostra recursos do sistema

### 3. `check_output_files.sh`
**Script para executar diretamente no servidor remoto**

```bash
# No servidor remoto
./check_output_files.sh
```

**O que faz:**
- Verificação local no servidor
- Conta arquivos por tipo
- Verifica logs do pipeline
- Mostra configurações

### 4. `validate_remote_output.py`
**Validação detalhada com Python**

```bash
# No servidor remoto
python validate_remote_output.py
```

**O que faz:**
- Análise detalhada dos arquivos Feather
- Verifica se todos os engines funcionaram
- Analisa qualidade dos dados
- Gera relatório estruturado

## 📊 O que Esperar

### ✅ Pipeline Funcionou Corretamente
Se o pipeline funcionou, você verá:

```
📄 ARQUIVOS FEATHER:
   Total encontrados: 11
   Primeiros arquivos:
     - /workspace/feature_genesis/output/AUDUSD/AUDUSD.feather
     - /workspace/feature_genesis/output/EURUSD/EURUSD.feather
     - /workspace/feature_genesis/output/GBPUSD/GBPUSD.feather
     ...

🔍 VALIDAÇÃO DOS ARQUIVOS:
   Arquivos válidos: 3/3
   Média de linhas: 50000
   Média de colunas: 150
   Média de engines funcionando: 4.0/4

📋 CONCLUSÃO:
✅ Arquivos de saída foram criados com sucesso!
✅ Pipeline funcionou corretamente
```

### ❌ Pipeline Falhou
Se o pipeline falhou, você verá:

```
📄 ARQUIVOS FEATHER:
   Total encontrados: 0
   ❌ Nenhum arquivo Feather encontrado!

📋 CONCLUSÃO:
❌ Nenhum arquivo de saída encontrado!
   Verifique se o pipeline foi executado corretamente
```

## 🔍 Detalhes da Validação

### Engines Verificados
O script verifica se todos os 4 engines funcionaram:

1. **StationarizationEngine** - Features: `fracdiff_*`, `log_*`, `diff_*`, `rolling_*`
2. **SignalProcessor** - Features: `bk_filter_*`
3. **StatisticalTests** - Features: `adf_*`, `dcor_*`, `stationarity_*`
4. **GARCHModels** - Features: `garch_*`

### Qualidade dos Dados
- **Linhas mínimas**: 1.000 linhas por arquivo
- **Engines funcionando**: 4/4 engines devem ter gerado features
- **Dados válidos**: Verificação de NaNs e valores extremos

### Pares de Moedas Esperados
- AUDUSD, EURAUD, EURCAD, EURCHF, EURGBP
- EURJPY, EURUSD, GBPUSD, USDCAD, USDCHF, USDJPY

## 🛠️ Solução de Problemas

### Nenhum Arquivo Encontrado
```bash
# 1. Verificar se o pipeline foi executado
ssh -p <PORT> -i ~/.ssh/id_ed25519 root@<HOST> 'ps aux | grep python'

# 2. Verificar logs
ssh -p <PORT> -i ~/.ssh/id_ed25519 root@<HOST> 'tail -f /workspace/feature_genesis/logs/pipeline_execution.log'

# 3. Executar pipeline novamente
./run_pipeline_vast.sh
```

### Arquivos Parciais ou Corrompidos
```bash
# Verificar tamanho dos arquivos
ssh -p <PORT> -i ~/.ssh/id_ed25519 root@<HOST> 'du -h /workspace/feature_genesis/output/*/*.feather'

# Verificar integridade
ssh -p <PORT> -i ~/.ssh/id_ed25519 root@<HOST> 'python validate_remote_output.py'
```

### Problemas de Recursos
```bash
# Verificar espaço em disco
ssh -p <PORT> -i ~/.ssh/id_ed25519 root@<HOST> 'df -h'

# Verificar memória
ssh -p <PORT> -i ~/.ssh/id_ed25519 root@<HOST> 'free -h'
```

## 📁 Estrutura de Arquivos Esperada

```
/workspace/feature_genesis/
├── output/
│   ├── AUDUSD/
│   │   └── AUDUSD.feather
│   ├── EURUSD/
│   │   └── EURUSD.feather
│   ├── GBPUSD/
│   │   └── GBPUSD.feather
│   └── ... (outros pares)
├── logs/
│   └── pipeline_execution.log
└── config/
    └── config.yaml
```

## 🔧 Configuração

### Variáveis de Ambiente
```bash
export SSH_KEY_PATH="$HOME/.ssh/id_ed25519"
export AUTO=1  # Para execução não interativa
```

### Dependências
- `jq` - Para parsing JSON
- `ssh` - Para conexão remota
- `nc` - Para verificação de conectividade
- `rsync` - Para sincronização de arquivos
- `pandas` - Para validação Python (instalado automaticamente)

## 📞 Comandos Úteis

### Verificar Status Rápido
```bash
# Verificar se há arquivos de saída
ssh -p <PORT> -i ~/.ssh/id_ed25519 root@<HOST> 'find /workspace/feature_genesis/output -name "*.feather" | wc -l'
```

### Verificar Logs em Tempo Real
```bash
ssh -p <PORT> -i ~/.ssh/id_ed25519 root@<HOST> 'tail -f /workspace/feature_genesis/logs/pipeline_execution.log'
```

### Baixar Arquivos de Saída
```bash
rsync -avz -e "ssh -p <PORT> -i ~/.ssh/id_ed25519" root@<HOST>:/workspace/feature_genesis/output/ ./local_output/
```

## 🎯 Próximos Passos

1. **Execute a verificação**: `./verify_pipeline_output.sh`
2. **Analise o relatório** gerado
3. **Se arquivos existem**: ✅ Pipeline funcionou!
4. **Se não existem**: Execute `./run_pipeline_vast.sh`
5. **Para análise detalhada**: Use `python validate_remote_output.py` no servidor

---

**Nota**: Todos os scripts são executáveis e incluem tratamento de erros. Se encontrar problemas, verifique os logs e recursos do sistema.

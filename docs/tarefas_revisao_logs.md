# Tarefas para Revisão dos Logs do Pipeline

## Objetivo Geral
Eliminar numeração por "Stage" nos logs e adotar mensagens orientadas à funcionalidade/etapa real do processo, com padronização, contexto e estruturação para facilitar leitura humana e análise automatizada.

---

## 📋 Fase 1: Infraestrutura de Logging Contextual

### Tarefa 1.1: Criar sistema de contexto de logging
- [ ] **Arquivo**: `utils/log_context.py`
  - Implementar `contextvars` para `run_id`, `task_id`, `pair`
  - Criar helper `bind_context(...)`
  - Garantir thread-safety para workers Dask

### Tarefa 1.2: Criar utilitários de logging estruturado
- [ ] **Arquivo**: `utils/logging_utils.py`
  - Implementar `LoggerAdapter` que injeta `component`, `event` e contexto padrão
  - Criar funções helpers: `info_event()`, `warn_event()`, `error_event()`, `critical_event()`
  - Implementar `set_log_record_factory` para preencher chaves ausentes
  - Garantir compatibilidade com formatters existentes

### Tarefa 1.3: Atualizar configuração de logging
- [ ] **Arquivo**: `config/logging.yaml`
  - Console: adicionar campos `%(event)s %(component)s`
  - JSON: incluir campos `event`, `component`, `run_id`, `task_id`, `pair`
  - Implementar fallback seguro para novos campos
  - Manter rotação de logs existente

---

## 📋 Fase 2: Refatoração do Pipeline Principal

### Tarefa 2.1: Refatorar orchestration/main.py
- [ ] **Linha 335**: Remover "Dynamic Stage 0 - Pipeline Execution..." 
  - Substituir por evento `pipeline.start`
- [ ] Adicionar logs de contexto com `run_id`, `hostname`, `dashboard_url`
- [ ] Padronizar logs de sinais/encerramento com eventos `pipeline.abort`, `cluster.shutdown`
- [ ] Implementar evento `pipeline.end` e `pipeline.summary`
- [ ] Remover todas as referências a "Stage 0" nos logs

### Tarefa 2.2: Refatorar orchestration/pipeline_orchestrator.py
- [ ] **Linha 263**: "Starting driver-side processing..." → `task.execution.start`
- [ ] **Linha 279**: "Task i/n ..." → `task.start` com `index`, `total`, `pair`
- [ ] Substituir separadores "====" por eventos estruturados
- [ ] Em `_log_cluster_diagnostics`: usar evento `cluster.ready` com `gpu_count`, `workers`, `dashboard_url`
- [ ] Em `discover_tasks`: usar `task.discovery.start` e `task.discovery.found`

### Tarefa 2.3: Refatorar orchestration/data_processor.py
- [ ] **Linha 275**: Remover "STAGE {order}" das mensagens
- [ ] **Linha 565**: Remover "STAGE {order}" das mensagens
- [ ] Implementar eventos `engine.start`/`engine.end` com campos:
  - `engine`, `order`, `desc`, `rows_before/after`, `cols_before/after`, `new_cols`
- [ ] Para operações Dask: logar `duration_ms` e eventos `io.save.*`
- [ ] Padronizar logs de persistência e wait

---

## 📋 Fase 3: Refatoração dos Engines

### Tarefa 3.1: Ajustar features/base_engine.py
- [ ] Manter logs funcionais existentes
- [ ] Substituir textos que referenciam "stage" em mensagens por nomes de operação
- [ ] Manter assinatura `_check_memory_usage(stage: str)` mas usar `operation_name` nos logs
- [ ] Não afetar chaves de DB (manter "stage" como rótulo técnico interno)

### Tarefa 3.2: Refatorar features/statistical_tests.py
- [ ] Substituir mensagens "Stage 3 ..." por eventos funcionais
- [ ] Implementar eventos: `engine.sampling`, `engine.wrapper_fit`, `engine.feature_selection`
- [ ] Adicionar contexto: modelo, folds, thresholds
- [ ] Manter funcionalidade, apenas mudar apresentação dos logs

---

## 📋 Fase 4: Validação e Testes

### Tarefa 4.1: Testes de integração
- [ ] Executar pipeline com 1-2 pares pequenos
- [ ] Verificar ausência total de "Stage [0-9]" nas mensagens
- [ ] Validar logs console e arquivo JSON
- [ ] Confirmar que DB continua funcionando (schema inalterado)

### Tarefa 4.2: Validação de critérios de aceitação
- [ ] ✅ Nenhuma mensagem de log exibe "Stage [0-9]"
- [ ] ✅ Logs mostram eventos funcionais e campos contextuais consistentes
- [ ] ✅ `config/logging.yaml` suporta novos campos sem quebras
- [ ] ✅ Arquivo JSON tem chaves `event`, `component`, `run_id`, `task_id`, `pair`
- [ ] ✅ Console fica legível, sem "ASCII art" excessiva

---

## 📋 Fase 5: Documentação e Limpeza

### Tarefa 5.1: Atualizar documentação
- [ ] Documentar nova taxonomia de eventos
- [ ] Criar guia de uso dos novos helpers de logging
- [ ] Atualizar exemplos de logs esperados

### Tarefa 5.2: Limpeza final
- [ ] Remover comentários legados sobre "stages"
- [ ] Verificar consistência em todos os módulos
- [ ] Otimizar níveis de log (DEBUG para detalhes, INFO conciso)

---

## 🎯 Alvos Prioritários (PR 1)

### Implementação Imediata
1. **orchestration/main.py:335** → `pipeline.start`
2. **orchestration/pipeline_orchestrator.py:263** → `task.execution.start`
3. **orchestration/pipeline_orchestrator.py:279** → `task.start` com contexto
4. **orchestration/data_processor.py:275, :565** → `engine.start`/`engine.end`
5. **features/statistical_tests.py** → eventos funcionais

---

## ⚠️ Riscos e Mitigações

### Risco 1: Quebra de formatters
- **Mitigação**: Implementar `LogRecordFactory`/Filter com defaults

### Risco 2: Volume excessivo de logs
- **Mitigação**: Usar DEBUG para detalhes, INFO mais conciso

### Risco 3: Incompatibilidade com workers Dask
- **Mitigação**: Usar adapters locais e ContextVars, evitar estado global

---

## 📊 Exemplo de Resultado Esperado

### Console (humano)
```
INFO  orchestration.main  pipeline.start  run_id=42 hostname=worker-01
INFO  orchestration.orchestrator  task.discovery.found  count=8 path=/data/forex
INFO  orchestration.cluster  cluster.ready  gpus=4 dashboard=http://localhost:8787
INFO  orchestration.processor  task.start  pair=EURUSD file=EURUSD_2023.parquet size_mb=512.4
INFO  features.FeatureEngineeringEngine  engine.start  pair=EURUSD order=2 desc="BK filter"
INFO  features.FeatureEngineeringEngine  engine.end    cols_before=120 cols_after=158 new_cols=38 duration_ms=8423
INFO  data_io.local_loader  io.save.end   pair=EURUSD parts=32 path=/out/EURUSD/
INFO  orchestration.processor  task.success pair=EURUSD
INFO  orchestration.main  pipeline.summary total=8 success=8 failed=0
```

### Arquivo JSON (mesmo evento, com campos extra)
```json
{"ts":"...","level":"INFO","component":"orchestration.processor","event":"task.start","run_id":42,"task_id":101,"pair":"EURUSD","filename":"EURUSD_2023.parquet","size_mb":512.4}
```

---

## 🚀 Próximos Passos

1. **Implementar** camada de logging contextual (`utils/log_context.py`, `utils/logging_utils.py`)
2. **Atualizar** `config/logging.yaml` com novos campos
3. **Refatorar** `orchestration/main.py` e `orchestration/data_processor.py`
4. **Validar** padrão antes de propagar para demais módulos
5. **Testar** com pipeline completo

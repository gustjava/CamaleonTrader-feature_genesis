# Plano de Revisão dos Logs do Pipeline (orchestration/main.py)

Objetivo: eliminar numeração por “Stage” nos logs e adotar mensagens orientadas à funcionalidade/etapa real do processo, com padronização, contexto e estruturação para facilitar leitura humana e análise automatizada.

## 1) Estado atual (resumo)

- Arquivo principal: `orchestration/main.py` (lifecycle do cluster + execução do pipeline)
  - Cabeçalho com “Dynamic Stage 0” nos logs e docstrings.
  - Ex.: `orchestration/main.py:335` — log “Dynamic Stage 0 - Pipeline Execution...”.
- Orquestração: `orchestration/pipeline_orchestrator.py`
  - Logs funcionais, porém com separadores “====” e mensagens de progresso pouco estruturadas (ex.: Task i/n).
- Processamento por par: `orchestration/data_processor.py`
  - Usa “STAGE {order}” nas mensagens de cada engine. Ex.: `orchestration/data_processor.py:275` e `:565`.
- Engines base: `features/base_engine.py`
  - Logs funcionais por engine; usa `stage` como rótulo em métricas internas/DB (ok manter no DB; evitar em mensagens de log).
- Módulos com referências de estágio (comentários e logs legados): `features/statistical_tests.py` (diversas mensagens “Stage 3 ...”), além de docstrings/config.
- Configuração: `config/logging.yaml`
  - Console: formato simples; arquivo: JSON (pythonjsonlogger). Faltam campos contextuais padrões (run_id, task_id, pair, event, component).

Geral: os loggers já seguem o namespace por módulo (`logging.getLogger(__name__)`), o que é bom. O “Stage” é mais forte em `data_processor.py` e em alguns pontos de `features/*` (especialmente StatisticalTests). 

## 2) Metas de design para os logs (funcionais e consistentes)

- Remover números de “Stage” das mensagens apresentadas ao usuário e em arquivos de log; manter apenas nomes de funcionalidades e eventos.
- Adotar eventos semânticos: `pipeline.start`, `pipeline.summary`, `task.discovery.start`, `cluster.start`, `cluster.ready`, `engine.start`, `engine.end`, `io.load.start`, `io.load.end`, `save.start`, `save.end`, `error.*`, etc.
- Incluir contexto estruturado em todos os logs relevantes: `run_id`, `task_id`, `pair`, `engine`, `hostname`, `gpu_count`, `dashboard_url`, etc.
- Padronizar níveis: INFO para fluxo normal; WARNING para condições degradadas/recuperáveis; ERROR para falhas do passo; CRITICAL para abortos/stop imediato.
- Reduzir “ASCII art”/separadores e emojis (deixar limpo e parseável). Manter console legível e arquivo JSON rico.
- Reaproveitar `logger` por namespace e padronizar campos extras via `LoggerAdapter` e/ou `Filter`/`LogRecordFactory` para não quebrar formatters.

## 3) Taxonomia de componentes e eventos

- Componentes (via `logger.name` e/ou campo `component`):
  - `orchestration.main` (lifecycle geral do pipeline)
  - `orchestration.cluster` (Dask/UCX/RMM)
  - `orchestration.orchestrator` (descoberta de tarefas, resumo, coordenação)
  - `orchestration.processor` (execução por par)
  - `features.*` (cada engine mantém seu logger por classe)
  - `data_io.*` (load/save/local/DB)
  - `utils.*` (memória/GPU/erros)
- Catálogo inicial de eventos (exemplos práticos):
  - Pipeline: `pipeline.start`, `pipeline.end`, `pipeline.summary`, `pipeline.abort`, `pipeline.error`
  - Cluster: `cluster.start`, `cluster.config.rmm`, `cluster.client.created`, `cluster.ready`, `cluster.shutdown`
  - Descoberta/filas: `task.discovery.start`, `task.discovery.found`, `task.queue.added`
  - Execução por par: `task.start`, `task.success`, `task.failure`
  - Engines: `engine.start`, `engine.end`, `engine.validate.before`, `engine.validate.after`
  - I/O: `io.load.start`, `io.load.end`, `io.save.start`, `io.save.end`
  - Monitoria: `memory.alert`, `gpu.alert`

## 4) Campos padrão em logs (arquivo JSON e, quando útil, console)

- `event`: nome do evento semântico (ex.: `engine.start`)
- `component`: origem lógica (ex.: `orchestration.processor`)
- `run_id`, `task_id`, `pair`, `engine`, `hostname`
- Outros contextuais: `gpu_count`, `workers`, `dashboard_url`, `duration_ms`, `rows_before/after`, `cols_before/after`, `new_cols`

Estratégia técnica: introduzir um adaptador/Filter que garante esses campos (com valores padrão) para que o `logging.yaml` possa referenciá-los sem causar `KeyError`. 

## 5) Alterações propostas (alto nível)

1. Infra de logging contextual
   - Criar `utils/log_context.py` com `contextvars` para `run_id`, `task_id`, `pair` e helpers `bind_context(...)`.
   - Criar `utils/logging_utils.py` com:
     - `LoggerAdapter` que injeta `component`, `event` e contexto padrão.
     - Funções helpers: `info_event(logger, event, msg, **fields)`, `warn_event(...)`, etc.
     - Opcional: `set_log_record_factory` para preencher chaves ausentes.
   - Atualizar `config/logging.yaml` para incluir campos: `%(event)s %(component)s %(run_id)s %(task_id)s %(pair)s` no formatter JSON (com fallback seguro via factory/filter).

2. Refatorar `orchestration/main.py`
   - Remover “Dynamic Stage 0 ...” dos logs; usar evento `pipeline.start`/`pipeline.end`.
     - Referência: `orchestration/main.py:335`.
   - Registrar `run_id`, `hostname`, `dashboard_url` nos logs com `event` adequado.
   - Padronizar logs de sinais/encerramento e de erro crítico com eventos (`pipeline.abort`, `cluster.shutdown`).

3. Refatorar `orchestration/pipeline_orchestrator.py`
   - Substituir separadores “====” por eventos com contexto: `task.start`, `task.success`, `task.failure`.
   - Em `_log_cluster_diagnostics`, publicar `cluster.ready` com `gpu_count`, `workers`, `dashboard_url`.
   - Em `discover_tasks`, usar `task.discovery.start` e `task.discovery.found`.

4. Refatorar `orchestration/data_processor.py`
   - Remover “STAGE {order}” nas mensagens. Ex.: `orchestration/data_processor.py:275`, `:565`.
   - Usar `engine.start`/`engine.end` com campos `engine`, `order`, `desc`, `rows_before/after`, `cols_before/after`, `new_cols`.
   - Para Dask, manter `persist()/wait()` mas logar `duration_ms` e `io.save.*` de forma padronizada.

5. Ajustes pontuais em `features/*`
   - `features/base_engine.py`: manter logs funcionais; trocar textos que referenciam “stage” em mensagens por nomes de operação (sem afetar chaves de DB). Ex.: `_check_memory_usage(stage: str)` — manter assinatura, mas em logs, usar `operation_name` no texto.
   - `features/statistical_tests.py`: substituir mensagens “Stage 3 ...” por `engine.*`/`selection.*` com contexto (modelo, folds, thresholds). Ex.: linhas com “Stage 3 sampling / wrapper fit / selected features”.

6. Configuração (`config/logging.yaml`)
   - Console: manter simples e legível, sem `stage`/números; incluir `event` e `component` curtos.
   - Arquivo JSON: incluir campos padronizados; rotação já ok.
   - Garantir compatibilidade: filtros/adapters devem prover valores default para novos campos.

## 6) Exemplo de como ficar (conceito)

Console (humano):

```
INFO  orchestration.main  pipeline.start  run_id=42 hostname=worker-01
INFO  orchestration.orchestrator  task.discovery.found  count=8 path=/data/forex
INFO  orchestration.cluster  cluster.ready  gpus=4 dashboard=http://localhost:8787
INFO  orchestration.processor  task.start  pair=EURUSD file=EURUSD_2023.feather size_mb=512.4
INFO  features.FeatureEngineeringEngine  engine.start  pair=EURUSD order=2 desc="BK filter"
INFO  features.FeatureEngineeringEngine  engine.end    cols_before=120 cols_after=158 new_cols=38 duration_ms=8423
INFO  data_io.local_loader  io.save.end   pair=EURUSD parts=32 path=/out/EURUSD/
INFO  orchestration.processor  task.success pair=EURUSD
INFO  orchestration.main  pipeline.summary total=8 success=8 failed=0
```

Arquivo JSON (mesmo evento, com campos extra):

```
{"ts":"...","level":"INFO","component":"orchestration.processor","event":"task.start","run_id":42,"task_id":101,"pair":"EURUSD","filename":"EURUSD_2023.parquet","size_mb":512.4}
```

## 7) Passo a passo de implementação

1. Introduzir infraestrutura de contexto/adapter
   - Adicionar `utils/log_context.py` e `utils/logging_utils.py` com:
     - ContextVars p/ `run_id`, `task_id`, `pair`.
     - Adapter com `component` fixo por módulo (ou derivado de `__name__`).
     - Helpers `info_event/warn_event/error_event/critical_event`.
     - Filtro/factory que garante defaults para novos campos.
2. Atualizar `config/logging.yaml`
   - Console: `%(asctime)s %(name)s %(levelname)s %(event)s %(message)s`
   - JSON: incluir `event`, `component`, `run_id`, `task_id`, `pair`.
3. Aplicar no pipeline principal
   - `orchestration/main.py`: trocar cabeçalhos “Dynamic Stage 0 ...” por `pipeline.start`/`pipeline.end` e propagar `run_id`/`hostname`.
4. Orquestrador e Processor
   - `orchestration/pipeline_orchestrator.py`: padronizar `task.*` e `cluster.*`, reduzir separadores.
   - `orchestration/data_processor.py`: remover “STAGE {order}” e usar `engine.*` com campos estruturados.
5. Engines e estatística
   - `features/base_engine.py` e `features/statistical_tests.py`: substituir menções a “Stage N” nas mensagens por eventos/nomes funcionais claros.
6. Verificação e ajustes
   - Rodar pipeline com 1–2 pares pequenos e verificar logs (console e arquivo) — checar ausencia total de “Stage [0-9]” nas mensagens.
   - Garantir que DB (tabelas de métricas/estágios) continuam funcionando sem alteração de schema (o “stage” de DB pode permanecer como rótulo técnico).

## 8) Critérios de aceitação

- Nenhuma mensagem de log exibirá “Stage [0-9]”.
- Logs mostrarão eventos funcionais e campos contextuais consistentes.
- `config/logging.yaml` suporta novos campos sem quebras (defaults presentes).
- Arquivo JSON terá chaves `event`, `component`, e (quando aplicável) `run_id`, `task_id`, `pair`.
- Console fica legível, sem “ASCII art” excessiva.

## 9) Riscos e mitigação

- Falta de campos `extra` pode quebrar formatter: mitigar com `LogRecordFactory`/Filter que adiciona defaults.
- Volume de logs: avaliar níveis (usar DEBUG para detalhes de alto volume; INFO mais conciso).
- Compatibilidade com workers Dask: garantir que setup de logging contextual não dependa de estado global não serializável; usar adapters locais e ContextVars.

## 10) Alvos concretos para começar (PR 1)

- `orchestration/main.py:335` — trocar mensagem-título por `pipeline.start`.
- `orchestration/pipeline_orchestrator.py:263` — “Starting driver-side processing ...” -> `task.execution.start` (ou `pipeline.phase.start`).
- `orchestration/pipeline_orchestrator.py:279` — “Task i/n ...” -> `task.start` com `index`, `total`, `pair`.
- `orchestration/data_processor.py:275` e `:565` — remover “STAGE {order}” e padronizar `engine.start`/`engine.end`.
- `features/statistical_tests.py` — substituir “Stage 3 ...” por eventos funcionais (sampling, split, wrapper fit, selected features).

---

Próximo passo sugerido: implementar a camada de logging contextual e atualizar `config/logging.yaml`; em seguida, refatorar `orchestration/main.py` e `orchestration/data_processor.py` para validar o padrão antes de propagar a mudança aos demais módulos.


# Backtest UI (Angular)

Interface Angular para acompanhar backtests executados pelo módulo `backtesting`.

## Pré-requisitos

- Node.js 18+
- npm ou yarn

## Instalação

```bash
cd frontend/backtest-ui
npm install
```

## Execução em desenvolvimento

1. Certifique-se de que o backend FastAPI esteja rodando (por padrão em `http://localhost:8000`).
2. Ajuste os endpoints em `src/environments/environment.ts` se necessário.
3. Inicie o servidor de desenvolvimento:

```bash
npm start
```

A aplicação ficará disponível em `http://localhost:4200`.

## Build de produção

```bash
npm run build:prod
```

Os arquivos otimizados serão gerados em `dist/backtest-ui`.

## Estrutura principal

- `src/app/services/backtest-websocket.service.ts`: mantém a conexão WebSocket e encaminha eventos em tempo real.
- `src/app/services/backtest-api.service.ts`: chamadas REST auxiliares (ex. listar modelos disponíveis).
- `src/app/components/backtest-dashboard`: componente principal com formulário, gráfico `ngx-charts` e tabelas de métricas/trades.

## Próximos passos sugeridos

- Implementar no backend um endpoint `GET /backtest/models` para popular automaticamente as opções do formulário.
- Adicionar autenticação (token/JWT) se o backend exigir proteção.
- Criar testes unitários para o `BacktestDashboardComponent` validando parsing dos eventos e atualização do gráfico.

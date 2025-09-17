import { ChangeDetectionStrategy, ChangeDetectorRef, Component, OnDestroy, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Subscription } from 'rxjs';

import {
  BacktestEvent,
  BacktestSignalEvent,
  BacktestSummaryEvent,
  TradeSummary
} from '../../models/backtest-event';
import { BacktestApiService, BacktestRequestPayload } from '../../services/backtest-api.service';
import { BacktestWebsocketService } from '../../services/backtest-websocket.service';

interface ChartSeriesPoint {
  name: string;
  value: number;
  signal?: 'BUY' | 'SELL';
}

@Component({
  selector: 'app-backtest-dashboard',
  templateUrl: './backtest-dashboard.component.html',
  styleUrls: ['./backtest-dashboard.component.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class BacktestDashboardComponent implements OnInit, OnDestroy {
  form: FormGroup;
  loadingModels = false;
  connecting = false;
  events: BacktestSignalEvent[] = [];
  summary?: BacktestSummaryEvent;
  trades: TradeSummary[] = [];
  metrics: Record<string, number> = {};
  error?: string;

  priceSeries: Array<{ name: string; series: ChartSeriesPoint[] }> = [{
    name: 'Preço',
    series: []
  }];

  readonly colorScheme = {
    domain: ['#42b883']
  };

  private socketSub?: Subscription;

  constructor(
    private readonly fb: FormBuilder,
    private readonly api: BacktestApiService,
    private readonly ws: BacktestWebsocketService,
    private readonly cdr: ChangeDetectorRef
  ) {
    this.form = this.fb.group({
      datasetPath: ['', Validators.required],
      metadataPath: ['', Validators.required],
      modelPath: ['', Validators.required],
      batchSize: [1, [Validators.required, Validators.min(1)]],
      autoStream: [true]
    });
  }

  ngOnInit(): void {
    this.fetchAvailableModels();
  }

  ngOnDestroy(): void {
    this.socketSub?.unsubscribe();
    this.ws.close();
  }

  fetchAvailableModels(): void {
    this.loadingModels = true;
    this.api.listAvailableModels().subscribe({
      next: (models) => {
        this.loadingModels = false;
        if (models?.length) {
          const first = models[0];
          this.form.patchValue({
            datasetPath: first.datasetPath,
            metadataPath: first.metadataPath,
            modelPath: first.modelPath
          });
        }
        this.cdr.markForCheck();
      },
      error: () => {
        this.loadingModels = false;
        // Mantém formulário manual caso endpoint não esteja disponível
        this.cdr.markForCheck();
      }
    });
  }

  startBacktest(): void {
    if (this.form.invalid) {
      this.form.markAllAsTouched();
      return;
    }

    const payload: BacktestRequestPayload = {
      datasetPath: this.form.value.datasetPath,
      metadataPath: this.form.value.metadataPath,
      modelPath: this.form.value.modelPath,
      batchSize: this.form.value.batchSize ?? 1
    };

    this.resetState();
    this.connecting = true;
    this.cdr.markForCheck();

    this.socketSub = this.ws.connect(payload).subscribe({
      next: (event) => this.handleEvent(event),
      error: (err) => {
        this.error = typeof err === 'string' ? err : 'Falha na conexão com o WebSocket';
        this.connecting = false;
        this.cdr.markForCheck();
      },
      complete: () => {
        this.connecting = false;
        this.cdr.markForCheck();
      }
    });
  }

  cancelBacktest(): void {
    this.ws.close();
    this.socketSub?.unsubscribe();
    this.connecting = false;
    this.cdr.markForCheck();
  }

  private resetState(): void {
    this.events = [];
    this.summary = undefined;
    this.trades = [];
    this.metrics = {};
    this.error = undefined;
    this.priceSeries = [{ name: 'Preço', series: [] }];
  }

  private handleEvent(event: BacktestEvent): void {
    if (event.type === 'tick') {
      this.events.push(event);
      this.pushChartPoint(event);
    }
    if (event.type === 'summary') {
      this.summary = event;
      this.trades = event.trades;
      this.metrics = event.metrics;
    }
    this.cdr.markForCheck();
  }

  private pushChartPoint(event: BacktestSignalEvent): void {
    const series = this.priceSeries[0].series;
    series.push({
      name: event.timestamp,
      value: event.price,
      signal: event.signal === 'BUY' || event.signal === 'SELL' ? event.signal : undefined
    });
    // Mantém apenas últimos 1000 pontos para performance
    if (series.length > 1000) {
      series.shift();
    }
    this.priceSeries = [{ name: 'Preço', series: [...series] }];
  }

  trackByTimestamp(_index: number, item: BacktestSignalEvent): string {
    return item.timestamp;
  }

  trackByTrade(_index: number, item: TradeSummary): string {
    return `${item.entry_time}-${item.exit_time}`;
  }
}

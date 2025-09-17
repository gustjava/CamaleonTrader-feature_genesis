import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

import { environment } from '../../environments/environment';
import { AvailableModel } from '../models/backtest-event';

export interface BacktestRequestPayload {
  datasetPath: string;
  metadataPath: string;
  modelPath?: string;
  batchSize?: number;
}

@Injectable({ providedIn: 'root' })
export class BacktestApiService {
  private readonly apiUrl = environment.apiUrl;

  constructor(private readonly http: HttpClient) {}

  listAvailableModels(): Observable<AvailableModel[]> {
    const endpoint = `${this.apiUrl}/backtest/models`;
    return this.http.get<AvailableModel[]>(endpoint);
  }

  runBacktest(payload: BacktestRequestPayload): Observable<BacktestRunResponse> {
    const endpoint = `${this.apiUrl}/backtest/run`;
    const serialized = this.serializePayload(payload);
    return this.http.post<BacktestRunResponse>(endpoint, serialized);
  }

  serializePayload(payload: BacktestRequestPayload): Record<string, unknown> {
    return {
      dataset_path: payload.datasetPath,
      metadata_path: payload.metadataPath,
      model_path: payload.modelPath,
      batch_size: payload.batchSize ?? 1
    };
  }
}

export interface BacktestRunResponse {
  events: unknown[];
  metrics: Record<string, number>;
  trades: unknown[];
}

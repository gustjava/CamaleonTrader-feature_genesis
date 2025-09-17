import { Injectable, NgZone } from '@angular/core';
import { Observable, ReplaySubject } from 'rxjs';

import { environment } from '../../environments/environment';
import { BacktestEvent } from '../models/backtest-event';
import { BacktestApiService, BacktestRequestPayload } from './backtest-api.service';

@Injectable({ providedIn: 'root' })
export class BacktestWebsocketService {
  private socket?: WebSocket;
  private eventStream?: ReplaySubject<BacktestEvent>;

  constructor(private readonly zone: NgZone, private readonly api: BacktestApiService) {}

  connect(payload: BacktestRequestPayload): Observable<BacktestEvent> {
    this.close();
    const stream = new ReplaySubject<BacktestEvent>(1);
    this.eventStream = stream;
    const url = environment.websocketUrl;
    const serialized = this.api.serializePayload(payload);

    this.socket = new WebSocket(url);

    this.socket.onopen = () => {
      this.socket?.send(JSON.stringify(serialized));
    };

    this.socket.onmessage = (event) => {
      this.zone.run(() => {
        try {
          const parsed = JSON.parse(event.data) as BacktestEvent;
          stream.next(parsed);
        } catch (error) {
          console.error('Erro ao analisar evento do WebSocket', error);
        }
      });
    };

    this.socket.onerror = (event) => {
      console.error('Erro na conexão WebSocket', event);
    };

    this.socket.onclose = () => {
      this.zone.run(() => stream.complete());
    };

    return stream.asObservable();
  }

  close(): void {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.close();
    }
    this.socket = undefined;
  }
}

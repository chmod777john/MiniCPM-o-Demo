import { realtimeUrl } from '../shared/session/urls';

interface Hooks {
  onEvent: (event: any) => void;
  onError: (message: string) => void;
  onClosed: () => void;
}
/** Native simplex protocol, including history and multimodal input. */
export class ChatTransport {
  private ws: WebSocket | null = null;
  private timer?: ReturnType<typeof setTimeout>;
  private closed = false;
  constructor(private hooks: Hooks) {}
  start(endpoint: string, input: unknown) {
    this.ws = new WebSocket(realtimeUrl(endpoint, 'chat'));
    let initialized = false,
      sent = false,
      completed = false;
    const fail = (message: string) => {
      if (!this.closed) this.hooks.onError(message);
      this.stop();
    };
    this.timer = setTimeout(() => fail('等待回答超时，请结束后重试。'), 600000);
    this.ws.onmessage = ({ data }) => {
      if (this.closed) return;
      try {
        const event = JSON.parse(data);
        if (event.type === 'session.queue_done' && !initialized) {
          initialized = true;
          this.ws!.send(JSON.stringify({ type: 'session.init', payload: {} }));
        } else if (event.type === 'session.created' && !sent) {
          sent = true;
          this.ws!.send(JSON.stringify({ type: 'input.append', input }));
        }
        this.hooks.onEvent(event);
        if (event.type === 'response.done') {
          completed = true;
          clearTimeout(this.timer);
          this.ws!.send(JSON.stringify({ type: 'session.close', reason: 'turn_done' }));
          this.timer = setTimeout(() => this.stop(), 5000);
        } else if (event.type === 'error') {
          fail(event.error?.message || event.message || '模型返回错误。');
        } else if (event.type === 'session.closed') {
          if (!completed)
            fail(event.diagnostic?.message || event.reason || '回答完成前连接已关闭。');
          else this.stop();
        }
      } catch (error) {
        fail(error instanceof Error ? error.message : '无法读取模型响应。');
      }
    };
    this.ws.onerror = () => fail('连接失败，请检查 API 地址、证书和跨域配置。');
    this.ws.onclose = () => {
      if (!this.closed && !completed) this.hooks.onError('连接意外断开，请重新发送。');
      this.stop();
    };
  }
  stop() {
    if (this.closed) return;
    this.closed = true;
    clearTimeout(this.timer);
    if (this.ws) {
      this.ws.onmessage = this.ws.onclose = this.ws.onerror = null;
      if (this.ws.readyState === WebSocket.OPEN)
        this.ws.send(JSON.stringify({ type: 'session.close', reason: 'client_cancel' }));
      this.ws.close();
      this.ws = null;
    }
    this.hooks.onClosed();
  }
}

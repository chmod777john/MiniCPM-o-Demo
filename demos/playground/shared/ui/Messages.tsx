import { useEffect, useRef, useState } from 'react';
import brand from '../assets/figma-brand.svg';
import { Copy, Paperclip, Pencil, RotateCcw } from 'lucide-react';
import type { Message, ContentPart } from '../session/types';
import { ContentEditor } from './ContentEditor';
import { ChatAudio } from './ChatAudio';
import { MediaPreview } from './MediaPreview';
import { IconButton } from './controls';

export function Messages({
  messages,
  compact = false,
  editable,
  onEdit,
  onResend,
}: {
  messages: Message[];
  compact?: boolean;
  editable?: boolean;
  onEdit?: (id: string, content: ContentPart[]) => void;
  onResend?: (id: string) => void;
}) {
  const [edit, setEdit] = useState<{ id: string; content: ContentPart[] } | null>(null);
  const [error, setError] = useState('');
  const [editingBusy, setEditingBusy] = useState(false);
  const closeEdit = () => {
    setEdit(null);
    setEditingBusy(false);
  };
  const bottom = useRef<HTMLDivElement>(null);
  useEffect(() => {
    bottom.current?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }, [messages]);
  return (
    <div
      className={`messages ${compact ? 'compact' : ''}`}
      aria-live="polite"
      aria-relevant="additions text"
    >
      {messages.map((message) => (
        <article
          key={message.id}
          className={`message ${message.role} ${message.failed ? 'failed' : ''}`}
        >
          <div className="message-byline">
            {message.role === 'assistant' ? (
              <>
                <img className="mini-brand" src={brand} alt="" /> MiniCPM-o
              </>
            ) : (
              '你'
            )}
          </div>
          {edit?.id === message.id && (
            <div className="message-edit">
              <ContentEditor
                onBusy={setEditingBusy}
                onCancel={() => {
                  setEdit(null);
                  setEditingBusy(false);
                }}
                onSubmit={() => {
                  if (editable && !editingBusy) {
                    onEdit?.(edit.id, edit.content);
                    setEdit(null);
                  }
                }}
                value={edit.content}
                onChange={(content) => setEdit({ ...edit, content })}
                disabled={!editable}
                onError={(e) => setError(String(e))}
              />
              {error && <p>{error}</p>}
              <button
                className="secondary-button"
                disabled={!editable || editingBusy}
                onClick={() => {
                  onEdit?.(edit.id, edit.content);
                  setEdit(null);
                }}
              >
                保存内容
              </button>
              <button className="text-button" onClick={closeEdit}>
                取消
              </button>
            </div>
          )}
          <div className="message-content">
            {Array.isArray(message.content)
              ? message.content.map((part, index) =>
                  part.type === 'text' ? (
                    <span className="ordered-text" key={index}>
                      {part.text}
                    </span>
                  ) : (
                    <MediaPreview key={index} part={part} />
                  ),
                )
              : message.text ||
                (message.pending ? (
                  <span className="typing-dots">
                    <i />
                    <i />
                    <i />
                  </span>
                ) : message.failed ? (
                  '这次回答未完成。'
                ) : (
                  '本轮没有文本输出。'
                ))}
            {message.pending && message.text && <span className="text-caret" />}
          </div>
          {message.audioPlayer ? (
            <ChatAudio player={message.audioPlayer} />
          ) : (
            message.audioUrl && <audio controls src={message.audioUrl} className="message-audio" />
          )}
          {message.cancelled && <small>已停止生成 · 保留已收到的内容</small>}
          {message.chatMeta && (
            <details className="message-meta">
              <summary>
                {message.chatMeta.mode} · Total {(message.chatMeta.totalMs / 1000).toFixed(2)}s · In{' '}
                {message.chatMeta.inputTokens ?? '—'} / Gen{' '}
                {message.chatMeta.generatedTokens ?? '—'}
              </summary>
              <p>
                TTFS（含连接与排队）
                {message.chatMeta.ttfsMs == null
                  ? '—'
                  : `${(message.chatMeta.ttfsMs / 1000).toFixed(2)}s`}{' '}
                · Generate / Process{' '}
                {message.chatMeta.processMs == null
                  ? '—'
                  : `${(message.chatMeta.processMs / 1000).toFixed(2)}s`}
              </p>
              <button
                className="text-button"
                onClick={() => {
                  void navigator.clipboard
                    .writeText(message.chatMeta!.requestId)
                    .then(() => setError('请求标识已复制'))
                    .catch(() => setError('复制失败，请手动选择标识'));
                }}
              >
                本地请求标识：{message.chatMeta.requestId}
              </button>
            </details>
          )}
          {editable && onResend && (
            <div className="message-actions">
              {message.role === 'user' && (
                <IconButton
                  label="编辑消息"
                  onClick={() => {
                    setError('');
                    setEditingBusy(false);
                    setEdit({
                      id: message.id,
                      content: structuredClone(
                        Array.isArray(message.content)
                          ? message.content
                          : [{ type: 'text', text: message.text }],
                      ),
                    });
                  }}
                >
                  <Pencil size={13} />
                </IconButton>
              )}
              <IconButton
                label={
                  message.role === 'user'
                    ? '从此处重新发送（清除后续消息）'
                    : '重新生成（清除后续消息）'
                }
                onClick={() => {
                  closeEdit();
                  onResend(message.id);
                }}
              >
                <RotateCcw size={13} />
              </IconButton>
            </div>
          )}
          {!message.pending && message.text && message.role === 'assistant' && (
            <IconButton
              label="复制回答"
              className="copy-message"
              onClick={() => {
                navigator.clipboard?.writeText(message.text).catch(() => {});
              }}
            >
              <Copy size={13} />
            </IconButton>
          )}
        </article>
      ))}
      {error && <p role="status">{error}</p>}
      <div ref={bottom} />
    </div>
  );
}

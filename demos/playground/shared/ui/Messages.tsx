import { useEffect, useRef } from 'react';
import brand from '../assets/figma-brand.svg';
import { Copy, RotateCcw } from 'lucide-react';
import type { Message } from '../session/types';
import { assistantText } from '../session/text';
import { ChatAudio } from './ChatAudio';
import { MediaPreview } from './MediaPreview';
import { IconButton } from './controls';

export function Messages({
  messages,
  compact = false,
  canRegenerate = false,
  onRegenerate,
}: {
  messages: Message[];
  compact?: boolean;
  canRegenerate?: boolean;
  onRegenerate?: (id: string) => void;
}) {
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
      {messages.map((message) => {
        const assistant = message.role === 'assistant';
        const text = assistant ? assistantText(message.text, message.pending) : message.text;
        return (
          <article
            key={message.id}
            className={`message ${message.role} ${message.failed ? 'failed' : ''}`}
          >
            <div className="message-byline">
              {assistant ? (
                <>
                  <img className="mini-brand" src={brand} alt="" /> MiniCPM-o
                </>
              ) : (
                '你'
              )}
            </div>
            <div className="message-content">
              {Array.isArray(message.content)
                ? message.content.map((part, index) =>
                    part.type === 'text' ? (
                      <span className="ordered-text" key={index}>
                        {assistant ? assistantText(part.text || '', message.pending) : part.text}
                      </span>
                    ) : (
                      <MediaPreview key={index} part={part} />
                    ),
                  )
                : text ||
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
              {message.pending && text && <span className="text-caret" />}
            </div>
            {message.audioPlayer ? (
              <ChatAudio player={message.audioPlayer} />
            ) : (
              message.audioUrl && (
                <audio controls src={message.audioUrl} className="message-audio" />
              )
            )}
            {message.cancelled && <small>已停止生成 · 保留已收到的内容</small>}
            {assistant && !message.pending && !compact && (onRegenerate || text) && (
              <div className="message-actions">
                {onRegenerate && (
                  <IconButton
                    label="重新生成（清除后续消息）"
                    disabled={!canRegenerate}
                    onClick={() => onRegenerate(message.id)}
                  >
                    <RotateCcw size={14} />
                  </IconButton>
                )}
                {text && (
                  <IconButton
                    label="复制回答"
                    onClick={() => {
                      navigator.clipboard?.writeText(text).catch(() => {});
                    }}
                  >
                    <Copy size={14} />
                  </IconButton>
                )}
              </div>
            )}
          </article>
        );
      })}
      <div ref={bottom} />
    </div>
  );
}

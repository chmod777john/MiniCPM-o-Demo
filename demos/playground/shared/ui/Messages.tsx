import { useEffect, useRef } from 'react';
import brand from '../assets/figma-brand.svg';
import { Copy, Paperclip } from 'lucide-react';
import type { Message } from '../session/types';
import { IconButton } from './controls';

export function Messages({
  messages,
  compact = false,
}: {
  messages: Message[];
  compact?: boolean;
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
          <div className="message-content">
            {message.files?.length ? (
              <div className="message-files">
                {message.files.map((name, i) => (
                  <span key={i}>
                    <Paperclip size={13} />
                    {name}
                  </span>
                ))}
              </div>
            ) : null}
            {message.text ||
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
      <div ref={bottom} />
    </div>
  );
}

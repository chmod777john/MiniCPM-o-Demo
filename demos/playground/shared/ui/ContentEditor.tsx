import { useEffect, useRef, useState } from 'react';
import { ArrowDown, ArrowUp, Plus, Trash2, Upload } from 'lucide-react';
import type { ContentPart } from '../session/types';
import { contentFiles, wavReference } from '../media/files';
import { MediaPreview } from './MediaPreview';
import { IconButton } from './controls';

export function ContentEditor({
  value,
  onChange,
  disabled,
  onError,
  onBusy,
  onSubmit,
  onCancel,
  maxAudioSeconds = Infinity,
}: {
  value: ContentPart[];
  onChange: (value: ContentPart[]) => void;
  disabled: boolean;
  onError: (error: unknown) => void;
  onBusy?: (busy: boolean) => void;
  onSubmit?: () => void;
  onCancel?: () => void;
  maxAudioSeconds?: number;
}) {
  const latest = useRef({ value, onChange });
  latest.current = { value, onChange };
  const mounted = useRef(true);
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      onBusy?.(false);
    };
  }, []);
  const locked = disabled || loading;
  const update = (index: number, part: ContentPart) =>
    onChange(value.map((p, i) => (i === index ? part : p)));
  const move = (index: number, delta: number) => {
    const next = [...value];
    [next[index], next[index + delta]] = [next[index + delta], next[index]];
    onChange(next);
  };
  return (
    <div
      className="content-editor"
      onKeyDown={(e) => {
        if (e.key === 'Escape' && onCancel) {
          e.preventDefault();
          onCancel();
        }
        if (
          e.key === 'Enter' &&
          !e.shiftKey &&
          !e.nativeEvent.isComposing &&
          onSubmit &&
          !locked &&
          (e.target as HTMLElement).tagName === 'TEXTAREA'
        ) {
          e.preventDefault();
          onSubmit();
        }
      }}
    >
      {value.map((part, index) => (
        <div className="content-editor-item" key={index}>
          <div className="content-editor-heading">
            <span>
              {index + 1} · {part.type}
            </span>
            <div>
              <IconButton
                label="上移"
                disabled={locked || index === 0}
                onClick={() => move(index, -1)}
              >
                <ArrowUp size={13} />
              </IconButton>
              <IconButton
                label="下移"
                disabled={locked || index === value.length - 1}
                onClick={() => move(index, 1)}
              >
                <ArrowDown size={13} />
              </IconButton>
              <IconButton
                label="删除内容"
                disabled={locked}
                onClick={() => onChange(value.filter((_, i) => i !== index))}
              >
                <Trash2 size={13} />
              </IconButton>
            </div>
          </div>
          {part.type === 'text' ? (
            <textarea
              aria-label={`文本 ${index + 1}`}
              rows={3}
              disabled={locked}
              value={part.text || ''}
              onChange={(e) => update(index, { ...part, text: e.target.value })}
            />
          ) : (
            <>
              <MediaPreview part={part} />
              <span className="field-hint">
                {part.name || '尚未选择文件'}
                {part.duration ? ` · ${part.duration.toFixed(1)}s` : ''}
              </span>
              <input
                aria-label={`选择 ${part.type}`}
                type="file"
                accept={`${part.type}/*`}
                disabled={locked}
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  e.target.value = '';
                  if (!file) return;
                  setLoading(true);
                  onBusy?.(true);
                  const apply = (next: ContentPart) => {
                    if (!mounted.current) return;
                    const current = latest.current;
                    if (!current.value.includes(part)) return;
                    current.onChange(current.value.map((p) => (p === part ? next : p)));
                  };
                  void (async () => {
                    if (part.type === 'audio') {
                      const audio = await wavReference(file, maxAudioSeconds);
                      apply({
                        type: 'audio',
                        name: audio.name,
                        data: audio.pcm,
                        duration: audio.duration,
                      });
                    } else {
                      const [content] = await contentFiles([file]);
                      apply(content);
                    }
                  })()
                    .catch((e) => {
                      if (mounted.current) onError(e);
                    })
                    .finally(() => {
                      if (mounted.current) {
                        setLoading(false);
                        onBusy?.(false);
                      }
                    });
                }}
              />
            </>
          )}
        </div>
      ))}
      <div className="editor-add-row">
        {(['text', 'audio', 'image', 'video'] as const).map((type) => (
          <button
            className="secondary-button"
            key={type}
            disabled={locked}
            onClick={() => onChange([...value, { type, ...(type === 'text' ? { text: '' } : {}) }])}
          >
            <Plus size={12} />
            {type}
          </button>
        ))}
      </div>
    </div>
  );
}

export function SegmentEditor({
  value,
  onChange,
  disabled,
  onError,
  onReference,
}: {
  value: string;
  onChange: (value: string) => void;
  disabled: boolean;
  onError: (error: unknown) => void;
  onReference: (path: string) => void;
}) {
  let segments: any[];
  try {
    segments = JSON.parse(value || '[]');
    if (
      !Array.isArray(segments) ||
      segments.some((s) => !s || typeof s !== 'object' || !['text', 'audio'].includes(s.kind))
    )
      throw new Error();
  } catch {
    return <p className="field-hint">JSON 无效，请先在原始 JSON 中修正。</p>;
  }
  const write = (next: any[]) => onChange(JSON.stringify(next, null, 2));
  const update = (index: number, part: any) =>
    write(segments.map((s, i) => (i === index ? part : s)));
  return (
    <div className="content-editor">
      {segments.map((part, index) => (
        <div className="content-editor-item" key={index}>
          <div className="content-editor-heading">
            <span>
              {index + 1} · {part.kind}
            </span>
            <div>
              {[-1, 1].map((delta) => (
                <IconButton
                  key={delta}
                  label={delta < 0 ? '上移' : '下移'}
                  disabled={disabled || index + delta < 0 || index + delta >= segments.length}
                  onClick={() => {
                    const next = [...segments];
                    [next[index], next[index + delta]] = [next[index + delta], next[index]];
                    write(next);
                  }}
                >
                  {delta < 0 ? <ArrowUp size={13} /> : <ArrowDown size={13} />}
                </IconButton>
              ))}
              <IconButton
                label="删除片段"
                disabled={disabled}
                onClick={() => write(segments.filter((_, i) => i !== index))}
              >
                <Trash2 size={13} />
              </IconButton>
            </div>
          </div>
          {part.kind === 'text' ? (
            <textarea
              aria-label={`System 文本 ${index + 1}`}
              value={part.text || ''}
              rows={4}
              disabled={disabled}
              onChange={(e) => update(index, { kind: 'text', text: e.target.value })}
            />
          ) : (
            <>
              <input
                aria-label={`音频路径 ${index + 1}`}
                placeholder="工具服务上的音频路径"
                disabled={disabled}
                value={part.audio?.file_path || ''}
                onChange={(e) =>
                  update(index, {
                    kind: 'audio',
                    audio: { source: 'path', file_path: e.target.value },
                  })
                }
              />
              <button
                className="text-button"
                disabled={disabled || !part.audio?.file_path}
                onClick={() => onReference(part.audio.file_path)}
              >
                设为 TTS 参考音频
              </button>
            </>
          )}
        </div>
      ))}
      <div className="editor-add-row">
        <button
          className="secondary-button"
          disabled={disabled}
          onClick={() => write([...segments, { kind: 'text', text: '' }])}
        >
          <Plus size={12} />
          文本
        </button>
        <button
          className="secondary-button"
          disabled={disabled}
          onClick={() =>
            write([...segments, { kind: 'audio', audio: { source: 'path', file_path: '' } }])
          }
        >
          <Upload size={12} />
          音频路径
        </button>
      </div>
    </div>
  );
}

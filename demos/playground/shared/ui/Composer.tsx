import { useEffect, useRef, useState } from 'react';
import {
  ArrowUp,
  AudioLines,
  FileAudio,
  Image,
  Mic,
  Paperclip,
  Plus,
  Square,
  Video,
  X,
} from 'lucide-react';
import { ContentEditor } from './ContentEditor';
import { contentFiles } from '../media/files';
import type { ContentPart } from '../session/types';
import { IconButton } from './controls';

export function Composer({
  value,
  onChange,
  onSend,
  onStop,
  busy,
  disabled,
  attachments = true,
  placeholder,
}: {
  value: string;
  onChange: (value: string) => void;
  onSend: (text: string, files: File[], ordered?: ContentPart[]) => Promise<boolean>;
  onStop: () => void;
  busy: boolean;
  disabled?: boolean;
  attachments?: boolean;
  placeholder: string;
}) {
  const [ordered, setOrdered] = useState<ContentPart[] | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState('');
  const [converting, setConverting] = useState(false);
  const [editorBusy, setEditorBusy] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const recorder = useRef<MediaRecorder | null>(null);
  const media = useRef<MediaStream | null>(null);
  const mounted = useRef(true);
  const intent = useRef(0);
  const cancelledRecording = useRef(false);
  const recordingRequested = useRef(false);
  const keyboardRecord = useRef<() => Promise<void>>(async () => {});
  const addFiles = (next: File[]) => {
    setError('');
    if (ordered) {
      setConverting(true);
      void contentFiles(next)
        .then((parts) => {
          if (mounted.current) setOrdered((old) => [...(old || []), ...parts]);
        })
        .catch((e) => setError(String(e)))
        .finally(() => {
          if (mounted.current) setConverting(false);
        });
    } else setFiles((old) => [...old, ...next]);
  };
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      intent.current++;
      if (recorder.current) {
        recorder.current.onstop = null;
        if (recorder.current.state !== 'inactive') recorder.current.stop();
      }
      media.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);
  useEffect(() => {
    if (textarea.current) {
      textarea.current.style.height = 'auto';
      textarea.current.style.height = `${Math.min(textarea.current.scrollHeight, 150)}px`;
    }
  }, [value]);
  async function record() {
    if (recording) {
      recorder.current?.stop();
      return;
    }
    if (recordingRequested.current) return;
    recordingRequested.current = true;
    cancelledRecording.current = false;
    const request = ++intent.current;
    try {
      if (!navigator.mediaDevices?.getUserMedia)
        throw new Error('录音需要通过 HTTPS 或 localhost 打开。');
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (!mounted.current || intent.current !== request) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      media.current = stream;
      const instance = new MediaRecorder(stream);
      recorder.current = instance;
      const chunks: Blob[] = [];
      instance.ondataavailable = (event) => {
        if (event.data.size) chunks.push(event.data);
      };
      instance.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        recordingRequested.current = false;
        if (!mounted.current) return;
        setRecording(false);
        if (cancelledRecording.current) return;
        const mime = instance.mimeType || 'audio/webm';
        addFiles([
          new File(chunks, `语音.${mime.includes('mp4') ? 'm4a' : 'webm'}`, { type: mime }),
        ]);
      };
      instance.start(100);
      setRecording(true);
      setError('');
    } catch (error) {
      recordingRequested.current = false;
      if (mounted.current) {
        setError(error instanceof Error ? error.message : '无法录音');
        media.current?.getTracks().forEach((t) => t.stop());
      }
    }
  }
  keyboardRecord.current = record;
  useEffect(() => {
    if (!attachments) return;
    let pressed = 0;
    const cancel = () => {
      cancelledRecording.current = true;
      recordingRequested.current = false;
      intent.current++;
      recorder.current?.state === 'recording' && recorder.current.stop();
      media.current?.getTracks().forEach((t) => t.stop());
      setRecording(false);
    };
    const down = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && recordingRequested.current) {
        event.preventDefault();
        cancel();
        return;
      }
      const focused = document.activeElement;
      if (
        event.code !== 'Space' ||
        event.repeat ||
        busy ||
        disabled ||
        converting ||
        (['INPUT', 'TEXTAREA', 'SELECT'].includes(focused?.tagName || '') &&
          !(focused === textarea.current && !value))
      )
        return;
      event.preventDefault();
      if (recordingRequested.current) {
        recorder.current?.stop();
        return;
      }
      pressed = performance.now();
      void keyboardRecord.current();
    };
    const up = (event: KeyboardEvent) => {
      if (event.code !== 'Space' || !pressed) return;
      const held = performance.now() - pressed;
      pressed = 0;
      if (held >= 300) {
        if (recorder.current?.state === 'recording') recorder.current.stop();
        else cancel();
      }
    };
    document.addEventListener('keydown', down);
    document.addEventListener('keyup', up);
    return () => {
      document.removeEventListener('keydown', down);
      document.removeEventListener('keyup', up);
    };
  }, [attachments, busy, disabled, converting, value]);
  async function submit() {
    if (
      (!value.trim() && !files.length && !ordered?.some((p) => p.text || p.data)) ||
      busy ||
      converting ||
      editorBusy ||
      disabled ||
      recording
    )
      return;
    setConverting(true);
    try {
      if (
        await onSend(
          ordered
            ? ordered
                .filter((p) => p.type === 'text')
                .map((p) => p.text)
                .join('\n')
            : value,
          ordered ? [] : files,
          ordered || undefined,
        )
      ) {
        onChange('');
        setFiles([]);
        setOrdered(null);
      }
    } finally {
      if (mounted.current) setConverting(false);
    }
  }
  return (
    <div className="composer-wrap">
      {(error || files.length > 0) && (
        <div className="attachment-list">
          {error && <span className="inline-error">{error}</span>}
          {files.map((file, i) => {
            const FileIcon = file.type.startsWith('image/')
              ? Image
              : file.type.startsWith('video/')
                ? Video
                : FileAudio;
            return (
              <span className="attachment" key={`${file.name}-${i}`}>
                <FileIcon size={14} />
                <span>{file.name}</span>
                <IconButton
                  label={`移除 ${file.name}`}
                  onClick={() => setFiles((list) => list.filter((_, j) => j !== i))}
                >
                  <X size={12} />
                </IconButton>
              </span>
            );
          })}
        </div>
      )}
      {ordered && (
        <div className="ordered-composer">
          <ContentEditor
            value={ordered}
            onChange={setOrdered}
            onBusy={setEditorBusy}
            onSubmit={() => void submit()}
            disabled={busy || converting}
            onError={(e) => setError(String(e))}
          />
        </div>
      )}
      <div
        className={`composer ${recording ? 'recording' : ''}`}
        onDragOver={(e) => {
          if (attachments && !busy) e.preventDefault();
        }}
        onPaste={(e) => {
          if (attachments && !busy && e.clipboardData.files.length) {
            e.preventDefault();
            addFiles(Array.from(e.clipboardData.files));
          }
        }}
        onDrop={(e) => {
          if (attachments && !busy) {
            e.preventDefault();
            addFiles(Array.from(e.dataTransfer.files));
          }
        }}
      >
        {!ordered && (
          <textarea
            ref={textarea}
            aria-label="消息内容"
            rows={1}
            placeholder={placeholder}
            value={value}
            disabled={disabled || busy || converting}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                void submit();
              }
            }}
          />
        )}
        <div className="composer-tools">
          <div className="composer-left">
            {attachments && (
              <>
                <input
                  ref={input}
                  type="file"
                  multiple
                  accept="image/*,audio/*,video/*"
                  className="visually-hidden"
                  onChange={(e) => {
                    addFiles(Array.from(e.target.files || []));
                    e.target.value = '';
                  }}
                />
                <IconButton
                  label="添加图片、音频或视频"
                  onClick={() => input.current?.click()}
                  disabled={busy || converting}
                >
                  <Plus size={19} />
                </IconButton>
                {!ordered && (
                  <button
                    className="text-button"
                    disabled={busy || converting}
                    onClick={() => {
                      setConverting(true);
                      void contentFiles(files)
                        .then((parts) => {
                          setOrdered([
                            ...(value ? [{ type: 'text' as const, text: value }] : []),
                            ...parts,
                          ]);
                          setFiles([]);
                        })
                        .catch((e) => setError(String(e)))
                        .finally(() => setConverting(false));
                    }}
                  >
                    编辑内容顺序
                  </button>
                )}
                <span className="composer-format">
                  <Paperclip size={12} /> 图片、音频、视频
                </span>
              </>
            )}
            {!attachments && (
              <span className="composer-format">将文本作为外部事件注入当前会话</span>
            )}
          </div>
          <div className="composer-right">
            {attachments && (
              <IconButton
                label={recording ? '结束录音' : '录制语音'}
                className={recording ? 'recording-button' : ''}
                onClick={() => void record()}
                disabled={busy || converting}
              >
                {recording ? <AudioLines size={18} /> : <Mic size={18} />}
              </IconButton>
            )}
            <button
              type="button"
              className="send-button"
              aria-label={busy ? '停止生成' : '发送消息'}
              onClick={() => (busy ? onStop() : void submit())}
              disabled={
                !busy &&
                (disabled ||
                  converting ||
                  editorBusy ||
                  recording ||
                  (!value.trim() && !files.length && !ordered?.some((p) => p.text || p.data)))
              }
            >
              {busy ? <Square size={15} fill="currentColor" /> : <ArrowUp size={20} />}
            </button>
          </div>
        </div>
      </div>
      <div className="composer-caption">
        {recording
          ? '正在录音 · 再次点击麦克风结束'
          : converting
            ? '正在准备输入…'
            : '内容由 AI 生成，请留意核实。'}
        <span>{attachments ? 'Enter 发送 · Shift + Enter 换行' : 'Enter 注入'}</span>
      </div>
    </div>
  );
}

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
  onSend: (text: string, files: File[]) => Promise<boolean>;
  onStop: () => void;
  busy: boolean;
  disabled?: boolean;
  attachments?: boolean;
  placeholder: string;
}) {
  const [files, setFiles] = useState<File[]>([]);
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState('');
  const [converting, setConverting] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const recorder = useRef<MediaRecorder | null>(null);
  const media = useRef<MediaStream | null>(null);
  const mounted = useRef(true);
  const intent = useRef(0);
  const addFiles = (next: File[]) => {
    setError('');
    setFiles((old) => [...old, ...next].slice(0, 8));
  };
  useEffect(
    () => () => {
      mounted.current = false;
      intent.current++;
      if (recorder.current) {
        recorder.current.onstop = null;
        if (recorder.current.state !== 'inactive') recorder.current.stop();
      }
      media.current?.getTracks().forEach((t) => t.stop());
    },
    [],
  );
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
        if (!mounted.current) return;
        setRecording(false);
        const mime = instance.mimeType || 'audio/webm';
        addFiles([
          new File(chunks, `语音.${mime.includes('mp4') ? 'm4a' : 'webm'}`, { type: mime }),
        ]);
      };
      instance.start();
      setRecording(true);
      setError('');
    } catch (error) {
      if (mounted.current) {
        setError(error instanceof Error ? error.message : '无法录音');
        media.current?.getTracks().forEach((t) => t.stop());
      }
    }
  }
  async function submit() {
    if ((!value.trim() && !files.length) || busy || converting || disabled || recording) return;
    setConverting(true);
    try {
      if (await onSend(value.trim(), files)) {
        onChange('');
        setFiles([]);
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
      <div
        className={`composer ${recording ? 'recording' : ''}`}
        onDragOver={(e) => {
          if (attachments && !busy) e.preventDefault();
        }}
        onDrop={(e) => {
          if (attachments && !busy) {
            e.preventDefault();
            addFiles(Array.from(e.dataTransfer.files));
          }
        }}
      >
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
                !busy && (disabled || converting || recording || (!value.trim() && !files.length))
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

import { useEffect, useRef, useState, useSyncExternalStore, type CSSProperties } from 'react';
import { Download, Pause, Play } from 'lucide-react';
import type { ChatAudioPlayer } from '../../turnbased/audio';
import './ChatAudio.css';

function formatTime(seconds: number) {
  const value = Number.isFinite(seconds) ? Math.max(0, Math.floor(seconds)) : 0;
  return `${Math.floor(value / 60)}:${String(value % 60).padStart(2, '0')}`;
}

function downloadAudio(url: string) {
  const link = document.createElement('a');
  link.href = url;
  link.download = 'assistant.wav';
  link.click();
}

function AudioBar({
  playing,
  duration,
  position,
  onToggle,
  onSeek,
  onDownload,
}: {
  playing: boolean;
  duration: number;
  position: number;
  onToggle: () => void;
  onSeek: (position: number) => void;
  onDownload: () => void;
}) {
  const length = Number.isFinite(duration) ? Math.max(0, duration) : 0;
  const cursor = Number.isFinite(position) ? Math.max(0, Math.min(position, length)) : 0;
  const progress = length ? (cursor / length) * 100 : 0;
  const label = playing ? '暂停音频' : '播放音频';
  return (
    <div className="chat-audio-bar" role="group" aria-label="回答语音">
      <button
        type="button"
        className="chat-audio-toggle"
        aria-label={label}
        title={label}
        disabled={!length}
        onClick={onToggle}
      >
        {playing ? (
          <Pause size={11} fill="currentColor" aria-hidden="true" />
        ) : (
          <Play size={11} fill="currentColor" aria-hidden="true" />
        )}
      </button>
      <input
        className="chat-audio-seek"
        type="range"
        aria-label="音频播放位置"
        aria-valuetext={`${formatTime(cursor)} / ${formatTime(length)}`}
        min={0}
        max={length || 1}
        step="any"
        value={cursor}
        disabled={!length}
        style={{ '--audio-progress': `${progress}%` } as CSSProperties}
        onChange={(event) => onSeek(Number(event.target.value))}
      />
      <span className="chat-audio-time" aria-hidden="true">
        {formatTime(cursor)} / {formatTime(length)}
      </span>
      <button
        type="button"
        className="chat-audio-download"
        aria-label="下载音频"
        title="下载音频"
        disabled={!length}
        onClick={onDownload}
      >
        <Download size={14} aria-hidden="true" />
      </button>
    </div>
  );
}

function StreamingChatAudio({ player }: { player: ChatAudioPlayer }) {
  useSyncExternalStore(player.subscribe, player.snapshot);
  return (
    <AudioBar
      playing={player.playing}
      duration={player.duration}
      position={player.position}
      onToggle={() => (player.playing ? player.pause() : player.play())}
      onSeek={(position) => player.seek(position)}
      onDownload={() => {
        const url = URL.createObjectURL(player.wav());
        downloadAudio(url);
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      }}
    />
  );
}

function FileChatAudio({ src }: { src: string }) {
  const audio = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [position, setPosition] = useState(0);
  const [error, setError] = useState('');
  useEffect(() => {
    const element = audio.current;
    return () => element?.pause();
  }, []);
  return (
    <>
      <audio
        ref={audio}
        src={src}
        preload="metadata"
        onLoadedMetadata={(event) => setDuration(event.currentTarget.duration)}
        onDurationChange={(event) => setDuration(event.currentTarget.duration)}
        onTimeUpdate={(event) => setPosition(event.currentTarget.currentTime)}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        onError={() => setError('音频暂时无法播放，请重试。')}
      />
      <AudioBar
        playing={playing}
        duration={duration}
        position={position}
        onToggle={() => {
          const element = audio.current;
          if (!element) return;
          setError('');
          if (!element.paused) element.pause();
          else void element.play().catch(() => setError('音频暂时无法播放，请重试。'));
        }}
        onSeek={(next) => {
          if (audio.current) {
            audio.current.currentTime = next;
            setPosition(next);
          }
        }}
        onDownload={() => downloadAudio(src)}
      />
      {error && (
        <p className="inline-error" role="status">
          {error}
        </p>
      )}
    </>
  );
}

export function ChatAudio(props: { player: ChatAudioPlayer } | { src: string }) {
  return 'player' in props ? (
    <StreamingChatAudio player={props.player} />
  ) : (
    <FileChatAudio key={props.src} src={props.src} />
  );
}

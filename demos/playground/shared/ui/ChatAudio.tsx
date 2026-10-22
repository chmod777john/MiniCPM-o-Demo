import { useSyncExternalStore } from 'react';
import type { ChatAudioPlayer } from '../../turnbased/audio';
export function ChatAudio({ player }: { player: ChatAudioPlayer }) {
  useSyncExternalStore(player.subscribe, player.snapshot);
  return (
    <div className="chat-audio-controls">
      <button
        className="secondary-button"
        onClick={() => (player.playing ? player.pause() : player.play())}
      >
        {player.playing ? '暂停音频' : '播放音频'}
      </button>
      <input
        type="range"
        aria-label="音频播放位置"
        min={0}
        max={player.duration || 1}
        step={0.05}
        value={player.position}
        onChange={(e) => player.seek(Number(e.target.value))}
      />
      <span>
        {player.position.toFixed(1)} / {player.duration.toFixed(1)}s
      </span>
      <button
        className="text-button"
        onClick={() => {
          const url = URL.createObjectURL(player.wav());
          const a = document.createElement('a');
          a.href = url;
          a.download = 'assistant.wav';
          a.click();
          setTimeout(() => URL.revokeObjectURL(url), 1000);
        }}
      >
        下载音频
      </button>
    </div>
  );
}

import { numericSetting, settingOptions } from '../session/presets';
import { useEffect, useRef, useState } from 'react';
import { exportConversation } from '../../turnbased/export';
import { Download } from 'lucide-react';
import type { SessionSettings, SessionView, CaseDefinition } from '../session/types';
import type { SessionController } from '../session/controller';
import { Field, Section, Toggle } from './controls';
export function SessionArtifacts({
  settings: s,
  view,
  definition,
  controller,
  onSettings,
}: {
  settings: SessionSettings;
  view: SessionView;
  definition: CaseDefinition;
  controller: SessionController;
  onSettings: (p: Partial<SessionSettings>) => void;
}) {
  const [comment, setComment] = useState('');
  const [link, setLink] = useState('');
  const [saving, setSaving] = useState(false);
  useEffect(() => {
    setComment('');
    setLink('');
  }, [view.recordingSessionId]);
  const currentId = useRef(view.recordingSessionId);
  currentId.current = view.recordingSessionId;
  const busy = controller.busy;
  const number = (key: keyof SessionSettings, label: string, min: number, max: number) => {
    [min, max] = numericSetting(s, key, [min, max]);
    return (
      <Field label={label}>
        <input
          type="number"
          min={min}
          max={max}
          value={s[key] as number}
          disabled={busy}
          onChange={(e) =>
            onSettings({ [key]: Math.max(min, Math.min(max, Number(e.target.value))) })
          }
        />
      </Field>
    );
  };
  return (
    <Section title="录制与分享">
      {definition.protocol === 'chat' && (
        <button
          className="secondary-button full"
          disabled={!view.messages.length}
          onClick={() =>
            void exportConversation(s.systemContent, view.messages).catch((e) =>
              controller.setError(e),
            )
          }
        >
          导出对话 .msgpack
        </button>
      )}
      {definition.protocol === 'ordinary' && (
        <>
          <Toggle
            label="录制会话"
            checked={s.recordSession}
            disabled={busy}
            onChange={(recordSession) => onSettings({ recordSession })}
          />
          {definition.video && s.recordSession && (
            <>
              <Field label="录制格式">
                <select
                  value={s.recordingMime}
                  disabled={busy}
                  onChange={(e) => onSettings({ recordingMime: e.target.value })}
                >
                  <option value="auto">自动</option>
                  {[
                    'video/webm;codecs=vp9,opus',
                    'video/webm;codecs=vp8,opus',
                    'video/webm;codecs=h264,opus',
                    'video/webm',
                    'video/mp4',
                    'audio/webm;codecs=opus',
                    'audio/mp4',
                  ]
                    .filter(
                      (mime) =>
                        typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(mime),
                    )
                    .map((mime) => (
                      <option key={mime}>{mime}</option>
                    ))}
                </select>
              </Field>
              {s.recordingMime.includes('mp4') && (
                <p className="field-hint">
                  原页面提示：Chrome 的 MP4 录制可能没有音轨，建议选择 WebM。
                </p>
              )}
              <Field label="视频码率">
                <select
                  value={s.recordingBitrate}
                  disabled={busy}
                  onChange={(e) => onSettings({ recordingBitrate: Number(e.target.value) })}
                >
                  {settingOptions(s, 'recordingBitrate', [1000000, 2000000, 4000000, 8000000]).map(
                    (n) => (
                      <option value={n} key={n}>
                        {n / 1000000} Mbps
                      </option>
                    ),
                  )}
                </select>
              </Field>
              <Toggle
                label="录入字幕"
                checked={s.recordingSubtitles}
                disabled={busy}
                onChange={(recordingSubtitles) => onSettings({ recordingSubtitles })}
              />
              {s.recordingSubtitles && (
                <>
                  {number('subtitleHeight', '字幕区域高度 · %', 5, 80)}
                  {number('subtitleBottom', '底部字幕不透明度 · %', 0, 100)}
                  {number('subtitleTop', '顶部字幕不透明度 · %', 0, 100)}
                </>
              )}
            </>
          )}
        </>
      )}
      {view.recordingPending && <p className="field-hint">正在完成录制…</p>}
      {view.recordingUrl && (
        <a
          className="secondary-button full"
          download={`o5-${view.recordingSessionId || 'session'}.${controller.recordingExtension}`}
          href={view.recordingUrl}
        >
          <Download size={14} />
          下载会话录制
        </a>
      )}
      <Field label="分享备注">
        <textarea
          value={comment}
          maxLength={2000}
          rows={2}
          onChange={(e) => setComment(e.target.value)}
        />
      </Field>
      <button
        className="secondary-button full"
        disabled={
          (busy && definition.video) ||
          saving ||
          view.recordingPending ||
          !view.recordingSessionId ||
          (definition.video && !view.recordingUrl)
        }
        onClick={() => {
          setSaving(true);
          const sharingId = view.recordingSessionId;
          void controller
            .share(s, comment)
            .then((link) => {
              if (currentId.current === sharingId) setLink(link);
            })
            .catch((e) => controller.setError(e))
            .finally(() => setSaving(false));
        }}
      >
        {saving ? '正在保存…' : '上传并生成分享链接'}
      </button>
      {link && (
        <a className="share-link" href={link} target="_blank" rel="noreferrer">
          {link}
        </a>
      )}
    </Section>
  );
}

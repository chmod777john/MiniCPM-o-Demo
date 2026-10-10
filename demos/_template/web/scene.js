// 场景：只负责右上主面板。会话握手、麦克风、播放、system 编辑、think 账本都归 shell。
// 完整契约见 static/demo-gallery/core/scene-contract.js 顶部注释。

import { escapeHtml } from '../../../static/demo-gallery/core/ui-utils.js';

export function createScene({ stageEl, api }) {
  let count = 0;

  function render() {
    if (!count) {
      stageEl.classList.add('is-empty');
      stageEl.innerHTML = '<div class="empty-state">开麦说点什么，看这里有什么变化。</div>';
      return;
    }
    stageEl.classList.remove('is-empty');
    stageEl.innerHTML = `<p>已收到 ${escapeHtml(String(count))} 段 think</p>`;
  }

  return {
    panelTitle: '模板场景',
    panelHint: '把这里换成你的面板',
    render,                              // 必填：首帧由你自己画，骨架里一个字都没有
    reset() { count = 0; render(); },
    onThinkEnd() { count += 1; render(); },
    // 需要工具就实现 onToolCallDone(event)，并用 api.url('<你的接口>') 调自己的后端。
  };
}

// FC Realtime 页面的场景注册表。
//
// 页面骨架（会话握手、麦克风、音频播放、system 编辑、think/tool_call 账本、数据流）对所有场景共用；
// 场景只负责右上主面板和与之相关的事件。新增一个 Demo = 写一个场景模块 + 在这里登记 +
// 在 gateway 里给它一个页面路由（同一份 fc_board.html，通过 <body data-scene> 选场景）。
//
// 场景对象必须提供：
//   id / pageTitle / panelTitle / panelHint / liveStatus
//   presetOptions: [{id, label}]   快捷设置下拉框（末尾自动追加 Custom）
//   fallbackQuickSetting           Gateway 未指定或指定了本场景没有的 preset 时用它
//   buildQuickSettings(defaults) → Map<id, {segments, ttsPromptAudioPath, runtime}>
//   tools(defaults) → session.init 的 system.tools
//   render() / reset()
//   onThinkBegin(event) / onThinkDelta(event) / onThinkEnd(event) / onToolCallDone(event)

import { createBoardScene } from './board-scene.js';
import { createSpeakerScene } from './speaker-scene.js';

const SCENES = {
  board: createBoardScene,
  speakers: createSpeakerScene,
};

export function createScene(sceneId, context) {
  const factory = SCENES[sceneId] || SCENES.board;
  return factory(context);
}

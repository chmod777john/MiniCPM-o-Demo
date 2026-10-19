// Gallery 场景契约。
//
// createScene(context) 收到的 context：
//   demoId   string              当前 demo id
//   stageEl  HTMLElement         右上主面板根节点（#demoStage.stage）。shell 只保证它存在、
//                                是 flex 子项、min-height:0；**内容与布局全由场景负责**，
//                                shell 永远不写它的 innerHTML。
//   getClient () => client|null  当前 Realtime client（未连线时 null）
//   api      { base, url(p) }    本 demo 的 API 前缀；工具接口从这里拼，不要写死绝对路径
//   assets   { base, url(p) }    本 demo 的静态资源前缀
//   shell    { setStatus }       可选的回写通道
//
// 返回的 scene 对象：
//   必填 render()   首帧语义：shell 在初始化时调用一次，此时 stageEl 是空的，
//                   **场景必须自己画出空状态**（骨架 HTML 里不再有任何场景内容）。
//   可选 pageTitle / panelTitle / panelHint / liveStatus
//   可选 presetOptions [{id,label}]、fallbackQuickSetting、buildQuickSettings(defaults)
//   可选 tools(defaults, presetId)  presetId = 当前选中的 preset（选「Custom」时为 fallbackQuickSetting）；
//                                   工具顺序是训练口径的一部分时，按 preset 返回各自的工具
//   可选 reset()、onThinkBegin/onThinkDelta/onThinkEnd/onToolCallDone
//   可选 onUnitStarted(event)       每个被处理的 Unit 的 response.unit.started（含 tool_events /
//                                   standalone_events 回执），用于展示事件落在哪个 Unit
//   可选 onSpokenDelta(event, text) 每条 response.spoken.delta 及从中解出的文本（可能为空串）
//   可选 onSpokenEnd(event)         每条 response.spoken.end（reason: listen / turn_eos …）
//   可选 onToolCallBegin(event)、onToolCallDelta(event, text)  工具调用参数逐 token 生成的过程
//   可选 prepareSystem(segments)    建会话前的最后一步：收到当前 system segments 的副本，返回（可 async）
//                                   真正发出去的 segments。只做确定性的、与训练口径一致的变换
//                                   （如把 system 正文里的会话时间刷新成当前时间）；shell 会把结果
//                                   写回 system 编辑区，页面上看到的就是实际发送的
//
// ⚠️ 场景**不得**改 stageEl 的 class 来解除 shell 给的布局（这类补丁已废弃）。
//    布局写在 demo 自己的 CSS 里，选择器以 [data-demo="<id>"] 开头。

const NOOP = () => {};

/** 补默认值并校验必填项；缺必填项在页面加载期就大声报错。 */
export function normalizeScene(scene, { demoId, title }) {
  if (!scene || typeof scene !== 'object') {
    throw new Error(`demo ${demoId}: createScene 必须返回对象`);
  }
  if (typeof scene.render !== 'function') {
    throw new Error(`demo ${demoId}: scene.render() 是必填项`);
  }
  const presetOptions = Array.isArray(scene.presetOptions) ? scene.presetOptions : [];
  const bind = (fn, fallback) => (typeof fn === 'function' ? fn.bind(scene) : fallback);
  const normalized = {
    pageTitle: scene.pageTitle ?? title,
    panelTitle: scene.panelTitle ?? title,
    panelHint: scene.panelHint ?? '',
    liveStatus: scene.liveStatus ?? 'Listening via /v1/realtime',
    presetOptions,
    fallbackQuickSetting: scene.fallbackQuickSetting ?? presetOptions[0]?.id ?? null,
    buildQuickSettings: bind(scene.buildQuickSettings, () => new Map()),
    tools: bind(scene.tools, () => []),
    render: scene.render.bind(scene),
    reset: bind(scene.reset, scene.render.bind(scene)),
    onThinkBegin: bind(scene.onThinkBegin, NOOP),
    onThinkDelta: bind(scene.onThinkDelta, NOOP),
    onThinkEnd: bind(scene.onThinkEnd, NOOP),
    onToolCallDone: bind(scene.onToolCallDone, NOOP),
    onUnitStarted: bind(scene.onUnitStarted, NOOP),
    onSpokenDelta: bind(scene.onSpokenDelta, NOOP),
    onSpokenEnd: bind(scene.onSpokenEnd, NOOP),
    onToolCallBegin: bind(scene.onToolCallBegin, NOOP),
    onToolCallDelta: bind(scene.onToolCallDelta, NOOP),
    prepareSystem: bind(scene.prepareSystem, (segments) => segments),
  };
  for (const key of ['pageTitle', 'panelTitle']) {
    if (typeof normalized[key] !== 'string' || !normalized[key]) {
      throw new Error(`demo ${demoId}: scene.${key} 必须是非空字符串`);
    }
  }
  return normalized;
}

/** 测试用：不构造 DOM，只校验一个 scene 对象是否满足契约。 */
export function validateScene(scene, demoId) {
  normalizeScene(scene, { demoId, title: demoId });
  return true;
}

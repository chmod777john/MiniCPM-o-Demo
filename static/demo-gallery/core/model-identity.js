// 「当前加载的模型」展示行定义。
//
// 数据来自所选 API 端点的 /v1/info（models[].model）与 session.created.model，字段由 API 侧
// core/fc_duplex/model_info.py 定义（有测试对拍：那边新增路径类字段而这里漏显示会红）。
// 路径一律完整展示、不截断 —— 截断会让「到底跑的哪个 ckpt」要靠猜。

/** @type {ReadonlyArray<[string, string]>} [后端字段名, 展示标签] */
export const MODEL_IDENTITY_ROWS = [
  ['pt_path', 'Checkpoint (PT)'],
  ['backbone_dir', 'TP2 backbone'],
  ['base_model_path', 'Base model'],
  ['checkpoint_sha256', 'SHA256'],
  ['deployment_mode', 'Deployment'],
  ['sdk_version', 'SDK (推理侧)'],
  ['tts_pad_protocol', 'tts_pad (训练侧)'],
];

/** 悬停说明：只给容易被误读的行。 */
export const MODEL_IDENTITY_HINTS = {
  sdk_version: '推理服务安装的 minicpm-o5-sdk 版本（启动硬门禁），与该 ckpt 训练时用的 SDK 无关',
  tts_pad_protocol: '按该 ckpt 训练时的 SDK 定：≤0.0.6a3 → legacy，≥0.0.6a4 → revised',
  pt_path: 'model.pt 常是 symlink，这里是解析后的真实路径（可能在别人目录下）',
};

import { ScanEye, PackageSearch, Trees } from 'lucide-react';
import type { CaseDefinition } from '../shared/session/types';

export const definition: CaseDefinition = {
  id: 'video-duplex',
  label: '视频双工',
  title: '一起看看眼前的世界',
  description: '让画面加入对话，分享你正在看见的事物。',
  protocol: 'ordinary',
  group: 'conversation',
  accent: '#4d9bac',
  prompts: ['看看桌上的物品', '聊聊眼前的景色', '帮我理解一段画面'],
  promptIcons: [PackageSearch, Trees, ScanEye],
  icon: ScanEye,
  video: true,
};

import { PanelsTopLeft, BookOpen, Map, ImagePlus } from 'lucide-react';
import type { CaseDefinition } from '../shared/session/types';

export const definition: CaseDefinition = {
  id: 'board',
  label: '灵感画板',
  title: '让对话跃然眼前',
  description: '描述一个场景，画板会随着对话逐渐展开。',
  protocol: 'fc',
  group: 'agent',
  accent: '#bb8b62',
  prompts: ['讲一个森林里的故事', '描绘一次理想的旅行', '把我提到的物品放到画板上'],
  promptIcons: [BookOpen, Map, ImagePlus],
  icon: PanelsTopLeft,
  loadScene: () => import('./scene').then((module) => module.factory),
};

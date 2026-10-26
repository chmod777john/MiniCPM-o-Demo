import { RadioTower, AlarmClock, MessageSquarePlus, Activity } from 'lucide-react';
import type { CaseDefinition } from '../shared/session/types';

export const definition: CaseDefinition = {
  id: 'standalone',
  label: '外部事件',
  title: '让新的信息加入对话',
  description: '在持续对话中注入消息，观察助手如何理解和回应。',
  protocol: 'fc',
  group: 'agent',
  accent: '#a68b69',
  prompts: ['提醒我五分钟后出发', '告诉助手一条新的消息', '模拟任务进度更新'],
  promptIcons: [AlarmClock, MessageSquarePlus, Activity],
  icon: RadioTower,
  loadScene: () => import('./scene').then((module) => module.factory),
};

import { UserRoundCheck, UserRound, Users, AudioLines } from 'lucide-react';
import type { CaseDefinition } from '../shared/session/types';

export const definition: CaseDefinition = {
  id: 'rejection',
  label: '说话人识别',
  title: '专注与你的对话',
  description: '在多人说话的环境里，观察助手如何辨认交流对象。',
  protocol: 'fc',
  group: 'agent',
  accent: '#759c85',
  prompts: ['先由你开口介绍自己', '让另一位朋友加入对话', '观察助手回应了谁'],
  promptIcons: [UserRound, Users, AudioLines],
  icon: UserRoundCheck,
  loadScene: () => import('./scene').then((module) => module.factory),
};

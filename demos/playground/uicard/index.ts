import { LayoutDashboard, CloudSun, MapPin, Route } from 'lucide-react';
import type { CaseDefinition } from '../shared/session/types';

export const definition: CaseDefinition = {
  id: 'uicard',
  label: '交互卡片',
  title: '说出来，让信息呈现',
  description: '聊聊天气、找个去处，让有用的信息变成卡片。',
  protocol: 'fc',
  group: 'agent',
  accent: '#5a96ae',
  prompts: ['北京今天的天气怎么样？', '找一找附近适合散步的地方', '帮我规划一个轻松的周末'],
  promptIcons: [CloudSun, MapPin, Route],
  icon: LayoutDashboard,
  loadScene: () => import('./scene').then((module) => module.factory),
};

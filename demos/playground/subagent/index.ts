import { Workflow, Search, ListChecks, Calculator } from 'lucide-react';
import type { CaseDefinition } from '../shared/session/types';

export const definition: CaseDefinition = {
  id: 'subagent',
  label: 'SubAgent',
  title: '从一个想法，到一次行动',
  description: '用自然对话交代任务，跟随助手查看执行进展。',
  protocol: 'fc',
  group: 'agent',
  accent: '#987ac2',
  prompts: ['帮我调研一个感兴趣的话题', '整理一份旅行准备清单', '帮我计算并解释结果'],
  promptIcons: [Search, ListChecks, Calculator],
  icon: Workflow,
  loadScene: () => import('./scene').then((module) => module.factory),
};

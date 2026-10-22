import { MessagesSquare } from 'lucide-react';
import type { CaseDefinition } from '../shared/session/types';

export const definition: CaseDefinition = {
  id: 'turnbased',
  label: '单工对话',
  title: '自由对话',
  description: '把问题交给我，文字、图像、声音都可以。',
  protocol: 'chat',
  group: 'conversation',
  accent: '#6477e8',
  prompts: ['帮我写一份工作计划', '一起读懂一张图片', '帮我理解一段代码'],
  icon: MessagesSquare,
};

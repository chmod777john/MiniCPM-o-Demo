import { AudioLines, MessagesSquare, Languages, Lightbulb } from 'lucide-react';
import type { CaseDefinition } from '../shared/session/types';

export const definition: CaseDefinition = {
  id: 'audio-duplex',
  label: '语音双工',
  title: '自然地聊一聊',
  description: '打开麦克风，随时开口，也随时打断。',
  protocol: 'ordinary',
  group: 'conversation',
  accent: '#7484ec',
  prompts: ['聊聊今天发生的趣事', '陪我练一段英语对话', '一起头脑风暴'],
  promptIcons: [MessagesSquare, Languages, Lightbulb],
  icon: AudioLines,
};

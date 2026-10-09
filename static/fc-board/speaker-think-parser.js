// Rejection（拒识）think 的说话人解析。
//
// 句式来源：rejection 0929 训练数据（zh/en 各两档长度）全部 think 去重后的封闭模板集合，
// 共 zh 11 种 / en 10 种。解析只做「把模型自己写下的身份判断读出来」，不做任何推断：
// 识别不了的 think 返回 null，由页面原样展示，不改变说话人状态。
//
// 返回值（ParsedSpeakerThink）：
//   kind     'register_user'      登记用户（第一个开口的人 / 应其要求登记）
//            'user'               判定为已登记用户
//            'new_speaker'        判定为新的干扰说话人，编号 speakerNo
//            'known_speaker'      判定为已编号的干扰说话人
//            'speaker_claims_user' 干扰说话人要求成为唯一对象，被拒
//   speakerKey  'user' 或 'speaker:<N>'
//   speakerNo   干扰说话人编号（user 为 null）
//   reason   register_user 时：'first_speaker' | 'requested'
//   action   'respond' | 'interrupt_listen' | 'backchannel_continue' | 'no_respond' | 'no_stop' | null
//            null = 句子还没写到动作部分（流式解析时常见）

const PATTERNS = {
  zh: {
    register: /登记为用户/,
    registerRequested: /要求我只和其对话/,
    user: /^\s*这是用户的声音/,
    newSpeaker: /新的说话人，?\s*我记为说话人\s*(\d+)\s*[，。,.;；]/,
    claim: /是说话人\s*(\d+)\s*[。.]\s*说话人\s*\d+\s*要求成为我唯一回应的对象/,
    knownSpeaker: /^\s*这是说话人\s*(\d+)\s*的声音/,
    interrupt: /打断/,
    backchannel: /附和/,
    respond: /正常交流/,
    noRespond: /我不回应/,
    noStop: /我不停下/,
  },
  en: {
    register: /register this speaker as the user/i,
    registerRequested: /asked me to talk only with them/i,
    user: /^\s*This is the user['’]s voice/i,
    newSpeaker: /new speaker,\s*whom I call Speaker\s*(\d+)\s*[.,;]/i,
    claim: /it is Speaker\s*(\d+)\s*\.\s*Speaker\s*\d+\s*asks to be the only one/i,
    knownSpeaker: /^\s*This is Speaker\s*(\d+)['’]s voice/i,
    interrupt: /interrupting/i,
    backchannel: /backchanneling/i,
    respond: /respond normally/i,
    noRespond: /I don['’]t respond/i,
    noStop: /I don['’]t stop/i,
  },
};

export function detectThinkLanguage(text) {
  return /[一-鿿]/.test(text || '') ? 'zh' : 'en';
}

export function parseSpeakerThink(text) {
  const source = String(text || '');
  if (!source.trim()) return null;
  const p = PATTERNS[detectThinkLanguage(source)];

  // 登记句里自带规则说明（含「说话人 1」字样），必须最先判，否则会被误读成干扰说话人。
  if (p.register.test(source)) {
    return {
      kind: 'register_user',
      speakerKey: 'user',
      speakerNo: null,
      reason: p.registerRequested.test(source) ? 'requested' : 'first_speaker',
      action: 'respond',
    };
  }
  if (p.user.test(source)) {
    let action = null;
    if (p.interrupt.test(source)) action = 'interrupt_listen';
    else if (p.backchannel.test(source)) action = 'backchannel_continue';
    else if (p.respond.test(source)) action = 'respond';
    return { kind: 'user', speakerKey: 'user', speakerNo: null, reason: null, action };
  }
  for (const [kind, re] of [
    ['speaker_claims_user', p.claim],
    ['new_speaker', p.newSpeaker],
    ['known_speaker', p.knownSpeaker],
  ]) {
    const match = source.match(re);
    if (!match) continue;
    const speakerNo = Number(match[1]);
    return {
      kind,
      speakerKey: `speaker:${speakerNo}`,
      speakerNo,
      reason: null,
      action: otherSpeakerAction(source, p),
    };
  }
  return null;
}

function otherSpeakerAction(source, p) {
  if (p.noStop.test(source)) return 'no_stop';
  if (p.noRespond.test(source)) return 'no_respond';
  return null;
}

// 说话人登记表：只由解析结果驱动。新出现的 key 创建，已存在的只刷新「最近识别」。
export class SpeakerRegistry {
  constructor() {
    this.reset();
  }

  reset() {
    this.speakers = new Map();
    this.activeKey = null;
    this.events = [];
  }

  // 同一个 think 块流式期间会反复解析；blockKey 相同的只记一次，后续只补全 action。
  apply(parsed, { unitIndex = null, blockKey = null, text = '' } = {}) {
    if (!parsed) return null;
    const existingEvent = blockKey !== null
      ? this.events.find((item) => item.blockKey === blockKey)
      : null;
    if (existingEvent) {
      if (existingEvent.parsed.speakerKey !== parsed.speakerKey) return existingEvent;
      existingEvent.parsed = parsed;
      existingEvent.text = text;
      const speaker = this.speakers.get(parsed.speakerKey);
      if (speaker) speaker.lastAction = parsed.action;
      return existingEvent;
    }

    let speaker = this.speakers.get(parsed.speakerKey);
    const created = !speaker;
    if (created) {
      speaker = {
        key: parsed.speakerKey,
        speakerNo: parsed.speakerNo,
        isUser: parsed.speakerKey === 'user',
        registeredUnit: unitIndex,
        // 模型没先登记就直接说「已知说话人 N / 用户的声音」：照样建卡，但标出来。
        // 「要求成为唯一对象」那句可以是该说话人第一次出现（训练数据里 382 例），它本身就在编号，算正常登记。
        registeredByModel: ['register_user', 'new_speaker', 'speaker_claims_user'].includes(parsed.kind),
        hits: 0,
        lastUnit: null,
        lastAction: null,
        claimedUser: false,
      };
      this.speakers.set(parsed.speakerKey, speaker);
    }
    speaker.hits += 1;
    speaker.lastUnit = unitIndex;
    speaker.lastAction = parsed.action;
    if (parsed.kind === 'speaker_claims_user') speaker.claimedUser = true;
    this.activeKey = parsed.speakerKey;
    const event = { blockKey, unitIndex, parsed, created, text };
    this.events.push(event);
    return event;
  }

  orderedSpeakers() {
    return [...this.speakers.values()].sort((a, b) => {
      if (a.isUser !== b.isUser) return a.isUser ? -1 : 1;
      return (a.speakerNo ?? 0) - (b.speakerNo ?? 0);
    });
  }
}

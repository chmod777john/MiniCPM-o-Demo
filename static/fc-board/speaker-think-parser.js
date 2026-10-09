// Rejection（拒识）think 的说话人解析。
//
// 解析只做「把模型自己写下的身份判断读出来」，不做任何推断：识别不了的 think 返回 null，
// 由页面原样展示，不改变说话人状态。
//
// 规则抓的是身份结构，不是整句模板。训练数据（rejection 0929，zh 11 / en 10 种模板）是逐字
// 固定的，但真实模型会换说法 —— 2026-10-09 在 0930 mergeAll iter_500 上实测到「这是用户的语音」
// （训练是「声音」）、「这是用户第一次对我说话…我将这段内容登记为用户」。所以只认三类骨架：
//   登记为用户 / 这是用户… / 不是用户…说话人 N（再看是新编号、要求唯一对象，还是已知）
//
// 返回值（ParsedSpeakerThink）：
//   kind     'register_user'       登记用户（第一个开口的人 / 应其要求登记）
//            'user'                判定为已登记用户
//            'new_speaker'         判定为新的干扰说话人，编号 speakerNo
//            'known_speaker'       判定为已编号的干扰说话人
//            'speaker_claims_user' 干扰说话人要求成为唯一对象，被拒
//   speakerKey  'user' 或 'speaker:<N>'
//   speakerNo   干扰说话人编号（user 为 null）
//   reason   register_user 时：'first_speaker' | 'requested'
//   action   'respond' | 'interrupt_listen' | 'backchannel_continue' | 'no_respond' | 'no_stop' | null
//            null = 句子还没写到动作部分（流式解析时常见），或模型没写明动作

// 编号后面必须已经出现非数字字符才算写完（流式时「说话人 1」之后可能还有「2」）；
// 整块结束时调用方传 final=true，允许编号落在句末。
const NUM = String.raw`(\d+)(?=\D)`;

const PATTERNS = {
  zh: {
    register: /登记[为成]用户|记为用户/,
    registerRequested: /要求我只和(其|他|她)|要求我只(回复|回应)(其|他|她)/,
    user: /^\s*这(?:仍|还|也)?是用户/,
    notUser: /不是用户/,
    speakerNo: new RegExp(String.raw`说话人\s*` + NUM),
    knownLead: new RegExp(String.raw`^\s*这(?:仍|还|也)?是说话人\s*` + NUM),
    newSpeaker: /新的?说话人|记为说话人|编号为?说话人/,
    claim: /要求成为|唯一(回应|回复)的?对象|只和(他|她|其)对话/,
    interrupt: /打断/,
    backchannel: /附和|应和/,
    respond: /正常(交流|回应|回复|对话)/,
    noStop: /不停下|继续说/,
    noRespond: /不(回应|回复|理会|搭理|作答)/,
  },
  en: {
    register: /register(?:ed|ing)?\s+(?:this|the)\s+(?:speaker|person)\s+as\s+(?:the\s+)?user/i,
    registerRequested: /asked me to (?:talk|speak|respond) only/i,
    user: /^\s*This is (?:still |also )?the user/i,
    notUser: /not the user/i,
    speakerNo: new RegExp(String.raw`Speaker\s*` + NUM, 'i'),
    knownLead: new RegExp(String.raw`^\s*This is (?:still |also )?Speaker\s*` + NUM, 'i'),
    newSpeaker: /new speaker|I(?:'ll| will)? call (?:them |him |her )?Speaker|label(?:led)? (?:them |him |her )?(?:as )?Speaker/i,
    claim: /only one I respond|asks? to be the only|wants? to be the only/i,
    interrupt: /interrupt/i,
    backchannel: /backchannel/i,
    respond: /respond normally|reply normally/i,
    noStop: /don['’]?t stop|do not stop|keep (?:talking|speaking)/i,
    noRespond: /don['’]?t respond|do not respond|not respond|ignore/i,
  },
};

export function detectThinkLanguage(text) {
  return /[一-鿿]/.test(text || '') ? 'zh' : 'en';
}

export function parseSpeakerThink(text, { final = false } = {}) {
  const raw = String(text || '');
  if (!raw.trim()) return null;
  // 整块结束：补一个句末字符，让落在句末的编号也能被 NUM 的 (?=\D) 接受。
  const source = final ? `${raw}\n` : raw;
  const p = PATTERNS[detectThinkLanguage(source)];

  // 登记句里常带规则说明（含「说话人 1」字样），必须最先判，否则会被误读成干扰说话人。
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
    else if (p.respond.test(source) || final) action = 'respond';
    return { kind: 'user', speakerKey: 'user', speakerNo: null, reason: null, action };
  }

  let kind = null;
  let match = null;
  if (p.notUser.test(source)) {
    match = source.match(p.speakerNo);
    if (match) kind = p.claim.test(source) ? 'speaker_claims_user' : (p.newSpeaker.test(source) ? 'new_speaker' : 'known_speaker');
  } else {
    match = source.match(p.knownLead);
    if (match) kind = p.claim.test(source) ? 'speaker_claims_user' : 'known_speaker';
  }
  if (!kind) return null;
  const speakerNo = Number(match[1]);
  return {
    kind,
    speakerKey: `speaker:${speakerNo}`,
    speakerNo,
    reason: null,
    action: otherSpeakerAction(source, p),
  };
}

function otherSpeakerAction(source, p) {
  if (p.noStop.test(source)) return 'no_stop';
  if (p.noRespond.test(source)) return 'no_respond';
  return null;
}

// 模型在这几类 think 里给说话人建立身份（「要求成为唯一对象」那句可以是该说话人第一次出现，
// 训练数据里 382 例，它本身就在编号）。
const MODEL_REGISTERING_KINDS = ['register_user', 'new_speaker', 'speaker_claims_user'];

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
      if (speaker) {
        speaker.lastAction = parsed.action;
        // 流式时「这是用户第一次…」先被读成 user，写到「登记为用户」才知道是登记。
        if (MODEL_REGISTERING_KINDS.includes(parsed.kind)) speaker.registeredByModel = true;
        if (parsed.kind === 'speaker_claims_user') speaker.claimedUser = true;
      }
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
        registeredByModel: MODEL_REGISTERING_KINDS.includes(parsed.kind),
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

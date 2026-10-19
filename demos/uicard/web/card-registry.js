// 会话内的卡片注册表与编号。
//
// 天气（w）与地图（m）卡由工具拼好，模型只拿到编号；编号在整个会话里递增不重复（第二次查地图是 m2），
// 之前的编号一直有效（训练数据的口径）。服务端无状态：页面告诉它这次从几号开始（card_index_base），
// 拿到结果后把卡片登记进来。工具执行串行进行，避免两个并发调用拿到同一个起始编号。
// 移植自 cjp/o5-fc-dev 分支 static/agent-tools-v2/agent-tools-v2-app.js。

const PREFIX_BY_TOOL = { amap_weather: 'w', map_search: 'm' };

export function cardPrefixFor(toolName) {
  return PREFIX_BY_TOOL[toolName] || null;
}

export class CardRegistry {
  constructor() {
    this.reset();
  }

  reset() {
    this.cards = new Map();
    this.counters = { w: 1, m: 1 };
    this.queue = Promise.resolve();
  }

  /** 本次调用的卡片编号起点（只对会产出卡片的工具有意义）。 */
  indexBaseFor(toolName) {
    const prefix = cardPrefixFor(toolName);
    return prefix ? this.counters[prefix] : null;
  }

  /** 登记工具返回的卡片，并推进对应前缀的计数器。 */
  register(toolName, cards) {
    const entries = Array.isArray(cards) ? cards.filter((entry) => entry?.card_id) : [];
    for (const entry of entries) this.cards.set(entry.card_id, entry.card);
    const prefix = cardPrefixFor(toolName);
    if (prefix) this.counters[prefix] += entries.length;
    return entries.map((entry) => entry.card_id);
  }

  get(cardId) {
    return this.cards.get(cardId) || null;
  }

  /** 串行执行：读计数器 → await 工具 → 推进计数器，中间不能被别的调用插队。 */
  runExclusive(task) {
    const result = this.queue.then(task, task);
    this.queue = result.then(() => {}, () => {});
    return result;
  }
}

/** show_card 引用了注册表里没有的编号：不画、并如实告诉模型（多半是给自写卡片传了编号）。 */
export function unknownCardIdMessage(cardId) {
  return `unknown card_id: ${cardId}。只有 amap_weather 和 map_search 的结果有编号；`
    + 'hint/emoji/animation/score/stat/list/choice/custom 请直接传 card 对象。';
}

/**
 * ns-segment-view.js
 * =============================================================================
 * FC Board 的 non-spoken segment「3 轨 StR 审计视图」。
 *
 * 作用域：一个 non-spoken segment = 一次 think（begin→end）或一次 tool_call
 * （begin→done），可跨多个 Unit。每个 segment 渲染成一张卡片，卡片内是一条
 * 连续的 token 栅格，每个模型 token（step）= 一个 3 行「微列」格子：
 *
 *   上轨（Unit 轨道）  = 标出每个 Unit 的起止边界；起点写 `Un · budget N · listen/speak`，
 *                       终点写 `used N`（触顶变红）。
 *   中轨（token 格）   = 一个 step 一个带边框背景的格子；真实 BPE：一个 token 可能多字
 *                       （规则）、latin 子词（display）、个别字需 pending+text 才成字。
 *   下轨（序号轨）      = 该 token 在本 Unit 内的序号（0 起、跨 Unit 重置）；
 *                       eos/no_action 用 ◆，budget_reached 用 `budget`。
 *
 * 特殊 cell：
 *   pending        ◐  未成字（字节碎片，占 1 budget）
 *   wrapper        ⟨think⟩ / ⟨/think⟩（占 1）
 *   eos/no_action  ■ + 下轨 ◆（模型终止，占 1）
 *   budget_reached ⏭ 红色虚线切断（framework，占 0，下个 Unit 待续）
 *
 * 数据来源（Semantic Realtime API v2，均可推导，不碰 token ID / source_steps）：
 *   response.think.begin/delta/end
 *   response.tool_call.begin/delta/done
 *   response.non_spoken.end(reason=eos|no_action|budget_reached)
 *   response.spoken.end(reason)  → 定该 Unit 是 listen 还是 speak → 选 budget 档
 *   session config non_spoken_budget_while_listening / _while_speaking → budget 上限
 *
 * 这份文件只做渲染 + 由事件驱动的状态机，不发网络请求、不解析 tool 业务。
 * =============================================================================
 */

const LATIN_RE = /^[A-Za-z_]/;

/**
 * 单个 non-spoken segment 的运行时渲染状态。
 * @typedef {Object} SegmentState
 * @property {'think'|'tool_call'} kind        segment 类型
 * @property {HTMLElement} card                卡片根节点
 * @property {HTMLElement} segEl               token 栅格容器
 * @property {HTMLElement} headEl              头部（kind / unit 范围 / tokens / 校验）
 * @property {?string} toolCallId              tool_call 的 id（think 为 null）
 * @property {?number} currentUnit             当前正在追加的 Unit
 * @property {boolean} ended                   语义 end/done 是否已到（等待终止 cell）
 * @property {?string} fullText                end/done 携带的完整文本，用于校验
 * @property {string} textConcat               累计所有 text step 文本，用于 join 校验
 * @property {number} tokenCount               计数 cell 数（占 budget 的 step）
 * @property {?number} minUnit                 segment 覆盖的最小 Unit
 * @property {?number} maxUnit                 segment 覆盖的最大 Unit
 */

/**
 * @typedef {Object} UnitRenderMeta
 * @property {?HTMLElement} startLabelEl  该 Unit 起点标签节点（延迟更新 lane/limit）
 * @property {?HTMLElement} usedEl        该 Unit 终点 used 标签节点
 * @property {?number} usedValue          该 Unit 已用计数（used）
 * @property {HTMLElement[]} cells        属于该 Unit 的所有 cell（用于 lane 上色）
 */

export class NsSegmentView {
  /**
   * @param {HTMLElement} container non-spoken 卡片容器（原 #nonSpokenStream）
   */
  constructor(container) {
    /** @type {HTMLElement} */
    this.container = container;
    /** @type {?number} listen 档 non-spoken budget 上限 */
    this.listeningBudget = null;
    /** @type {?number} speak 档 non-spoken budget 上限 */
    this.speakingBudget = null;
    this.reset();
  }

  /** 清空视图与全部运行时状态（新会话 / clearViews 时调用）。 */
  reset() {
    this.container.innerHTML = '<div class="placeholder">还没有 think / tool_call 块</div>';
    /** @type {?SegmentState} 当前打开的 segment（non-spoken lane 串行，至多一个） */
    this.active = null;
    /** @type {Map<string,'listen'|'speak'>} unit_index → lane */
    this.unitLane = new Map();
    /** @type {Map<string,number>} unit_index → 下一个 step 的 per-Unit 序号 */
    this.unitStepCount = new Map();
    /** @type {Map<string,UnitRenderMeta>} unit_index → 渲染 meta */
    this.unitMeta = new Map();
  }

  /**
   * 设置两档 non-spoken budget 上限（来自 session.init config）。
   * @param {number|string} listening listen 档上限
   * @param {number|string} speaking  speak 档上限
   */
  setBudgets(listening, speaking) {
    const l = Number.parseInt(String(listening ?? ''), 10);
    const s = Number.parseInt(String(speaking ?? ''), 10);
    this.listeningBudget = Number.isFinite(l) ? l : null;
    this.speakingBudget = Number.isFinite(s) ? s : null;
  }

  /**
   * 记录某 Unit 的 lane（来自 response.spoken.end.reason）。
   * listen → listening budget；其余（turn_eos/slot_eos/slot_end）→ speaking budget。
   * @param {number} unitIndex Unit 序号
   * @param {'listen'|'speak'} lane 该 Unit 的 spoken 决策
   */
  setUnitLane(unitIndex, lane) {
    if (unitIndex === null || unitIndex === undefined) return;
    const key = String(unitIndex);
    this.unitLane.set(key, lane);
    this._applyUnitLane(key);
  }

  /**
   * 取某 Unit 的 budget 上限（lane 未知时返回 null）。
   * @param {number|string} unitIndex Unit 序号
   * @returns {?number} 该 Unit 的 budget 上限
   */
  _limitForUnit(unitIndex) {
    const lane = this.unitLane.get(String(unitIndex));
    if (lane === 'listen') return this.listeningBudget;
    if (lane === 'speak') return this.speakingBudget;
    return null;
  }

  /**
   * 开一个 segment（think.begin / tool_call.begin）。追加 wrapper 起始 cell。
   * @param {'think'|'tool_call'} kind segment 类型
   * @param {number} unitIndex 起始 Unit
   * @param {{toolCallId?:string}} [opts] tool_call 需传 toolCallId
   */
  beginSegment(kind, unitIndex, opts = {}) {
    const placeholder = this.container.querySelector('.placeholder');
    if (placeholder) placeholder.remove();

    const card = document.createElement('article');
    card.className = `nsv-card kind-${kind}`;
    const headEl = document.createElement('div');
    headEl.className = 'nsv-head';
    const segEl = document.createElement('div');
    segEl.className = 'nsv-seg';
    // full 阅读区（end/done 到达后填入 full_text，与旧版一致）
    const fullEl = document.createElement('section');
    fullEl.className = 'nsv-full';
    fullEl.style.display = 'none';
    fullEl.innerHTML = '<div class="nsv-full-tag">full</div><div class="nsv-full-body"></div>';
    card.appendChild(headEl);
    card.appendChild(segEl);
    card.appendChild(fullEl);
    this.container.appendChild(card);

    /** @type {SegmentState} */
    const seg = {
      kind,
      card,
      segEl,
      headEl,
      toolCallId: opts.toolCallId ?? null,
      fullEl,
      currentUnit: null,
      ended: false,
      fullText: null,
      textConcat: '',
      tokenCount: 0,
      minUnit: null,
      maxUnit: null,
    };
    this.active = seg;

    const openGlyph = kind === 'tool_call' ? '⟨tool⟩' : '⟨think⟩';
    this._appendCell(seg, unitIndex, { kind: 'wrapper', glyph: openGlyph, counts: true });
    this._renderHead(seg);
    this._scrollToBottom();
  }

  /**
   * 追加一批 text/pending step（think.delta / tool_call.delta）。
   * @param {number} unitIndex 这批 step 所属 Unit
   * @param {Array<{kind:string,text?:string}>} steps discriminated-union step 列表
   */
  appendSteps(unitIndex, steps) {
    const seg = this.active;
    if (!seg) return;
    for (const step of Array.isArray(steps) ? steps : []) {
      if (step?.kind === 'text') {
        const text = step.text || '';
        seg.textConcat += text;
        this._appendCell(seg, unitIndex, { kind: 'text', glyph: text, counts: true });
      } else if (step?.kind === 'pending') {
        this._appendCell(seg, unitIndex, { kind: 'pending', glyph: '', counts: true });
      }
    }
    this._renderHead(seg);
    this._scrollToBottom();
  }

  /**
   * 语义 end/done（think.end / tool_call.done）。追加 wrapper 收尾 cell，
   * 但不立即关闭 segment —— 等 response.non_spoken.end 的终止 cell 到达后再 finalize。
   * @param {'think'|'tool_call'} kind segment 类型
   * @param {number} unitIndex 收尾所在 Unit
   * @param {{fullText?:string}} [opts]
   */
  endSegment(kind, unitIndex, opts = {}) {
    const seg = this.active;
    if (!seg) return;
    const closeGlyph = kind === 'tool_call' ? '⟨/tool⟩' : '⟨/think⟩';
    this._appendCell(seg, unitIndex, { kind: 'wrapper', glyph: closeGlyph, counts: true });
    if (opts.fullText !== undefined) seg.fullText = opts.fullText;
    seg.ended = true;
    // 填充 full 阅读区（有 full_text 才显示）
    if (seg.fullEl) {
      const body = seg.fullEl.querySelector('.nsv-full-body');
      if (seg.fullText) {
        if (body) body.textContent = seg.fullText;
        seg.fullEl.style.display = '';
      } else {
        seg.fullEl.style.display = 'none';
      }
    }
    this._renderHead(seg);
    this._scrollToBottom();
  }

  /**
   * 每个 Unit 的 non-spoken 终止（response.non_spoken.end）。
   *   budget_reached → ⏭ 切断 cell（framework，占 0），本 Unit 结束、segment 续到下个 Unit；
   *   eos/no_action  → ■/◆ 终止 cell（占 1），本 Unit 结束、segment 整体收尾。
   * 若当前无 active segment（如空 Unit 的 no_action），则不渲染。
   * @param {number} unitIndex Unit 序号
   * @param {'eos'|'no_action'|'budget_reached'} reason 终止原因
   */
  markNonSpokenEnd(unitIndex, reason) {
    const seg = this.active;
    if (!seg) return;
    if (reason === 'budget_reached') {
      this._appendCell(seg, unitIndex, { kind: 'budget', glyph: '', counts: false, unitEnd: true });
    } else {
      this._appendCell(seg, unitIndex, { kind: 'eos', glyph: '■', counts: true, unitEnd: true });
      this._finalizeSegment(seg);
      this.active = null;
    }
    this._renderHead(seg);
    this._scrollToBottom();
  }

  // ---------------------------------------------------------------------------
  // 内部渲染
  // ---------------------------------------------------------------------------

  /**
   * 追加一个 cell（3 轨微列）到 segment，并处理 Unit 起止边界与 per-Unit 序号。
   * @param {SegmentState} seg
   * @param {number} unitIndex
   * @param {{kind:string,glyph:string,counts:boolean,unitEnd?:boolean}} spec
   */
  _appendCell(seg, unitIndex, spec) {
    const key = String(unitIndex);
    const isNewUnit = seg.currentUnit === null || String(seg.currentUnit) !== key;
    if (isNewUnit) seg.currentUnit = unitIndex;

    // per-Unit 序号（只有计数 cell 才拿号）
    let idx = null;
    if (spec.counts) {
      idx = this.unitStepCount.get(key) ?? 0;
      this.unitStepCount.set(key, idx + 1);
    }

    // segment 覆盖 Unit 范围 + token 计数
    const un = Number(unitIndex);
    if (Number.isFinite(un)) {
      seg.minUnit = seg.minUnit === null ? un : Math.min(seg.minUnit, un);
      seg.maxUnit = seg.maxUnit === null ? un : Math.max(seg.maxUnit, un);
    }
    if (spec.counts) seg.tokenCount += 1;

    const lane = this.unitLane.get(key) || null;

    const cell = document.createElement('span');
    cell.className = `nsv-cell ${spec.kind}`
      + (spec.kind === 'text' && LATIN_RE.test(spec.glyph) ? ' latin' : '')
      + (lane ? ` lane-${lane}` : '')
      + (isNewUnit ? ' u-start' : '')
      + (spec.unitEnd ? ' u-end' : '');
    cell.dataset.unit = key;
    cell.title = `U${key} · ${spec.kind}` + (idx !== null ? ` · #${idx}` : ' · framework · 占0');

    // 上轨
    const u = document.createElement('span');
    u.className = 'nsv-u';
    const line = document.createElement('span');
    line.className = 'nsv-line';
    u.appendChild(line);

    const meta = this._unitMeta(key);
    if (isNewUnit) {
      const capL = document.createElement('span');
      capL.className = 'nsv-cap-l';
      u.appendChild(capL);
      const label = document.createElement('span');
      label.className = 'nsv-u-label';
      u.appendChild(label);
      meta.startLabelEl = label;
    }
    if (spec.unitEnd) {
      const capR = document.createElement('span');
      capR.className = 'nsv-cap-r';
      u.appendChild(capR);
      const usedEl = document.createElement('span');
      usedEl.className = 'nsv-u-used';
      u.appendChild(usedEl);
      meta.usedEl = usedEl;
      meta.usedValue = this.unitStepCount.get(key) ?? 0;
    }
    cell.appendChild(u);

    // 中轨
    const g = document.createElement('span');
    g.className = 'nsv-g';
    const box = document.createElement('span');
    box.className = 'nsv-box';
    if (spec.kind === 'text' && spec.glyph.includes(' ')) {
      // 空格属于该 text token（BPE 前导空格），可视化成淡 ␣，不归给前面的 pending
      box.innerHTML = [...spec.glyph]
        .map((ch) => (ch === ' ' ? '<span class="nsv-sp">␣</span>' : escapeHtml(ch)))
        .join('');
    } else {
      box.textContent = spec.glyph;
    }
    if (spec.kind === 'budget') {
      const mark = document.createElement('span');
      mark.className = 'nsv-cut';
      mark.textContent = '⏭';
      box.appendChild(mark);
    }
    g.appendChild(box);
    cell.appendChild(g);

    // 下轨
    const i = document.createElement('span');
    i.className = 'nsv-i';
    i.textContent = spec.kind === 'eos' ? '◆' : (spec.kind === 'budget' ? 'budget' : (idx ?? ''));
    cell.appendChild(i);

    seg.segEl.appendChild(cell);
    meta.cells.push(cell);

    // lane 若已知，立即刷新该 Unit 的标签与配色
    if (lane) this._applyUnitLane(key);
  }

  /**
   * 取或建某 Unit 的渲染 meta。
   * @param {string} key unit_index（字符串）
   * @returns {UnitRenderMeta}
   */
  _unitMeta(key) {
    let meta = this.unitMeta.get(key);
    if (!meta) {
      meta = { startLabelEl: null, usedEl: null, usedValue: null, cells: [] };
      this.unitMeta.set(key, meta);
    }
    return meta;
  }

  /**
   * lane 已知后：刷新该 Unit 起点标签文案、给所有 cell 上 lane 色、更新 used 触顶红。
   * @param {string} key unit_index（字符串）
   */
  _applyUnitLane(key) {
    const meta = this.unitMeta.get(key);
    if (!meta) return;
    const lane = this.unitLane.get(key);
    if (!lane) return;
    const limit = this._limitForUnit(key);

    for (const cell of meta.cells) {
      cell.classList.remove('lane-listen', 'lane-speak');
      cell.classList.add(`lane-${lane}`);
    }
    if (meta.startLabelEl) {
      const limitText = limit === null || limit === undefined ? '?' : limit;
      meta.startLabelEl.innerHTML =
        `U${key} <span class="b">· budget</span> ${limitText} <span class="b">·</span> ${lane}`;
    }
    if (meta.usedEl) {
      const used = meta.usedValue ?? 0;
      meta.usedEl.textContent = `used ${used}`;
      const capped = limit !== null && limit !== undefined && used >= limit;
      meta.usedEl.classList.toggle('cap', capped);
    }
  }

  /**
   * segment 收尾：校验 join(text steps) == full_text，更新头部。
   * @param {SegmentState} seg
   */
  _finalizeSegment(seg) {
    seg.card.classList.add('closed');
    if (seg.fullText !== null && seg.fullText !== undefined) {
      seg.valid = seg.textConcat === seg.fullText;
      seg.card.classList.toggle('mismatch', !seg.valid);
    }
    this._renderHead(seg);
  }

  /**
   * 渲染/刷新 segment 头部。
   * @param {SegmentState} seg
   */
  _renderHead(seg) {
    const kindText = seg.kind === 'tool_call' ? 'TOOL' : 'THINK';
    const range = seg.minUnit === null
      ? ''
      : (seg.minUnit === seg.maxUnit ? `U${seg.minUnit}` : `U${seg.minUnit}–U${seg.maxUnit}`);
    let check = '';
    if (seg.ended && seg.fullText !== null && seg.fullText !== undefined) {
      check = seg.valid === false
        ? '<span class="bad">✗ join≠full</span>'
        : '<span class="ok">✓ join==full</span>';
    }
    seg.headEl.innerHTML =
      `<span class="kind">${kindText}</span><span class="sep">·</span>`
      + (range ? `<span>${range}</span><span class="sep">·</span>` : '')
      + `<span>${seg.tokenCount} tokens</span>`
      + (check ? `<span class="sep">·</span>${check}` : '');
  }

  _scrollToBottom() {
    const scroller = this.container.parentElement;
    if (scroller) requestAnimationFrame(() => { scroller.scrollTop = scroller.scrollHeight; });
  }
}

/**
 * 轻量 HTML 转义（cell 文本用）。
 * @param {string} value 原文
 * @returns {string} 转义后文本
 */
function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

// shell 下方面板（AI Speech / Think·Tool-Call）的折叠与展开，所有 Demo 共用。
//
// 折叠状态记在浏览器 localStorage 里，跨 Demo、跨刷新保持；哪个面板折叠由 col-right 上的
// `<id>-collapsed` class 表达，布局全在 shell.css 里（两个都折叠时下方整行缩成标题条，场景面板占满）。
//
// DOM 约定：面板 `[data-collapsible="<id>"]`，其标题里的按钮 `[data-collapse-toggle="<id>"]`。

export const COLLAPSED_PANELS_KEY = 'demoGallery.collapsedPanels';
export const COLLAPSIBLE_PANELS = ['speech', 'ns'];

/** 读已保存的折叠集合；存储不可用或内容非法时视为全部展开。 */
export function loadCollapsedPanels(storage) {
  try {
    const raw = JSON.parse(storage?.getItem(COLLAPSED_PANELS_KEY) || '[]');
    return new Set(Array.isArray(raw) ? raw.filter((id) => COLLAPSIBLE_PANELS.includes(id)) : []);
  } catch {
    return new Set();
  }
}

export function saveCollapsedPanels(storage, collapsed) {
  try {
    storage?.setItem(COLLAPSED_PANELS_KEY, JSON.stringify([...collapsed].sort()));
  } catch {
    // 隐私模式等存储不可用时只是不记忆，不影响折叠本身。
  }
}

/** 切换一个面板，返回新的集合（不改入参）。 */
export function toggleCollapsedPanel(collapsed, id) {
  const next = new Set(collapsed);
  if (next.has(id)) next.delete(id);
  else next.add(id);
  return next;
}

/** 把折叠集合落到 DOM 上：容器 class、面板 class、按钮文案与 aria。 */
export function applyCollapsedPanels(container, collapsed) {
  if (!container) return;
  for (const id of COLLAPSIBLE_PANELS) {
    const isCollapsed = collapsed.has(id);
    container.classList.toggle(`${id}-collapsed`, isCollapsed);
    const panel = container.querySelector(`[data-collapsible="${id}"]`);
    panel?.classList.toggle('is-collapsed', isCollapsed);
    const button = container.querySelector(`[data-collapse-toggle="${id}"]`);
    if (button) {
      button.textContent = isCollapsed ? '展开 ▾' : '收起 ▴';
      button.setAttribute('aria-expanded', String(!isCollapsed));
    }
  }
}

/** 绑定按钮并恢复上次的折叠状态。 */
export function initPanelCollapse(container, storage = globalThis.localStorage) {
  if (!container) return;
  let collapsed = loadCollapsedPanels(storage);
  applyCollapsedPanels(container, collapsed);
  container.addEventListener('click', (event) => {
    const button = event.target.closest?.('[data-collapse-toggle]');
    if (!button || !container.contains(button)) return;
    collapsed = toggleCollapsedPanel(collapsed, button.dataset.collapseToggle);
    saveCollapsedPanels(storage, collapsed);
    applyCollapsedPanels(container, collapsed);
  });
}

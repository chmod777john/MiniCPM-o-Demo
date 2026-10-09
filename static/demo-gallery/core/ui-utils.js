// 页面与各场景共用的小工具。

export function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

export function textFromSteps(steps) {
  return (Array.isArray(steps) ? steps : [])
    .filter((step) => step?.kind === 'text')
    .map((step) => step.text || '')
    .join('');
}

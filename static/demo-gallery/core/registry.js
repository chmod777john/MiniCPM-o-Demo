// Demo Gallery 前端注册中心。
//
// 每个 demo 的 web/index.js 在模块顶层调用 registerDemo({...}) 完成自注册；
// shell 只知道后端注入的入口 URL，不认识任何具体 demo。
//
// **本文件里不允许出现任何 demo id 字面量**——这是「加一个 demo 不改共享文件」的直接编码，
// 有守卫测试盯着。

const DEMOS = new Map();

/**
 * 注册一个 demo。
 *
 * @param {{id: string, createScene: Function}} spec
 *   id 必须与 `demos/<id>/` 目录名、后端 demo_id 完全一致（有测试强制三者相等）。
 * @returns {object} 原样返回 spec
 */
export function registerDemo(spec) {
  if (!spec || typeof spec.id !== 'string' || !spec.id) {
    throw new Error('registerDemo: 必须提供字符串 id（与 demos/<id>/ 目录名一致）');
  }
  if (typeof spec.createScene !== 'function') {
    throw new Error(`registerDemo(${spec.id}): createScene 必须是函数`);
  }
  if (DEMOS.has(spec.id)) {
    throw new Error(`registerDemo(${spec.id}): 重复注册（入口模块被 import 了两次？）`);
  }
  DEMOS.set(spec.id, Object.freeze({ ...spec }));
  return spec;
}

/**
 * 取回已注册的 demo。
 *
 * 找不到就**抛错**，不静默回退到别的 demo —— 静默回退曾让 preset 错配而不报警。
 */
export function getRegisteredDemo(id) {
  const found = DEMOS.get(id);
  if (!found) {
    throw new Error(
      `demo "${id}" 未注册：demos/${id}/web/index.js 必须在模块顶层调用 `
      + `registerDemo({ id: '${id}', createScene })`,
    );
  }
  return found;
}

export function registeredDemoIds() {
  return [...DEMOS.keys()];
}

/** 仅供测试：清空注册表。 */
export function resetRegistryForTests() {
  DEMOS.clear();
}

// Gallery 入口：shell 只 import 这一个文件，import 的副作用就是自注册。
//
// `../../../static/demo-gallery/core/` 这条相对路径在浏览器（/demos/<id>/web/ → /static/...）
// 和 node 测试（<repo>/demos/<id>/web/ → <repo>/static/...）里解析到同一个模块 ——
// 这就是 demo 资源 URL 必须与磁盘路径同构的原因，别改挂载前缀。
import { registerDemo } from '../../../static/demo-gallery/core/registry.js';
import { createScene } from './scene.js';

registerDemo({ id: '_template', createScene });

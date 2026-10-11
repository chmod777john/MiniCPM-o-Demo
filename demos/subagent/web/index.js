// Gallery 入口：shell 只 import 这一个文件，import 的副作用就是自注册。
import { registerDemo } from '../../../static/demo-gallery/core/registry.js';
import { createScene } from './scene.js';

registerDemo({ id: 'subagent', createScene });

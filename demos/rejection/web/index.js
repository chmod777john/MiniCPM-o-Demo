// Rejection demo 的 Gallery 入口。说明见 demos/board/web/index.js。
import { registerDemo } from '../../../static/demo-gallery/core/registry.js';
import { createSpeakerScene } from './scene.js';

registerDemo({ id: 'rejection', createScene: createSpeakerScene });

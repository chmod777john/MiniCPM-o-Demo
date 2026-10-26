import { createScene } from '../../uicard/web/scene.js';
import css from '../../uicard/web/uicard.css?raw';
import type { SceneFactory } from '../shared/session/types';

// Read-only reuse of existing tool behavior, contained inside the React stage.
export const factory: SceneFactory = {
  create: createScene as unknown as SceneFactory['create'],
  css,
};

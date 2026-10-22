import { createBoardScene } from '../../board/web/scene.js';
import css from '../../board/web/board.css?raw';
import type { SceneFactory } from '../shared/session/types';

// Read-only reuse of existing tool behavior, contained inside the React stage.
export const factory: SceneFactory = {
  create: createBoardScene as unknown as SceneFactory['create'],
  css,
};

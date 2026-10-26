import { createSpeakerScene } from '../../rejection/web/scene.js';
import css from '../../rejection/web/rejection.css?raw';
import type { SceneFactory } from '../shared/session/types';

// Read-only reuse of existing tool behavior, contained inside the React stage.
export const factory: SceneFactory = {
  create: createSpeakerScene as unknown as SceneFactory['create'],
  css,
};

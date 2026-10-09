import { registerDemo } from '../../../../../static/demo-gallery/core/registry.js';

registerDemo({
  id: 'hello_gallery',
  createScene: ({ stageEl }) => ({
    panelTitle: 'Hello',
    render() { stageEl.innerHTML = '<p>hello gallery</p>'; },
  }),
});

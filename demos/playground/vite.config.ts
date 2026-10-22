import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const gallery = env.PLAYGROUND_GALLERY_TARGET;
  return {
    base: './',
    plugins: [react()],
    server: {
      port: 5186,
      strictPort: true,
      fs: { allow: [fileURLToPath(new URL('../..', import.meta.url))] },
      // Only application tool endpoints. Inference connects directly from the browser.
      proxy: gallery
        ? {
            '/api/demos': { target: gallery, changeOrigin: true },
            '/demos': { target: gallery, changeOrigin: true },
          }
        : undefined,
    },
    preview: { port: 5186, strictPort: true },
    build: { target: 'es2022', chunkSizeWarningLimit: 700 },
  };
});

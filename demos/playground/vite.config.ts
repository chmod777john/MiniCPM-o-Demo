import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import { applicationHandler } from './server/application';
import { fileURLToPath } from 'node:url';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const sessions = env.PLAYGROUND_SESSION_TARGET || env.PLAYGROUND_PRESET_TARGET;
  const gallery = env.PLAYGROUND_GALLERY_TARGET;
  const uicard = env.PLAYGROUND_UICARD_TARGET;
  const secure = env.PLAYGROUND_PROXY_TLS_INSECURE !== '1';
  return {
    base: './',
    plugins: [
      react(),
      {
        name: 'playground-application',
        configureServer(server) {
          const handle = applicationHandler(env);
          server.middlewares.use((req, res, next) => {
            void handle(req, res)
              .then((handled) => {
                if (!handled) next();
              })
              .catch(next);
          });
        },
        configurePreviewServer(server) {
          const handle = applicationHandler(env);
          server.middlewares.use((req, res, next) => {
            void handle(req, res)
              .then((handled) => {
                if (!handled) next();
              })
              .catch(next);
          });
        },
      },
    ],
    server: {
      port: 5186,
      strictPort: true,
      fs: { allow: [fileURLToPath(new URL('../..', import.meta.url))] },
      // Only application tool endpoints. Inference connects directly from the browser.
      proxy: {
        ...(uicard ? { '/api/demos/uicard': { target: uicard, changeOrigin: true, secure } } : {}),
        ...(gallery
          ? {
              '/api/demos': { target: gallery, changeOrigin: true, secure },
              '/demos': { target: gallery, changeOrigin: true, secure },
            }
          : {}),
        ...(sessions
          ? Object.fromEntries(
              ['/api/sessions', '/s/'].map((path) => [
                path,
                { target: sessions, changeOrigin: true, secure },
              ]),
            )
          : {}),
      },
    },
    preview: { port: 5186, strictPort: true },
    build: { target: 'es2022', chunkSizeWarningLimit: 700 },
  };
});

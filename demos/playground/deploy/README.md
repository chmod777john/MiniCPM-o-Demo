# Playground CPU application

The production Node server serves built React assets, conversation presets and bundled reference audio. It forwards only application tool/preset/assets/session routes to deployment-configured upstreams. Model discovery and realtime inference remain browser-direct to the API Base and Model ID compiled into the frontend. No `/v1` or WebSocket inference proxy is exposed.

Build from a recorded repository snapshot on a Docker host using `demos/playground/deploy/Dockerfile`. `source-snapshot.json` in the build context records source commit, any explicitly included local UI overlays and file hashes; it is retained in the image. Omit `.env*`, credentials, model files, logs, node_modules and local runtime data from the context. Reference audio belongs at `assets/ref_audio/`. Use `DEFAULT_API`, `DEFAULT_MODEL`, and `SOURCE_COMMIT` build arguments. The image uses only Node on CPU and listens on8080.

Deploy `app.yaml` with appctl and a published immutable image digest. `o5-test` is independent of the existing `o5-demo` and O45 applications. The current configuration uses9011 for application Gallery/tools and session storage, so those functions still depend on9011. Browser TLS validation also applies to the configured model API.

`PLAYGROUND_UICARD_TARGET` may override the UI Card service. `PLAYGROUND_PROXY_TLS_INSECURE=1` applies only to server-side application upstream requests; it does not change browser certificate validation. Credentials remain with the existing application tool service.

Health is `/health`; local presets readiness is `/api/demos/playground/health`. No inference is performed by either endpoint. Deployment evidence and published digest live outside this repository under `analysis-runs/20261026-o5-test-appctl`.

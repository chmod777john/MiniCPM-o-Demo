FROM node:22.23.3-bookworm-slim AS build
WORKDIR /src/demos/playground
COPY demos/playground/package.json demos/playground/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY demos/ /src/demos/
COPY static/ /src/static/
ARG DEFAULT_API=https://82.157.64.212:9011
ARG DEFAULT_MODEL=minicpm-o5
ENV VITE_PLAYGROUND_DEFAULT_API=$DEFAULT_API VITE_PLAYGROUND_DEFAULT_MODEL=$DEFAULT_MODEL
RUN npm run build

FROM node:22.23.3-bookworm-slim
ARG SOURCE_COMMIT
ENV NODE_ENV=production PORT=8080 SOURCE_COMMIT=$SOURCE_COMMIT
LABEL org.opencontainers.image.revision=$SOURCE_COMMIT
WORKDIR /app/demos/playground
COPY --from=build /src/demos/playground/dist/ ./dist/
COPY demos/playground/server/ ./server/
COPY assets/ref_audio/ /app/assets/ref_audio/
COPY source-snapshot.json /app/source-snapshot.json
USER node
EXPOSE 8080
CMD ["node", "--experimental-strip-types", "server/production.ts"]

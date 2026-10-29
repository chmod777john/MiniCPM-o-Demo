/** Same-origin model discovery for the legacy desktop Audio and Chat pages. */
(function () {
    async function resolveRealtimeUrl(mode, { signal } = {}) {
        const controller = new AbortController();
        const abort = () => controller.abort(signal.reason);
        if (signal?.aborted) abort();
        else signal?.addEventListener('abort', abort, { once: true });
        const timeout = setTimeout(() => controller.abort(new Error('Model lookup timed out. Please retry.')), 15000);
        try {
            const response = await fetch('/v1/models', { signal: controller.signal, cache: 'no-store' });
            if (!response.ok) throw new Error(`Cannot read this service's models (HTTP ${response.status}).`);
            const catalog = await response.json();
            if (catalog?.object !== 'list' || !Array.isArray(catalog.data)
                || catalog.data.some(model => !model || typeof model.id !== 'string' || !model.id.trim())) {
                throw new Error('This service returned an invalid model list.');
            }
            // Busy models stay eligible so the existing queue can handle them.
            const models = catalog.data.filter(model =>
                Array.isArray(model.capabilities?.modes) && model.capabilities.modes.includes(mode));
            if (!models.length) throw new Error(`This service has no model supporting ${mode}. Please retry after it is ready.`);
            if (models.length !== 1) throw new Error(`This page requires one ${mode}-capable model on this service. Please ask the deployment owner to configure it.`);
            const url = new URL('/v1/realtime', location.origin);
            url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            url.searchParams.set('model', models[0].id);
            url.searchParams.set('mode', mode);
            return window.ClientIdentity ? window.ClientIdentity.appendToUrl(url.toString()) : url.toString();
        } finally {
            clearTimeout(timeout);
            signal?.removeEventListener('abort', abort);
        }
    }
    window.LocalModelDiscovery = Object.freeze({ resolveRealtimeUrl });
})();

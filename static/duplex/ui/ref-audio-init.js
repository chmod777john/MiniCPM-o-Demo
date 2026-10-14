/**
 * ui/ref-audio-init.js — Shared RefAudioPlayer initialization for duplex pages
 *
 * Handles: create RefAudioPlayer instance, fetch default ref audio, wire upload/remove.
 * Returns accessor object for the current ref audio state.
 */

/**
 * Initialize a RefAudioPlayer with default ref audio fetching.
 *
 * @param {string} containerId - DOM element ID for the RefAudioPlayer container
 * @param {object} [callbacks]
 * @param {function} [callbacks.onTtsHintUpdate] - Called after default ref audio loads (for TTS hint refresh)
 * @returns {{ getBase64: () => string|null, getName: () => string, isDefault: () => boolean, rap: RefAudioPlayer }}
 */
export function initRefAudio(containerId, callbacks = {}) {
    const RefAudioPlayer = window.RefAudioPlayer;
    let base64 = null;
    let name = '';
    let isDefault = false;
    let revision = 0;

    const rap = new RefAudioPlayer(document.getElementById(containerId), {
        theme: 'light',
        onUpload(b64, n, duration) {
            revision += 1;
            base64 = b64;
            name = n;
            isDefault = false;
        },
        onRemove() {
            const requestedRevision = ++revision;
            base64 = null;
            name = '';
            isDefault = false;
            fetch('/api/default_ref_audio').then(r => r.json()).then(data => {
                if (revision !== requestedRevision) return;
                base64 = data.base64;
                name = data.name;
                isDefault = true;
                rap.setAudio(data.base64, data.name, data.duration);
            }).catch(() => {});
        },
    });

    // Load default ref audio unless a user/preset has already supplied one.
    const defaultRevision = revision;
    fetch('/api/default_ref_audio').then(r => r.json()).then(data => {
        if (revision !== defaultRevision) return;
        base64 = data.base64;
        name = data.name;
        isDefault = true;
        rap.setAudio(data.base64, data.name, data.duration);
        callbacks.onTtsHintUpdate?.();
        console.log(`Default ref audio loaded: ${data.name} (${data.duration}s)`);
    }).catch(e => {
        console.warn('Failed to load default ref audio:', e);
    });

    return {
        getBase64: () => base64,
        getName: () => name,
        isDefault: () => isDefault,
        rap,
        setAudio(b64, n, dur) {
            revision += 1;
            base64 = b64;
            name = n || '';
            isDefault = false;
            rap.setAudio(b64, n, dur);
        },
    };
}

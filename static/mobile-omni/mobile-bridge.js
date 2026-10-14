/* ============================================================
 * mobile-bridge.js
 *
 * Adapts the desktop omni page DOM (loaded by omni-app.js) for mobile:
 *   1. Force `body.video-fullscreen` so the page boots in fullscreen.
 *   2. Inject a top-left back button (returns to /mobile/).
 *   3. Re-skin bottom-right #fullscreenBtn as a gear icon that opens
 *      the React-based OmniSettingsWidget (shared with mobile app).
 *   4. Camera extras: torch toggle + pinch-to-zoom.
 *   5. Bridge `sessionStorage['mobileOmni:settings']` -> desktop DOM.
 * ========================================================= */

(function bootstrapMobileOmni() {
    function _mbT(key, fallback) {
        return window.I18n?.t?.[key] ?? fallback;
    }
    const BACK_URL = '/mobile/';

    const ICON_BACK = '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M15 18l-6-6 6-6"/></svg>';
    const ICON_GEAR = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h0a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51h0a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v0a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>';
    const ICON_TORCH = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 2h6l-1 6h-4z"/><path d="M8 8h8l-2 8h-4z"/><path d="M11 16v6"/></svg>';
    // Outward-pointing arrow over a tray — matches iOS-style "share" idiom
    // and is recognizable at 22px.
    const ICON_SHARE = '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3v12"/><path d="m7 8 5-5 5 5"/><path d="M5 14v5a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-5"/></svg>';

    function ensureFullscreenClass() {
        document.body.classList.add('video-fullscreen');
        document.body.classList.add('mobile-omni');
    }

    // ========================================================================
    // Back button (top-left)
    // ========================================================================
    function injectBackButton() {
        const container = document.getElementById('videoContainer');
        if (!container || container.querySelector('.mobile-back-btn')) return;
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'mobile-back-btn';
        btn.setAttribute('aria-label', _mbT('backToMobile', 'Back to mobile'));
        btn.innerHTML = ICON_BACK;
        const goBack = (e) => {
            if (e) { e.preventDefault(); e.stopPropagation(); }
            // Eagerly release camera + mic before navigating away so the OS
            // camera handle is freed by the time the user returns. Without
            // this we observed a black-screen camera on the *second* entry
            // into /mobile-omni/ — beforeunload alone is too late on some
            // mobile browsers (Android WebView / iOS Safari).
            try {
                if (typeof window.__omniCleanupMedia === 'function') {
                    window.__omniCleanupMedia();
                }
            } catch (_) {}
            try { window.location.assign(BACK_URL); } catch (_) { window.location.href = BACK_URL; }
        };
        btn.addEventListener('click', goBack);
        btn.addEventListener('touchend', goBack, { passive: false });
        container.appendChild(btn);
    }

    // ========================================================================
    // React Settings Widget (shared with mobile app)
    // ========================================================================
    let widgetApi = null;

    function getDesktopInput(id) {
        return document.getElementById(id);
    }

    function setDesktopInput(id, value) {
        const el = document.getElementById(id);
        if (!el) return;
        el.value = value;
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
    }

    const numericOptions = {
        maxKvTokens: ['maxKvTokens', 262144], playbackDelayMs: ['playbackDelay', 200],
        basicWindowHighTokens: ['basicWindowHighTokens', 8000], basicWindowLowTokens: ['basicWindowLowTokens', 6000],
        contextMaxUnits: ['contextMaxUnits', 250], textSink: ['textSink', 512], textSlidingWindow: ['textSlidingWindow', 10240],
        speakThreshold: ['speakThreshold', 0.3], visionFixedSliceNums: ['visionFixedSliceNums', 2], visionStrategySliceNums: ['visionStrategySliceNums', 4],
    };
    function getDuplexOptions() {
        const result = {};
        for (const [key, [id, fallback]] of Object.entries(numericOptions)) {
            const value = Number(getDesktopInput(id)?.value);
            result[key] = Number.isFinite(value) ? value : fallback;
        }
        result.slidingWindowMode = document.querySelector('input[name="slidingWindowMode"]:checked')?.value || 'basic';
        result.listenSpeakDecision = !!getDesktopInput('listenSpeakDecisionEnabled')?.checked;
        result.useTts = !!getDesktopInput('ttsEnabled')?.checked;
        result.ttsRefAudioBase64 = window.__omniTtsRef?.mode === 'independent' ? window.__omniTtsRef.audioBase64 : null;
        result.visionHDMode = getDesktopInput('visionHDMode')?.value || 'fixed';
        return result;
    }
    function setDuplexOptions(value) {
        if ('ttsRefAudioBase64' in value && window.__omniTtsRef) {
            const ctrl = window.__omniTtsRef;
            ctrl.audioBase64 = value.ttsRefAudioBase64;
            const mode = value.ttsRefAudioBase64 ? 'independent' : 'extract';
            const radio = document.querySelector(`input[name="omniTtsRefMode"][value="${mode}"]`);
            if (radio) radio.checked = true;
            ctrl.onModeChange();
            if (value.ttsRefAudioBase64) ctrl.rap?.setAudio(value.ttsRefAudioBase64, 'Mobile TTS reference', 0);
            else ctrl.rap?.clear();
        }
        for (const [key, [id]] of Object.entries(numericOptions)) {
            if (Number.isFinite(value[key])) setDesktopInput(id, String(value[key]));
        }
        for (const [key, id] of [['listenSpeakDecision', 'listenSpeakDecisionEnabled'], ['useTts', 'ttsEnabled']]) {
            const el = getDesktopInput(id);
            if (el && typeof value[key] === 'boolean') {
                el.checked = value[key];
                el.dispatchEvent(new Event('change', { bubbles: true }));
            }
        }
        if (value.visionHDMode) setDesktopInput('visionHDMode', value.visionHDMode);
        const radio = [...document.querySelectorAll('input[name="slidingWindowMode"]')].find((el) => el.value === value.slidingWindowMode && !el.disabled);
        if (radio) { radio.checked = true; radio.dispatchEvent(new Event('change', { bubbles: true })); }
    }
    function buildBridge() {
        return {
            getDuplexOptions,
            setDuplexOptions,
            getAllowedWindowModes() {
                return [...document.querySelectorAll('input[name="slidingWindowMode"]')].filter((el) => !el.disabled).map((el) => el.value);
            },
            isSessionActive() { return !!getDesktopInput('btnStart')?.disabled; },
            getSystemPrompt() {
                return getDesktopInput('systemPrompt')?.value || '';
            },
            setSystemPrompt(v) {
                setDesktopInput('systemPrompt', v);
            },
            getRefAudioBase64() {
                return window.__omniRefAudio?.getBase64() || null;
            },
            setRefAudioBase64(b64, name, duration) {
                window.__omniRefAudio?.setAudio(b64, name, duration);
            },
        };
    }

    function mountSettingsWidget() {
        if (typeof window.mountOmniSettings !== 'function') {
            console.warn('[mobile-omni] settings-widget.js not loaded');
            return;
        }
        const container = document.createElement('div');
        container.id = 'omniSettingsWidgetRoot';
        document.body.appendChild(container);
        const bridge = buildBridge();
        widgetApi = window.mountOmniSettings(container, bridge);
    }

    function openSettingsSheet() {
        if (widgetApi) {
            widgetApi.open();
        }
    }

    // ========================================================================
    // Gear button on bottom-right
    // ========================================================================
    function rebindFullscreenButtonAsGear() {
        const btn = document.getElementById('fullscreenBtn');
        if (!btn) return;
        const clone = btn.cloneNode(false);
        clone.id = 'fullscreenBtn';
        clone.className = btn.className;
        clone.classList.add('visible');
        clone.setAttribute('title', _mbT('settings', 'Settings'));
        clone.setAttribute('aria-label', _mbT('openSettings', 'Open settings'));
        clone.innerHTML = ICON_GEAR;
        btn.parentNode.replaceChild(clone, btn);
        const open = (e) => { if (e) { e.preventDefault(); e.stopPropagation(); } openSettingsSheet(); };
        clone.addEventListener('click', open);
        clone.addEventListener('touchend', open, { passive: false });
    }

    function keepGearVisible() {
        const btn = document.getElementById('fullscreenBtn');
        if (!btn) return;
        const obs = new MutationObserver(() => {
            if (!btn.classList.contains('visible')) btn.classList.add('visible');
        });
        obs.observe(btn, { attributes: true, attributeFilter: ['class'] });
    }

    // ========================================================================
    // Stage share button — sits inside #videoContainer right next to the
    // gear. We learned the hard way that a body-level position:fixed
    // element gets covered by something in this WebView, but injecting
    // into #videoContainer (same parent + stacking context as the gear)
    // works reliably. The button's click is forwarded to the original
    // hidden #save-share-container .ss-btn so all downstream logic
    // (comment modal, upload, copy link, toast) runs unchanged via
    // SaveShareUI from save-share.js.
    // ========================================================================
    function injectStageMetrics() {
        const container = document.getElementById('videoContainer');
        const kv = document.getElementById('kvCacheDisplay');
        const count = document.getElementById('windowCountDisplay');
        const latest = document.getElementById('windowLatestDisplay');
        if (!container || !kv || !count || !latest) return;
        const badge = document.createElement('div');
        badge.className = 'mb-stage-metrics';
        badge.setAttribute('aria-live', 'polite');
        const refresh = () => {
            badge.textContent = `KV ${kv.textContent || '0'} · ✂ ${count.textContent || '0'} · ${latest.textContent || ''}`;
            badge.title = latest.title;
        };
        refresh();
        container.appendChild(badge);
        const observer = new MutationObserver(refresh);
        for (const el of [kv, count, latest]) observer.observe(el, { childList: true, characterData: true, subtree: true, attributes: true, attributeFilter: ['title'] });
        window.addEventListener('pagehide', () => observer.disconnect(), { once: true });
    }

    function injectStageShareButton() {
        const container = document.getElementById('videoContainer');
        if (!container) return;
        if (container.querySelector('.mb-stage-share-btn')) return;
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'mb-stage-share-btn';
        btn.setAttribute('aria-label', _mbT('shareCallLabel', 'Share call'));
        btn.title = _mbT('share', 'Share');
        btn.innerHTML = ICON_SHARE;
        // Inline styles to bypass any specificity / load-order surprises.
        // Mirrors the gear's geometry (.fullscreen-btn): 36x36 at bottom:70
        // with a glass background. Sits at right:56 (gear-right 12 + gear
        // width 36 + gap 8).
        btn.style.cssText = [
            'position:absolute',
            'bottom:calc(env(safe-area-inset-bottom, 0px) + 70px)',
            'right:56px',
            'width:36px',
            'height:36px',
            'padding:0',
            'border:0',
            'border-radius:10px',
            'background:rgba(255,255,255,0.2)',
            '-webkit-backdrop-filter:blur(8px)',
            'backdrop-filter:blur(8px)',
            'color:#fff',
            'display:flex',
            'align-items:center',
            'justify-content:center',
            'z-index:200',
            'cursor:pointer',
            '-webkit-tap-highlight-color:rgba(255,255,255,0.3)',
        ].join(';');

        function syncDisabled() {
            const src = document.querySelector('#save-share-container .ss-btn');
            const disabled = !src || !!src.disabled;
            btn.disabled = disabled;
            btn.style.opacity = disabled ? '0.5' : '1';
            btn.style.cursor = disabled ? 'default' : 'pointer';
        }

        btn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            const src = document.querySelector('#save-share-container .ss-btn');
            if (!src || src.disabled) return;
            setTimeout(() => src.click(), 0);
        });

        container.appendChild(btn);
        syncDisabled();

        // SaveShareUI may not be ready yet — re-check until it lands,
        // then attach a MutationObserver for live disabled-state sync.
        const tryAttach = () => {
            const src = document.querySelector('#save-share-container .ss-btn');
            if (!src) {
                setTimeout(tryAttach, 100);
                return;
            }
            syncDisabled();
            new MutationObserver(syncDisabled).observe(src, {
                attributes: true,
                attributeFilter: ['disabled'],
            });
        };
        tryAttach();
    }

    // ========================================================================
    // Session storage bridge from /mobile/
    // ========================================================================
    function applyMobileSettings() {
        let raw;
        try { raw = sessionStorage.getItem('mobileOmni:settings'); } catch (_) { return; }
        if (!raw) return;
        let s;
        try { s = JSON.parse(raw); } catch (_) { return; }
        if (!s || typeof s !== 'object') return;
        if (typeof s.systemPrompt === 'string') setDesktopInput('systemPrompt', s.systemPrompt);
        // Match the service defaults already applied by the desktop runtime.
        if (s.options) setDuplexOptions(s.options);
        if (s.refAudio?.base64) buildBridge().setRefAudioBase64(s.refAudio.base64, s.refAudio.name || 'Mobile reference', s.refAudio.duration || 0);
    }

    // ========================================================================
    // Camera extras: torch + pinch zoom
    // ========================================================================
    let torchBtn = null;
    let lastVideoTrack = null;
    let pinchState = null;

    function getActiveVideoTrack() {
        const v = document.getElementById('videoEl');
        if (!v || !v.srcObject) return null;
        const stream = v.srcObject;
        if (typeof stream.getVideoTracks !== 'function') return null;
        const tracks = stream.getVideoTracks();
        return tracks && tracks.length ? tracks[0] : null;
    }

    function injectTorchButton() {
        const container = document.getElementById('videoContainer');
        if (!container) return;
        if (container.querySelector('.mobile-torch-btn')) {
            torchBtn = container.querySelector('.mobile-torch-btn');
            return;
        }
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'mobile-torch-btn';
        btn.setAttribute('aria-label', _mbT('toggleFlashlight', 'Toggle flashlight'));
        btn.innerHTML = ICON_TORCH;
        btn.addEventListener('click', toggleTorch);
        btn.addEventListener('touchend', (e) => { e.preventDefault(); toggleTorch(); }, { passive: false });
        container.appendChild(btn);
        torchBtn = btn;
    }

    async function toggleTorch() {
        const track = getActiveVideoTrack();
        if (!track || typeof track.applyConstraints !== 'function') return;
        const caps = typeof track.getCapabilities === 'function' ? track.getCapabilities() : {};
        if (!caps || !caps.torch) return;
        const settings = typeof track.getSettings === 'function' ? track.getSettings() : {};
        const next = !settings.torch;
        try {
            await track.applyConstraints({ advanced: [{ torch: next }] });
            torchBtn?.classList.toggle('active', next);
        } catch (err) {
            console.warn('[mobile-omni] torch toggle failed:', err);
        }
    }

    function refreshCameraCapabilities() {
        const track = getActiveVideoTrack();
        if (track === lastVideoTrack) { updateTorchVisibility(track); return; }
        lastVideoTrack = track;
        torchBtn?.classList.remove('active');
        updateTorchVisibility(track);
    }

    function updateTorchVisibility(track) {
        if (!torchBtn) return;
        let supported = false;
        if (track && typeof track.getCapabilities === 'function') {
            try { supported = !!(track.getCapabilities().torch); } catch (_) {}
        }
        torchBtn.classList.toggle('visible', supported);
    }

    function watchVideoElement() {
        setInterval(refreshCameraCapabilities, 1000);
    }

    function distance(t1, t2) { return Math.hypot(t1.clientX - t2.clientX, t1.clientY - t2.clientY); }

    async function handlePinchStart(e) {
        if (e.touches.length !== 2) return;
        const track = getActiveVideoTrack();
        if (!track || typeof track.getCapabilities !== 'function') return;
        let caps = {};
        try { caps = track.getCapabilities() || {}; } catch (_) {}
        if (!caps.zoom) return;
        const settings = typeof track.getSettings === 'function' ? track.getSettings() : {};
        pinchState = {
            startDist: distance(e.touches[0], e.touches[1]),
            startZoom: settings.zoom || caps.zoom.min || 1,
            min: caps.zoom.min || 1,
            max: caps.zoom.max || 1,
            track,
        };
        e.preventDefault();
    }

    function handlePinchMove(e) {
        if (!pinchState || e.touches.length !== 2) return;
        e.preventDefault();
        const d = distance(e.touches[0], e.touches[1]);
        const ratio = d / pinchState.startDist;
        let next = pinchState.startZoom * ratio;
        if (next < pinchState.min) next = pinchState.min;
        if (next > pinchState.max) next = pinchState.max;
        pinchState.track.applyConstraints({ advanced: [{ zoom: next }] }).catch(() => {});
    }

    function handlePinchEnd(e) {
        if (e.touches.length < 2) pinchState = null;
    }

    function bindPinchZoom() {
        const v = document.getElementById('videoContainer');
        if (!v) return;
        v.addEventListener('touchstart', handlePinchStart, { passive: false });
        v.addEventListener('touchmove', handlePinchMove, { passive: false });
        v.addEventListener('touchend', handlePinchEnd, { passive: false });
        v.addEventListener('touchcancel', handlePinchEnd, { passive: false });
    }

    // ========================================================================
    // Init
    // ========================================================================
    function init() {
        ensureFullscreenClass();
        setTimeout(() => {
            injectBackButton();
            mountSettingsWidget();
            rebindFullscreenButtonAsGear();
            keepGearVisible();
            Promise.resolve(window.__omniFrontendReady).then(applyMobileSettings).catch(console.warn);
            injectTorchButton();
            watchVideoElement();
            bindPinchZoom();
            injectStageShareButton();
            injectStageMetrics();
        }, 0);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init, { once: true });
    } else {
        init();
    }
})();

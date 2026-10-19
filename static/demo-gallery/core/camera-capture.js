// 摄像头采集(omni 模式):预览 + 每个音频 Unit 结束时截一帧 JPEG。
//
// 训推口径:模型每个 Unit 看「这 1 秒结束时的那一帧」(SDK user_video 取帧规则 frame_phase=1.0),
// 所以由麦克风每攒满 1 秒回调时调用 captureFrame() 即可。
//
// 一个会话里视觉 token 数由帧宽高决定,必须恒定 —— 中途翻转前后摄像头时两颗镜头的分辨率可能不同。
// 因此会话开始时 lockCaptureSize() 按当时画面的纵横比定死输出尺寸(长边 640),之后每帧都画到这个
// 尺寸的 canvas 上(不同纵横比的画面居中裁切),保证整个会话帧尺寸不变。

const LONG_EDGE = 640;
const JPEG_QUALITY = 0.8;

/** 按画面纵横比选输出尺寸:横屏 640×480 / 640×360,竖屏反过来。挑与实际比例最接近的一档。 */
export function chooseCaptureSize(width, height) {
  if (!width || !height) throw new Error('摄像头还没有画面');
  const ratio = width / height;
  const candidates = ratio >= 1
    ? [[LONG_EDGE, 480], [LONG_EDGE, 360]]
    : [[480, LONG_EDGE], [360, LONG_EDGE]];
  let best = candidates[0];
  for (const c of candidates) {
    if (Math.abs(c[0] / c[1] - ratio) < Math.abs(best[0] / best[1] - ratio)) best = c;
  }
  return { width: best[0], height: best[1] };
}

/** 把 (sw, sh) 的画面「居中裁切」铺满 (dw, dh):返回 drawImage 的源矩形。 */
export function coverCrop(sw, sh, dw, dh) {
  const scale = Math.max(dw / sw, dh / sh);
  const cw = dw / scale;
  const ch = dh / scale;
  return { sx: (sw - cw) / 2, sy: (sh - ch) / 2, sw: cw, sh: ch };
}

export class CameraCapture {
  /**
   * @param {HTMLVideoElement} videoEl 预览用的 video 元素。
   * @param {object} [hooks] onState(text) 状态回调。
   */
  constructor(videoEl, { onState } = {}) {
    this.videoEl = videoEl;
    this.onState = onState;
    this.stream = null;
    this.facing = 'user';
    this.mirror = true;            // 前置默认镜像,符合照镜子的直觉;后置默认不镜像
    this.size = null;              // 会话内锁定的输出尺寸
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d');
  }

  get active() { return Boolean(this.stream); }

  /** 申请相机权限并开始预览。失败抛出带人话的错误。 */
  async open(facing = this.facing) {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('浏览器不支持摄像头(需要 HTTPS 访问)');
    }
    this.close();
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 } },
      });
    } catch (err) {
      const denied = err?.name === 'NotAllowedError' || err?.name === 'SecurityError';
      throw new Error(denied ? '没有摄像头权限:请在浏览器地址栏允许使用摄像头' : `打不开摄像头:${err?.message || err}`);
    }
    this.facing = facing;
    this.mirror = facing === 'user';
    this.videoEl.srcObject = this.stream;
    this.videoEl.muted = true;
    this.videoEl.playsInline = true;
    await this.videoEl.play().catch(() => {});
    await this._waitForFirstFrame();
    this._applyPreviewMirror();
    this.onState?.(`摄像头已开启(${facing === 'user' ? '前置' : '后置'} ${this.videoEl.videoWidth}×${this.videoEl.videoHeight})`);
  }

  async _waitForFirstFrame() {
    const v = this.videoEl;
    for (let i = 0; i < 50 && !v.videoWidth; i++) await new Promise((r) => setTimeout(r, 100));
    if (!v.videoWidth) throw new Error('摄像头没有画面');
  }

  /** 前后摄像头切换(手机)。会话中切换也安全:输出尺寸已锁定。 */
  async flip() {
    const next = this.facing === 'user' ? 'environment' : 'user';
    const previous = this.facing;
    try {
      await this.open(next);
    } catch (err) {
      await this.open(previous).catch(() => {});
      throw err;
    }
  }

  toggleMirror() {
    this.mirror = !this.mirror;
    this._applyPreviewMirror();
  }

  _applyPreviewMirror() {
    this.videoEl.style.transform = this.mirror ? 'scaleX(-1)' : 'none';
  }

  /** 会话开始时调用:按当前画面定死本会话的帧尺寸。 */
  lockCaptureSize() {
    this.size = chooseCaptureSize(this.videoEl.videoWidth, this.videoEl.videoHeight);
    this.canvas.width = this.size.width;
    this.canvas.height = this.size.height;
    return this.size;
  }

  unlockCaptureSize() { this.size = null; }

  /** 截当前一帧,返回 JPEG 的 base64(无 data: 前缀)。画面暂时不可用时返回 null。 */
  captureFrame() {
    const v = this.videoEl;
    if (!this.stream || !this.size || !v.videoWidth) return null;
    const { width, height } = this.size;
    const crop = coverCrop(v.videoWidth, v.videoHeight, width, height);
    const ctx = this.ctx;
    ctx.save();
    if (this.mirror) {
      ctx.translate(width, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(v, crop.sx, crop.sy, crop.sw, crop.sh, 0, 0, width, height);
    ctx.restore();
    return this.canvas.toDataURL('image/jpeg', JPEG_QUALITY).split(',')[1];
  }

  close() {
    if (this.stream) {
      for (const track of this.stream.getTracks()) track.stop();
    }
    this.stream = null;
    this.videoEl.srcObject = null;
  }
}

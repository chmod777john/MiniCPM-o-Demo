# LIS分容器本地验证

## 2026-10-14 计划/认领

Owner: docker_lis_package; root 已明确授权。仅自身 packaging WT `o5-fc-dev-external-hf-20261013-hf-chat-20261014-docker-lis-20261014`，基准30352d29。数值源码、HF/accel及固定runtime继承v9 `sha256:93aa6ee9df9abc96669d09235bcbd499be6a369c4a4fa63ec3507e3402afaaf2`。

资源：34两卡无compute/3MiB，新Docker network `o5-lis-split-20261014`；容器 `o5-lis-split-gateway-20261014` 和 `o5-lis-split-worker-20261014`；host端口18024、registry18025；内部非默认端口18006/18007、22424/22524。避让已有所有旧服务、Pro5000 caitianchi任务。root确认独占这次34双卡测试。模型与vocoder只读挂载原任务目录model。

目标：固定gateway/worker镜像，gateway遵守平台CLI和只读/app/config.json；backend bind0.0.0.0；worker注册重试、定期恢复；独立容器的health、API、gateway重启恢复。仅本地模拟已核对历史contract，不push镜像，不登录/注册/部署LIS。

## 机制与风险

历史builder只给gateway挂载runtime.json/config.json，worker只挂模型。因此worker的注册策略固化在镜像内，不能假设worker会读到平台ConfigMap。本次明确选择允许的 `periodic_idempotent`，refresh20s、max_backoff20s，与待用template声明保持一致；非声称所有旧template都强制periodic。初次连接失败保留已加载model，1/2/4/8/16/20s重试。HTTP有限timeout且shutdown可唤醒等待。

原有gateway `WorkerPool.register_worker` 相同endpoint不清除busy状态；其健康更新也保留gateway已dispatch但worker瞬时idle的保护。重复PUT可复用此实现，不改Demo服务源码。

验证门：CPU入口/readonly配置/参数/注册恢复/退出；镜像内源码hash；分容器初始registry不可达；真实TP2 API；重启gateway保留worker/backend PID并恢复可调用；配置hash不变/recording=false；正常退出GPU释放。未通过前不把本地结果当作LIS在线验收。

## 实测进行中

实现提交78aa5c45。worker镜像3cc2bcfa3af6dbcdd0449a8af6ffe90bc2c2d870beb924c1ee9edb35b0de208f，gateway49b13ac0c55621b484b5fae4ad29bd9740fa61b9e1ed97639f7390393af3b025。初始gateway不存在，worker model/backend/worker正常ready，DNS注册失败已记录1/2/4/8/16/20s退避而未退出；GPU60065/24795MiB。启动gateway进行恢复。SSH隧道工具session71735占本机18024/18025，由本agent负责关闭。

## 最终v11验收与收尾（2026-10-14 07:47UTC）

package实现78aa5c45，flat交付输入修正c8f7bba8；root已分别合入41abd31a/e506533b。最终worker `sha256:e2a1f41bd37d50343b3882a4fb7f62fe7aae45b1adf4e30166f257c83af3962b`；gateway `sha256:d7e3b5ed6619a78b9e0d9e327cf507ef3830681b4c9df692cd35c75d11c0e3dd`。本机tag `o5-external-hf-{worker,gateway}:20261014-split-v11`，未推送。package全部文件在两容器内逐文件对manifest核SHA通过；entrypoint SHA650df93a98ab2106c7c3cc179f0bb4a7dbfadf6d460e331ff4485d222acf658a。v11仅相对v10修交付contract/document，不改变执行代码，但已重新做全部API门。

- worker先起，gateway缺失时模型与worker正常ready，观察DNS失败1/2/4/8/16/20s重试；随后gateway启动自动成功注册。
- 最终image文字+真实音频单工2项、普通audio/video各2unit、FC8unit均PASS。FC3真实模型tool-call、2录制结果被消费、9spoken.delta；5份PCM经root独立核对finite/nonzero。不是在线外部工具调用质量评测。
- 活跃普通会话26unit，31.922s；26次采样同一ticket q_4c0dee472126且duplex_active，第二连接始终queued；会话期间真实定期PUT日志将在registration-overlap.json中归档。
- 单独停gateway22s，worker容器ID/StartedAt、父/模型backend/worker PID全部保持，未重载模型。gateway重新启动后9.155s恢复total=1,idle=1。随后文字+音频2项API再PASS。
- 保留平台readonly config SHA2da49d6ae6ce5072cd561f40045f767ebc48eaa7aed250393cafe3b0ea8bf549；effective recording=false。非默认端口与Pod样式DNS成功；gateway容器跨网络访问worker22424/backend22524 health成功。
- 07:47UTC停止两本任务容器均exit0/noOOM；保留重命名-v11-completed以供inspect；删除本任务无连接network。双GPU恢复3MiB，无compute；已停止本任务SSH tunnel pid3912826。

限制：仅34两张96GBPro6000上Pro5000 TP2预设的本地部署验收，非实体Pro5000显存/性能；FC客户端耗时0.498–1.462s，不声称所有unit实时；未重新hidden/logits paired（本轮无推理/数值修改）；ordinary小probe均listen，未强制触发basic裁剪。gateway重启恢复的是新请求服务，不是活跃会话透明续传。未推新镜像、未登录/注册/部署LIS。资源16CPU/192GiB是保守示例，未测最低值，GPU selector必须按平台资源目录替换。

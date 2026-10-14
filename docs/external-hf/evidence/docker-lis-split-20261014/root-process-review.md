# Split入口执行链复核

2026-10-14 UTC。Root只读审查packager提交78aa5c45；没有占用GPU/修改其worktree。

## 执行链

- configure从只读平台config或显式O5_DEMO_CONFIG_PATH读取，保留内容，在独立/tmp目录生成有效配置；worker补原始ckpt路径和service.compile=false。当前Demo config.py中的compile实际属于service，写入位置正确。
- worker先启动backend并等待health，再启动worker并等待health；backend监听0.0.0.0，worker经localhost访问。各child独立process group，父监督退出会回收child。
- registration使用单线程父监督每tick最多一个HTTP调用，timeout5s；错误不销毁已加载模型，backoff上限20s。成功后20s刷新。此为所选periodic_idempotent契约，不表示所有历史模板都要求此策略。
- gateway相同worker ID/endpoint重复PUT时，register_worker不会走endpoint_changed清理分支。其_refresh_worker_status保留_gateway_dispatched时的idle保护，所以源码层无直接把活跃ticket清空的路径。需要真实活跃session跨refresh检查，已要求packager执行。
- gateway重启会失去内存连接/会话；重新注册是新请求可用性恢复，不能声称活跃会话透明恢复。
- HTTP200健康由既有Demo实现；模型初始化等待在backend lifespan，父进程有child退出检测。不将包装health检查当完整模型功能测试。

## 已重复的CPU门

在项目venv、PYTHONDONTWRITEBYTECODE=1下执行当前packager提交：test_entrypoint.py、test_lis_contract.py、test_launcher_image.py均PASS。覆盖进程消失/超时/信号与Popen竞态、只读配置、raw MODEL_PATH、非默认CLI、注册失败/恢复/周期刷新/stop与镜像ID解析。

## 待收尾

- 实际双容器首次registry不可达、非默认端口、API、gateway重启及worker/backend PID保持待packager提供。
- 活跃会话跨20s refresh及第二连接排队待实测。
- 当前lis-contract.json为nested说明片段，不可直接作为create-realtime --contract-json输入；其registration_status.service_role=gateway还不符合历史public/registry枚举。已通知独立reviewer/packager，最终需给出通过固定schema的flat交付输入。
- 当前在线LIS未验；不做publish/注册。Pro6000本地结果不替代Pro5000真机显存/性能。

## 固定LIS schema独立CPU核对

Root读取packager已提交c8f7bba8，使用项目venv直接import固定lis-source@08f73fd。flat contract/resources经build_realtime_base_profiles和parse_realtime_base_profiles往返通过，periodic/public/2GPU保持；merge_realtime_base_profiles对错误gateway role明确拒绝，对nested说明文件作为flat输入也明确拒绝。未执行lisctl创建操作。资源selector和16CPU/192GiB仅示例，不是当前平台GPU目录确认或内存最低值实测。

## v11 API及源码独立复核

Root按真实events.jsonl重新计数，FC8个unit/3个tool call/2个已消费结果/9个spoken.delta，与summary逐项一致；5份float32 PCM均finite且非零。重启前、后各两项单工文字/语音API均PASS；普通音频/视频各2个listen unit均finalized，不能代表说话或实际滑窗裁剪验收。原始核对见ROOT_SPLIT_V11_API_AUDIT.json。

active26会话持续31.922秒，26个worker快照均保持duplex_active和同一ticket，第二客户端始终排队。实际PUT时间与会话交集等最终gateway日志落盘后单独核对。

本轮root HEAD e506533b与v11 release-input-manifest-split-v11.json声明的split_packaging全部18个文件SHA256逐项相同；manifest固定源提交c8f7bba8。已合入的源码与接受镜像声明一致，容器内文件证据仍以packager最终归档为准。

## v11最终生命周期核对

Root直接读取gateway-final.log，确认07:42:40.841和07:43:00.866 UTC两次PUT200落在31.922秒活跃会话内。前述待核对项已经闭环，活动ticket和排队保护不是仅以持续时间推断。

gateway重启前/停机中/恢复后的worker容器ID、StartedAt、父进程558209、backend558460、worker558797逐项完全相同；gateway重启后9.1549秒恢复1个idle worker。平台只读config hash在前后保持一致，recording=false。worker/gateway镜像身份与清理记录一致，镜像内manifest hash和本地文件一致，package文件校验全通过。

两容器退出码0、无OOM；停止后两卡均3MiB，无compute进程。network与SSH隧道回收由资源owner记录。此次验证是本地Docker中历史LIS契约模拟，未进行LIS在线部署，也未替代Pro5000真机验收。

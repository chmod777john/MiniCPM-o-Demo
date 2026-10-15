# 大checkpoint上传时的跨文件并发冲突记录

日期2026-10-15。仅部署证据与推断，不修改LIS服务端。

## 观察

lisctl1.2.35，9文件共75,920,482,457B，model.pt74,606,011,097B，单part64MiB。`--file-concurrency 2 --upload-concurrency 8`创建job b6b4c36b-ba03-49f0-bdd4-5cb10ee972db 后，小文件SHA256SUMS完成561B，大model.pt分批请求预签URL时得到两个不同upload_id，客户端按设计拒绝继续。

日志时序：
- 08:55:18创建job。
- 08:55:19.156小文件预签完成。
- 08:55:19.450小文件complete完成。
- 08:55:19.549/19.839大文件分批预签返回。
- 08:55:19.848客户端中止；自动DELETE content返回404。
- 08:57:14确认owner与状态后，root通过upload abort删除自己的失败job。

单个model.pt约1112part，CLI每批最多500个，所以需要多次预签；不同file并发发生于首次大文件预签与小文件complete之间。

## 源码推断

固定源码lis-source@08f73fd：`routers/asset_uploads.py:create_file_multipart_part_urls`对job行加锁、写`asset_metadata._obs_file_uploads`；`complete_file_multipart_upload`未取行锁，读取旧asset_metadata、等待OBS complete后把整个metadata回写。因此小文件complete可能覆盖同时提交的大文件upload_id，下一批预签就初始化新upload_id。

未证明线上部署等于该固定源码；不把推断写成已证明的在线根因。报错是现场实测。

## 本次处理

无需修改权重、拆分model.pt或修改平台代码。改`--file-concurrency 1 --upload-concurrency 8`，串行处理不同文件，单文件仍8part并发。新job af790c8d-bb41-4a9e-8233-f64559520950 已将全部9文件上传至OBS，随后进入平台OBS→SWR阶段。对比支持跨文件竞争解释，但不是平台修复的回归测试。

已完成的8个vocoder文件平台SHA256逐项等于本地已有manifest，checkpoint转存完成后继续核对其SHA256。原始证据model-attempt2-progress.jsonl、model-attempt2-status.json、vocoder-upload-sha-check.json。

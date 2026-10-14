# FC 双工与普通双工特性兼容性审查

日期：2026-10-14。Owner：fc_feature_compat_review。任务：只读审查，未修改三物料、未启动 GPU/容器、未执行新增测试。

## 1. 结论

**这些能力并不因“FC”而原则上互斥，但当前 FC 并没有自动继承普通双工的全部策略。** 共享 LLM、StreamDecoder、TTS 和 accel，只说明基础执行模块共享；普通/FC capability 仍各自安排 slot、unit、窗口和采样生命周期。

| 特性 | 机制上 | 当前 FC 路径 | 已有证据与缺口 |
|---|---|---|---|
| Basic 滑窗 | 可以做；需保留系统前缀、完整 unit 和跨 unit 语义所需状态 | **未接通**：无 system/unit 注册、无 enforce 调用；默认 window=off | 普通 binary/basic 24unit、3次裁剪是普通路径证据。FC 无实际裁剪验收 |
| KV surgery / context 窗口 | 可设计 FC 版本；**不可照搬普通的单段生成文本搬运** | **未接通**；普通布局假设与 FC 双轨 slot 不同 | 未验证 FC；需要多段span/协议保留策略 |
| Listen/speak | FC 协议本来就包含该决策；语音与非语音轨道独立 | 有 FC FSM 约束解码；**无普通 `speak_threshold`/binary 配置透传** | constrained 下，turn关闭首步是listen/speak；legacy续说时是speak/pad，不能强行都改二选一 |
| 3D-RoPE：音频/文本/tool event | 兼容，均使用共享 canvas decoder；该类token的T/H/W相同 | **已接入通用位置路径** | 39次fixed-feed精度门与短FC API支持此范围；不代表视觉/长上下文裁剪 |
| 3D-RoPE：图片/视频 | 原理兼容，需要FC图像slot和canvas spec | **未实现：`frame_list`被直接丢弃** | API接受并解码帧，但模型不使用；不能把API成功当FC视频成功 |
| TTS normalization / causal vocoder / graph hidden所有权 | 与FC控制流相容，底层模型与加速可共享 | 关键修复已接通；保留FC自己的condition与turn策略 | 已有短FC语音返回与causal尾部门；尚无FC pad/EOS/跨窗的完整数值/音质验证；FA2不是当前默认启用 |

当前已经跑通的音频 FC API 不因本报告被推翻。需要撤回/避免的是“FC 已经具有普通双工的所有选项、视觉3D位置、窗口和相同TTS策略”的泛化。

## 2. 审查快照与来源

下文 `HF`、`Demo`、`Accel` 为以下绝对路径，引用格式为路径内的文件:行号。

- HF `/user/weihongliang/worktrees/minicpm-o-5-fix-3d-rope-o5fc-20261013-hf-chat-20261014`，HEAD `5fea76841a268720e83ce68bd14f8c064ad72bde`。
- Demo `/user/weihongliang/worktrees/o5-fc-dev-external-hf-20261013-hf-chat-20261014`，取证时 HEAD `51ecc6a8a7e6c8a67bc38f28e05b9c10c89bec66`。root正在合入包装/文档提交；本报告所引用API/模型桥接代码的hash固定在 `SOURCE_FACTS.json`。
- Accel `/user/weihongliang/worktrees/hf-accel-no-fc-3d-rope-20261009-canvas-window-20261010-combined-batch-tts-20261010-exact-window-20261010-bulk-window-20261010-paired-rotation-20261010-direct-table-20261010-o5fc`，HEAD `6e33931cdf7e1fcff2f896d706bb73b949fc4a39`。
- 同目录 `SOURCE_FACTS.json` 记录源码 SHA256、Git状态、AST调用定位。它是静态提取证据，不是运行时测试。
- 当前FC来源是 `o5-fc-dev@732b075841f837d67d9f512835cbf1aec174911e`，由HF `fc/migration/README.md`和manifest固定；不是此次再读远端最新。manifest中 `streaming_prefill`、`streaming_spoken_generate`、`_tts_condition` 都是来源AST原样；`_spoken_audio`才有具名causal修正。因此“丢帧/现有FC决策方式”等并非本轮移植时意外删掉普通路径，而是上游FC能力边界保留到了HF。

## 3. 真实调用链和 API 风险

### 普通双工

`Demo py_backend/server.py:403`判断FC开关。普通分支:414读取config，:421调用 `backend.set_duplex_config`。
`modeling/external_o5_facade.py:292`用 `ExternalO5DuplexConfig`验证（schema `core/schemas/external_o5_duplex.py:8`为extra=forbid），:314-315设置普通 decoder 窗口；:354-364在HF prepare之后设置LS策略。
最终普通HF `MiniCPMODuplex.prepare/prefill/generate`注册边界与执行窗口。

### FC双工

`Demo py_backend/server.py:404-412`创建FC runtime、prepare后立即return，**不会经过普通set_duplex_config**。
`py_backend/fc_duplex_runtime.py:253-257`只构造system/TTS prompt/generate_audio的prepare请求；:259起读取FC已知配置。
`core/processors/unified.py:2226-2257`处理工具/standalone事件后调用model adapter；`core/fc_duplex/model_adapter.py:139-157`透传；`modeling/external_o5_facade.py:476-501`转外部HF `fc_duplex`方法。
初始化时普通/FC是不同capability与不同decoder（facade:154-160），共享底层model，不共享普通capability的方法调用。

**API风险（高）**：FC `config`目前是dict按已知key读取，未通过普通 `ExternalO5DuplexConfig` 的严格校验。给FC session额外传 `sliding_window_mode`、`basic_window_high_tokens`、`ls_decision_mode`、`speak_threshold`，源码路径既不应用也不明确拒绝。FC `session_config`回显只有pad协议、unit policy、non_spoken调度、decoding policy、profile（runtime:388-401），不回显这些普通选项；所以不是“明确回显为生效”，但存在静默接受造成误解的风险。
`FcDuplexPrepareRequest`自身extra=forbid不能解决这个问题，因为runtime只把挑选的三字段交给它，原config没有交给它。

**帧的静默忽略（高）**：`FcDuplexPrefillRequest.frame_list`（schema:254）、runtime:676的base64解码、View:2254的透传均存在；到 `HF fc/capability.py:742`执行 `del frame_list, sample_rate, max_slice_nums`。这是明确的入口接受但推理未用。实现在支持前至少应明确拒绝非空帧，并披露只支持音频FC；本轮只报告，不改代码。

## 4. Sliding window

### 4.1 当前为什么没有生效

FC decoder在 `HF fc/capability.py:181-185`独立构造。公共decoder初始化window config的mode=off（`utils.py:1192,1672`）。
FC prepare只feed system（capability:716-728），没有 `register_system_prompt`。
FC prefill没有 `register_unit_start`；FC finalize（:974-989）只补非语音终止、关闭slot、feed UNIT_END、记录自己的`units_info`。**它的units_info不是decoder._unit_history**。
对比普通：`modeling_minicpmo.py:3483/3497`注册system；:3592注册unit开始；:4228注册unit结束；:4235-4240显式调用context/kv_surgery/basic。
AST提取确认FC capability中没有任何window/system-unit注册调用。

Accel `api.py:95-99`确实给FC decoder装了与普通相同的窗口计算补丁；`duplex/canvas_window.py:97-113`仅包装已有enforce/重建方法，**不会创造调用时机或登记unit**。所以“attach了canvas window优化”不能证明FC窗口已工作。

### 4.2 Basic：可实现，但不能只补一个enforce调用

公共Basic按high/low触发、删除最早完整unit（`utils.py:2097-2137`），可复用原始cache/3D重建机制。但接入要补：

1. system前缀的真实KV长度和canvas origin登记；每个FC unit的完整input/spoken/non_spoken/closing span登记；只在finalize完成、batch已flush后的安全边界裁剪。
2. 保留FC FSM、未闭合think/tool_call缓冲、tool ID映射/已消费结果与TTS turn状态。FC wrapper可因budget跨unit延续：FSM `:183-186`和`:394-400`；capability非语音分支 `:934-958`；仅删除一个完整unit，并不等于删除了一个完整逻辑tool call/turn。
3. 明确定义活跃tool call的参数前缀、未完成think/spoken turn是否需要pin住。若窗口删掉正在生成的长tool-call前缀，Python parser可能还记得，但LLM KV不再能看到它，语法约束能强制闭合却不能保证内容继续正确。需要协议/质量政策，不能用“没报错”证明无损。
4. 将审计序列和物理KV裁剪分开。`output_ids`是完整历史，增量FSM symbolizer按它推进；直接同步截短这个list会破坏索引与跨unit状态。KV裁剪不应被当成清空业务历史。
5. 当前公共hybrid cache裁剪只作用于full-attention K/V，linear conv/recurrent state保留（`utils.py:176-178`）。所以它本来也不是“仅保留尾部重新完整推理”的等价物。FC复用时应延续并披露这个基础语义；accel不能暗中重置linear state。

与3D结合：`utils.py:2039-2073`检查track长度并在删除后重建THW；需在FC实际裁剪前后验证track、KV、cursor一致。共享代码是基础，缺少FC生命周期验证仍不能签收。

### 4.3 KV surgery：需要专门的FC布局定义

普通kv_surgery将**unit末端的一段连续assistant生成token**搬到previous区：`utils.py:2993-3037`通过 `gen_tokens长度 + gen_tail_length`反推出单一 `[gen_start,gen_end)`；system prefix/suffix/previous布局在`:2155-2188`定义。
FC一个unit同时有spoken和non_spoken，二者间有runtime slot符号，非语音还可带think/tool_call，结果在下个unit输入中。把两轨文本简单cat后用普通“末端连续段”公式，会把错误KV行搬走；只保留spoken，则丢失tool历史语义；只调用现有方法而不mark generation end会回退普通drop而不是预期的文本保留。
因此需要选定FC保留的结构化内容与多段span，或先只支持Basic。**原理上FC可使用KV压缩/搬运，不代表当前普通surgery算法可原样接到FC。** 当前没有FC prefix anchor，`utils.py:2904-2915`的ready门也不会通过。

## 5. Listen/speak：已有FC策略，和普通策略不完全相同

普通binary在 `HF modeling_minicpmo.py:4128-4146`调用 `StreamDecoder.speak_probability`（`utils.py:3459-3473`），对listen/speak两logit在T=1归一化，和可配置threshold比；生成turn未结束时锁speak。温度/top-k/top-p/repetition不参与该决策。API会话可传threshold等。

FC按 `decoding_policy`区分：

- `raise_protocol_error`：capability `_constrained_for:392-400`不施加FSM mask；`_sample:402-406`全词表argmax/temperature sampling。后续协议检查不等于采样前二选一。
- `constrained_decoding`：`_ConstrainedDecoding.step:93-118`根据FSM仅保留合法token，唯一合法token可强制。
- FSM `fc/o5_protocol_fsm/fsm.py:320-341`：turn已关闭时首步 `{LISTEN,SPEAK}`。这时greedy等价于二选一阈值0.5的选择（完全相同logit的tie规则另说）。sampling仍按temperature抽样，**不是**普通确定阈值判定。
- turn未结束：revised协议首步只许SPEAK；legacy协议首步是 `{SPEAK,TTS_PAD}`。pad此处是已有声学流的继续，并非listen。机械套用普通锁SPEAK会抹掉legacy允许的pad分支。
- non_spoken有独立mask（FSM:343-354），listen不表示不能thinking/tool-call；把“listen时不执行生成”的普通直觉搬过来会破坏FC。

结论：binary threshold概念可移植到**FSM允许listen/speak且turn已关闭的那个位置**；不能替换整个FC采样，也不能凭普通环境变量认为生效。若将来加阈值，应新增明确的FC会话策略并保留legacy/revised规则和greedy/sampling定义。目前FC没有该阈值API。

## 6. 3D-RoPE / merge-prefill

FC的system/audio/event/spoken/non_spoken都调用公共 `StreamDecoder.feed`。`utils.py:1653`按model config启用canvas，`:3278-3296`生成并记录 `[text,T,H,W]`；`mrope_canvas.py:667-707`对文本类token使T/H/W同值、沿token递增。FC声音embedding、tool/standalone文本和控制符都走这条通用分支。3D模型在这些非视觉token上使用相同三轴坐标是现有定义，不是因为FC把canvas关闭了。

FC图像没有prefill：其frame在capability:742丢弃。普通则有 `modeling_minicpmo.py:3601`图像处理和`:3832-3836` `feed_image_group`，公共decoder`:3303-3318`验证spec长度并生成图像THW。将来FC视频需要正确图像slot序列、image group/spec和SDK协议支持，不是单独设置mrope_mode即可。

Accel merge在 `api.py:95-99`对FC同样安装；`unit_feed_batch.py:55-78`每逻辑feed保留4轴positions和spec、请求logits立刻flush并只返回所请求feed的hidden切片；`:100-134`恢复原生逐feed ledger。`flush_cache_reads=True`在`:41-53`保证读cache前发布已排队feed。该结构与FC音频/event兼容，但不能凭它补出不存在的图像prefill或滑窗。

**数值边界**：完整merge本身不是逐位等价。`Accel docs/FULL_BATCH_NUMERICS.md`已经记录普通3D真实4unit中hidden relativeL2最高42.66%、logits22.99%，来源是linear recurrent/chunk舍入与GEMM/路由放大；用户允许的数学正确性政策与metadata审计是独立依据。39次FC fixed-feed exact门不包含merge自身，不能借它宣称FC“merge前后也逐位一致”。当前报告没有新数值测量。

## 7. TTS修复与共享边界

### 已接通、源码自洽的部分

1. **同一TTS及vocoder对象**：普通与FC都调用 `model.tts.generate_chunk`，随后 `model.tts.audio_tokenizer.stream`。FC引用 `capability.py:623-664`；普通 `modeling_minicpmo.py:4318-4351,4533-4565`。FC的tts KV/text_start_pos/flow/hift base cache由自身管理，prepare/reset负责会话边界，底层权重共享。
2. **normalization**：FC `_tts_condition:592-603`按TTS config选择 `_normalize_projected`，公共`:1657-1664`支持L2和LayerNorm；当前config是L2。普通双工condition `:4502-4505`仍直接L2；因此两者不能笼统说“所有checkpoint normalization都完全相同”，当前L2配置没有此冲突。不同checkpoint若选择LayerNorm，普通路径也需单独审查。
3. **causal vocoder修正**：FC不再预填3个旧4218 token（capability:572-580）；每unit可force-flush短块（:645-655）；turn结束时即使token buffer空，也调用 `last_chunk=True`（:656-664）后reset。迁移patch清楚列出，和普通causal尾部修复同方向。已有synthetic valid-ID尾部测试支持工具契约。
4. **hidden/token对应和graph buffer所有权**：FC先把采样tid feed，再将其hidden存到tts_results（:819-827），未在HF维护graph分支。Accel公共graph输出已有clone（如`llm/graphs/composed_parallel_graph.py:39,47`、`native_single_graph.py:31`，TTS `speech/tts_decode_graph.py:186`，T2W `speech/t2w_graph.py:85-86`）；compact feed直接返回该独立hidden。责任仍在accel，符合三物料边界。
5. **共享加速适用范围**：Accel `api.py:117-128`安装在shared TTS/vocoder对象；不是只hook普通capability。语音单元来自FC也会经过它们，算法并不依赖FC是否调用tool。

### 不能据现状直接宣布全部验证的部分

- **FA2并非默认已启用**：公共HF `init_tts_module:373-379`仅顶层FA2时用FA2，否则设eager。当前实测配置是sdpa；Accel `AccelerationConfig.tts_fa2_whole_decode`默认False（config:24）。显式True才由`speech_backend.py:16-25`切FA2并构建whole decode。共享加速可支持FC，不等于本次Docker native或默认TP已开启FA2修复。若目标是此前特定性能组合，要将实际配置/attention class/graph状态列入验收。
- **FC条件序列不是普通条件序列的原样复制**：普通 `_convert_results_to_tts_input:4473-4485`在turn结束加text_eos；FC `_tts_condition`固定merged+audio_bos，没有这个参数。FC `_spoken_audio`用end_of_turn控制min_new_tokens、reset与flush。二者可能由不同训练协议规定，不能为了“统一”擅自套普通EOS。
- **值得进一步核对的来源继承疑点（中风险，未证实bug）**：FC spoken生成注释`:821-825`称训练condition包含`spoken_turn_eos`和`tts_pad`，但实际append条件`:826`只收普通spoken文本或tts_pad，未收special `spoken_turn_eos`；该method和`_tts_condition`迁移manifest都标记AST原样，非本轮移植遗漏。当前API能返回语音不足以证明terminal-only/turn尾condition与训练严格一致。需要对应版本训练condition序列或SDK/模型团队真值做对照，不能仅根据注释判错，也不能据注释签收正确。
- **pad、空终止、跨unit、切窗组合未做完整门**：pad应产真实声学块；terminal-only条件仅BOS是否正确、force_flush/empty-final波形是否精确、长spoken turn跨Basic裁剪时TTS cache是否延续，都没有从目前短API证据得到覆盖。

## 8. 已有验证到底覆盖了什么

1. HF `fc/validation/evidence/20261013-initial-fc/README.md`：job1209233、两段静音音频、spoken4/nonspoken8、constrained greedy，39次相同embedding feed的native→TP hidden/logits逐位一致。短非视觉、无滑窗、无merge自身等价和音质结论。
2. Demo `docs/external-hf/ACCEPTANCE.md`：普通basic1024/512、binary24unit发生3次窗口；这是普通路径。普通/FC切换、allmerge46unit语音/think，不代表FC工具/滑窗齐全。board46 summary明确自主tool未触发。
3. 同报告的FC ckpt basic-1 8unit，真实模型生成tool-call，**工具返回为录制fixture回填**，有音频；不是实时联网工具评测。这里case名称basic-1与滑窗basic模式无关。
4. 当前Docker `analysis-runs/20261014-docker-lis/REPORT.md`：Pro6000 native single_eager，FC8unit、3模型调用、2个fixture结果已消费、5个speech PCM，有FC后chat回归。没有加速TP2、Pro5000性能、FC视觉、阈值或滑窗验收。
5. 当前simplex/mobile的ordinary音/视频各2unit均listen；只能证明选项/链路，不能验证TTS说话状态和实际滑窗触发。

因此当前没有“FC + basic/kv_surgery + 可调binary + 图像3D + TTS pad/尾部 + accel全开”的组合验收。

## 9. 风险排序与最小后续路径

### 优先避免误用（小改动，需root另行决定实施）

- FC runtime严格验证/拒绝未支持的普通窗口与LS配置，不应静默接受。
- 在FC未实现视觉前，非空frame_list应明确unsupported，避免无感盲看。
- 能力回显明确列出FC支持哪些输入/策略/窗口；不把普通默认页面参数套给FC。

### 按机制完成支持

1. **Basic先行**：HF FC capability补system/unit登记与安全finalize边界，明确active wrapper/tool/turn保留策略；复用公共decoder/canvas实现，accel仅优化算子。首先小窗口强制多次裁剪，覆盖跨unit工具调用、已发调用的迟到结果、持续spoken/pad、reset/resume、普通/FC交替。检查token/hidden与TTS条件、canvas ledger/KV/linear state、协议事件。
2. **LS可选阈值**：仅在FSM允许{LISTEN,SPEAK}处提供可显式选择的threshold策略，legacy继续保留speak/pad；验证不改变non_spoken调度。
3. **FC视觉**：先明确SDK图像slot训练合同，再复用ordinary图像处理和canvas spec；用有可辨识内容的图片/多slice证明模型确实消费帧，不能只检验无异常。
4. **KV surgery后行**：先定义FC结构化历史保留和多段KV span，不能直接套一个gen_tail长度。
5. **TTS针对性门**：检查对应训练版本对EOS/pad的condition定义；在eager和目标accel/FA2组合下对照每步hidden/token、condition、TTS ids、waveform和跨replay所有权，固定随机条件区分数值与采样差异。

这份review完成的是“指出调用链、现状、机制兼容范围和验证缺口”；未替代上述新功能实现或组合验收。没有新增进程、GPU、容器等遗留资源。三物料未由本review改动。

## 10. 固定上游来源二次核实

按root补充要求，直接从本地Git对象读取固定 `o5-fc-dev@732b075841f837d67d9f512835cbf1aec174911e` 的源文件（未拉远端），不只相信已有manifest。完整记录在 `UPSTREAM_PROVENANCE.json`，源文件SHA与migration manifest相同。

- 上游 `streaming_prefill`（`modeling/o5/modeling_minicpmo_unified.py:1281`起）与当前HF AST完全相同，源里就有 `del frame_list, sample_rate, max_slice_nums`，没有图像slot/prefill。
- 上游 `finalize_unit`（源:1517）与当前HF AST完全相同，源里也无window enforce/unit登记。
- 上游 `__init__`/`prepare`没有window、system/unit注册调用；迁移改动涉及session TTS cache/权重检查等已列具名适配，不存在把原上游FC窗口调用删掉的迹象。
- 上游 `streaming_spoken_generate`/`_tts_condition`与当前HF AST完全相同：LS约束策略/terminal-condition疑点均来源继承，不是新桥接时改写。

因此这两处最重要缺口（FC视觉帧被丢、FC不调滑窗）**来自抽取时固定上游本来就没有实现该能力**，并非三物料迁移误丢了o5-fc-dev原有支持。与普通路径合并到同一个HF目录，并不会让FC capability自动调用普通生命周期。

## Root审查之后的配置桥接更新

本报告的FC能力/迁移来源结论针对所列固定版本，保持原样。Root随后在Demo提交21d4410a补上O5_TTS_FA2_WHOLE_DECODE和O5_CANVAS_WINDOW_EXACT_CHAIN向AccelerationConfig的显式映射；未设ENV时默认仍False。CPU参数/优先级检查通过，实际双卡镜像API仍在进行。该配置修正不增加FC滑窗/视觉/阈值能力，也不把原先普通或native实验改写为这些选项已启用。

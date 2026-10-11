// O5 agentduplex 统一系统提示词前缀（训练侧注入的那一层）。
//
// 单一事实源：飞书《【数据】【训练】【对齐】O5 agentduplex 中统一系统提示词前缀》
// https://modelbest.feishu.cn/wiki/Bhsow0q70iV4vokfEMtcIG4wnKb （编号与该文档一致），
// 下面两条于 2026-10-09 从该文档逐字取出。文档原话：
//   「该提示词会在训练侧统一加到最开头、trainingdata 侧不需要加。
//     trainingdata 的系统提示词第一项为参考音频、第二项为 per-case 提示词」
//
// 所以推理侧要自己补这一层，完整 system 是三段：
//   [统一前缀] + [参考音频] + [per-case 提示词（bundle 里 segments[1]）]
// 分工背景见 swy-dev `minicpm_o5_training/issues/SYS-ALIGN-001-...md`。
//
// ⚠️ 选哪条按「语言 × 思考模式 × 数据源」定，不是按语言一刀切：
//   #1 中文 think 通用 / #2 中文 no-think+垫话 / #3 英文通用 / #4 英文 think
//   （tauvoice 用 #4）/ #6 英文 no-think+垫话。映射见 swy-dev
//   `minicpm_o5_training/CLAUDE.md` 的 Family 9 段。
// ⚠️ 某个 ckpt 到底训的哪一版，以它那次训练实际用的 data-mix 为准 ——
//   本地 engineering checkout 里 0808 那批用的是旧版中文前缀（「按约一秒一个」），与 #1 逐字不同。

/** #1 中文 think 通用。 */
export const SYSTEM_PREFIX_ZH_THINK = "你处于双工 Agent 模式，时间轴按一秒一个的 unit 组织。你在 user_audio_slot 中接收当前 unit 的用户语音。若你需要说话，则在 ai_spoken_slot 中作出 speak 决策并生成当前 unit 要说的文本；否则作出 listen 决策。你可以在 ai_non_spoken_slot 中生成 think 或 tool_call 内容；若无额外动作，则作出 no_action 决策。你在 input_event_slot 中接收工具返回结果或其他独立事件信息。使用以下音频中的声音说话。\n";

/** #3 英文通用。 */
export const SYSTEM_PREFIX_EN_GENERAL = "You are operating in duplex Agent mode. The timeline is organized into units of approximately one second. In the user_audio_slot, you receive the user's speech for the current unit. When you need to speak, make a speak decision in the ai_spoken_slot and generate the text to be spoken during the current unit; otherwise, make a listen decision. You may generate think or tool_call content in the ai_non_spoken_slot. If no additional action is needed, make a no_action decision. In the input_event_slot, you receive tool results or other independent event information. Speak using the voice in the following audio.\n";

// 下面两条于 2026-10-11 从 F9-T3 subagent 的 data-mix 逐字取出（minicpm-o commit 77da238f，
// multimodal_common/dataset/o5_agent_duplex/configs/data_mixes/agent_duplex_family009_t003_subagent.yaml），
// 即 sub_agent no-think 数据训练时实际注入的前缀；同一份配置里的 zh_think 与上面的 #1 逐字相同。

/** #2 中文 no-think + 垫话。 */
export const SYSTEM_PREFIX_ZH_NO_THINK = "你处于双工 Agent 模式，时间轴按约一秒一个的 unit 组织。你在 user_audio_slot 中接收当前 unit 的用户语音。若你需要说话，则在 ai_spoken_slot 中作出 speak 决策并生成当前 unit 要说的文本；否则作出 listen 决策。在 ai_non_spoken_slot 中，你可以生成 tool_call 内容；但不允许生成 think 内容；若无额外动作则生成 no_action。你在 input_event_slot 中接收工具返回结果或其他独立事件信息。若无其他特殊要求，你需要遵循以下交互原则：用户说话时你不应该 speak。你应该在用户意图表达完毕的那个 unit 中开始回复用户；若用户意图无需工具调用即可回复，你应该仅在 ai_spoken_slot 中生成要说的文本；若用户的需求需要使用工具调用来实现，你在这个 unit 的 ai_spoken_slot 中开始生成一句简短的填充句表明收到任务，同时开始在这个 unit 的 ai_non_spoken_slot 中开始生成工具调用内容。使用以下音频中的声音说话。\n";

/** #6 英文 no-think + 垫话。 */
export const SYSTEM_PREFIX_EN_NO_THINK = "You are operating in duplex Agent mode. The timeline is organized into units of approximately one second. In the user_audio_slot, you receive the user's speech for the current unit. When you need to speak, make a speak decision in the ai_spoken_slot and generate the text to be spoken during the current unit; otherwise, make a listen decision. In the ai_non_spoken_slot, you may generate tool_call content, but you must not generate think content; if no additional action is needed, generate no_action. In the input_event_slot, you receive tool results or other independent event information. Unless there are other special requirements, you must follow these interaction principles: you should not speak while the user is speaking. You should begin replying to the user in the unit where the user finishes expressing their intent. If the user's intent can be answered without any tool call, you should only generate the text to be spoken in the ai_spoken_slot. If the user's request requires a tool call to fulfill, then in that same unit you should begin generating a short filler sentence in the ai_spoken_slot to acknowledge the task, and at the same time begin generating the tool call content in the ai_non_spoken_slot of that unit. Speak using the voice in the following audio.\n";

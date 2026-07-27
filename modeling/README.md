# O45/O5 Modeling Packages

本目录只跟踪可审计的模型代码，不保存模型权重、tokenizer 或 processor 资产。

- `o45/`：MiniCPM-o 4.5 modeling 与 O45 FC Capability。
- `o5/`：MiniCPM-o 5 modeling、MoE、TP2/LLM Graph 相关实现。

部署 Profile 的 `model_path` 提供 config、tokenizer 和 processor 资产；仓库 commit
决定 modeling 代码版本。运行时不得通过外部目录的 `trust_remote_code` 加载可变代码。

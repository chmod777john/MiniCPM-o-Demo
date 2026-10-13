# FC checkpoint real API v4 acceptance (2026-10-13)

Job 1209574, A100 TP2, agent-dev. Source: HF 335bfdd, accel f06e7d7,
Demo 875a00ec (service launched before subsequent ordinary option changes).
Environment: `/user/weihongliang/environments/o5-3drope-fc-20261013/.venv`;
SDK 0.0.6, minicpmo-utils 1.0.7. Original checkpoint:
`/user/chenjinpeng/Checkpoints/job_1054182/checkpoints/iter_0001275/job_1054182_iter_0001275.pt`.

- Board 46 units completed and returned speech, but no tool call. This was not
  a tool protocol pass. Board policy is 30/15 non-spoken tokens.
- Replayed the historically successful recorded `basic-1` with original prompt,
  input audio and 45/25 policy. Only request protocol changed v3 -> v4
  (path audio materialized as WAV). New protocol commits use unit_index instead
  of the old input_id; probe assertion adapted, model output not injected.
- 8/8 units committed; three real model tool calls: get_client_location,
  amap_weather, show_card. Responses used actual returned IDs. First two were
  confirmed consumed by subsequent units. Third was sent at unit 7 and has no
  subsequent unit in this bounded probe, so consumption is not claimed.
- Nine speech chunks returned; zero API errors. Tool results are recorded
  local fixtures (show_card uses local acknowledgement); no real tool executed.
- Client unit elapsed range 0.232–1.229s includes transport. This does not prove
  long-context performance or quality. No claim that Board should generate tools.

Raw events/audio: `/user/weihongliang/analysis-runs/20261013-o5-three-materials/api-fc-ckpt/recorded-basic1-v4/`.
The first v4 attempt stopped because the old probe expected input_id in a
commit event; retained as separate evidence in recorded-basic1/.

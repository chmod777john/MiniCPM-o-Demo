# First real external-HF API gate

A100x2 TP agent-dev job1209341, 2026-10-13. Demo serving started at8582f65f, HF335bfdd, accel4769f7a. HF source e43c73a + FC source732b075. Runtime Python environments/o5-3drope-fc-20261013/.venv (SDK0.0.6/utils1.0.7). Raw evidence /user/weihongliang/analysis-runs/20261013-o5-three-materials/api-allmerge.

Original allmerge checkpoint iter300, ordinary slot true, all existing graphs/merge enabled. Existing gateway->worker->backend uses one process for accelTP. No original service replaced; this isolated task stopped after gates.

Sequence: ordinary audio+video4units -> FCv4 Board8units -> FCv4 full Board46units -> ordinary audio+video24units. All sessions completed, ordinary24 includes TTS; ordinary last run server wall p50=369.65ms/max1037.8ms. These are short-context cases; no new64K/128K performance claim. Client first output does not represent finalization.

FC46 returned audio and think, but no toolcall.done, so autonomous tool/result gate is NOT satisfied by this allmerge checkpoint. Next step original FCckpt. v4 proves transport/unit completion/base64 audio and mode lifecycle only so far.

No application relocation: existing O5 Demo/API/worker organization preserved. CPU code review discovered a missing types import for reference-audio chat, fixed later in4bfd42c7; that chat branch remains GPU-uncovered.

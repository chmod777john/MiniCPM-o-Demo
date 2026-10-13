# Ordinary external HF runtime options

A100 TP2 job1210050, Demo2783bb4a/HF335bfdd/accelf06e7d7,
allmerge iter300, SDK0.0.6/utils1.0.7 in project env. Original media:
`/user/weihongliang/deploy-runs/20261009-no-fc-3d-rope/probe/media/normalized_00.mp4`.

24/24 units return with no errors. Server confirms basic1024/512,
LS binary (fix_enabled=true, threshold0.3), max speak8; KV sequence shows
three drops (947->433,955->441,968->436). Server wall p50=727.15ms,
max=1207.3ms. This is short-window functionality, not 128K acceptance.
Next session omits config entirely: logged window off, legacy LS, max speak20.
Thus session defaults reset instead of carrying prior session overrides.
No ordinary unit-complete event exists; first-output/metrics scope is in JSON.

The old retained probe included obsolete force_ctrl/main_veto_threshold/strategy_hd
keys from integration. New strict external options reject those; the local probe
removes them, and no longer silently claims they are effective. Retained actual
pages do not use these three fields.

Separate optional chat probe found retained ChatView inferred utils from facade
module name, which is not an HF package; model now explicitly supplies canonical
TTSSamplingParams. Its GPU follow-up remains pending until separately recorded.

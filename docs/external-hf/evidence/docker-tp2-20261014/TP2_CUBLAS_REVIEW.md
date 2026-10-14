# TP2 grouped cuBLAS编译失败：类型兼容与reduction语义

Reviewer：simplex_final_review；2026-10-14 UTC。只读日志、冻结runtime tar与本地PyTorch源码；没有改源码、编译扩展、SSH或GPU。

## 可立即实施的结论

1. TP2失败是C++接口类型变化，不是GEMM数值错误或CUDA驱动问题：`streamed_blas_grouped.cpp:61` 对 `at::CuBLASReductionOption` 使用 `!`，编译器拒绝。
2. 最小语义保持修复：写一个**按实际返回类型分派**的小header/helper。旧bool仍用`!value`；enum必须明确处理3个命名枚举，前2个映射原有math flag，禁止split-K的第3个明确报错（经典cuBLAS接口无法兑现）。不得用`static_cast<bool>`，不得仅用`!=Allow...`吞掉第3种。
3. 一共有4个调用点需要统一：streamed/direct/batched旧bool代码，以及parallel_streamed现有enum比较代码。parallel虽能在新Torch编译，但缺split-K拒绝，而且无法在旧bool头文件编译。
4. 不要用“版本>=2.14”判断：实际共享high-cu128的Torch2.11 Context.h也已经是enum；本地PyTorch v2.11.0 tag同样如此。按返回类型消除版本猜测。

## 原始证据

- `tp2-v7-full.log:21-29`：报错`no match for operator! (operand type is at::CuBLASReductionOption)`。
- 调用路径：accel attach/warmup → `parallel_experts.project` → `exact_grouped` → `direct_blas_grouped.py` → `streamed_blas_grouped.py` → JIT当前`.cpp`。native single没有走到它，所以此前native API成功与此不矛盾。
- 冻结运行包：`runtime/pro5000-python310-site.tar.gz`，SHA `6fd80ea8714a6f2e7a9f467bfd3c343b961a35cb846d7b7ddb3d94672ff99ec0`。
  - `lib/python3.10/site-packages/torch/include/ATen/Context.h:44-48`：enum值0/1/2分别为下表三种。
  - 同文件`:399-402`：getter返回enum，setter接收`bool allow_reduced_precision, bool allow_splitk=true`。
  - 同文件`:507-508`：默认AllowReducedPrecisionWithSplitK。
  - `torch/backends/cuda/__init__.py`：Python bool赋值转成`(value, True)`，tuple/list可设置split-K；getter分别暴露reduction及split_k布尔值。
- 本地完整PyTorch repo `/user/weihongliang/pytorch` **包含当前runtime精确commit** `08187d9e0fba026dc8217405802ab5381dc88d90`；已`git show`直接核对该commit的 `aten/src/ATen/cuda/CUDABlas.cpp:1125-1181` 和 `Context.cpp:681-703`。
- 精确commit的BF16经典cuBLAS helper在调用GemmEx之前明确拒绝 `DisallowReducedPrecisionDisallowSplitK`；其次用`!=AllowReducedPrecisionWithSplitK`决定`CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`。这就是本次适配应沿用的语义（原生报错文案误写fp16，适配自身可正确写bf16）。
- 旧bool接口可从本地PyTorch commit `37c6087334c^` 的 `aten/src/ATen/Context.h` 直接核对；`37c6087334c`引入split-K控制。v2.11.0源码及当前high-cu128 installed头文件均已enum，不能误称这是2.14才新增。

## 数学/flag映射

|实际接口值|映射|原因|
|---|---|---|
|旧bool true|`CUBLAS_DEFAULT_MATH`|允许reduced precision reduction；与旧代码一致|
|旧bool false|DEFAULT OR DISALLOW_REDUCED_PRECISION_REDUCTION|禁止reduced reduction；与旧代码一致|
|enum AllowReducedPrecisionWithSplitK (0)|DEFAULT|与Python设置true一致|
|enum DisallowReducedPrecisionAllowSplitK (1)|DEFAULT OR DISALLOW_REDUCED_PRECISION_REDUCTION|与Python设置false一致|
|enum DisallowReducedPrecisionDisallowSplitK (2)|明确TORCH_CHECK失败|classic GemmEx不能声明split-K禁用；PyTorch原生classic路径也拒绝|
|未来未知enum|明确报错|不能静默改变reduction政策|

`static_cast<bool>(enum)`是错误修法：Allow枚举是0（false），Disallow枚举是1/2（true），会反转原有逻辑。单纯`enum!=Allow`虽修当前默认编译，但会静默忽略split-K=false，所以也不完整。

## 推荐最小实现形状（不改算子）

```cpp
template<class T>
inline bool disallow_bf16_reduction(T option) {
  if constexpr (std::is_same_v<T, bool>) {
    return !option;
  } else {
    static_assert(std::is_enum_v<T>, "Unsupported cuBLAS reduction setting type");
    if (option == T::AllowReducedPrecisionWithSplitK) return false;
    if (option == T::DisallowReducedPrecisionAllowSplitK) return true;
    TORCH_CHECK(option != T::DisallowReducedPrecisionDisallowSplitK,
                "BF16 allow_splitk=False requires cuBLASLt; classic cuBLAS grouped path cannot honor it");
    TORCH_CHECK(false, "Unknown BF16 cuBLAS reduction option");
  }
}
```

关键是`T::枚举名`必须是依赖名并位于template中；旧头文件不存在`at::CuBLASReductionOption`时也能编译。普通非template函数里`if constexpr`包住未声明的`at::CuBLASReductionOption`仍会触发名字解析失败。公共header显式include `<type_traits>`，使用既有TORCH_CHECK。

四个调用点保持现有`cublasMath_t mode=DEFAULT`和OR逻辑，只换判定函数。shared helper不得改变：

- GemmEx / GemmBatchedEx选择及compute type；
- alpha/beta、转置、layout/stride、M/N/K、专家分片或offset；
- stream数、提交次序、归约或moe routing；
- caller-ownedhandle的math/pointer保存恢复；
- Python profile、选用streamed/parallel方式及加速开关。

当前`acceleration.py:47-48`设置reduction选项，默认bool true，在新runtime中就是enum0；正确修复后的math flag仍是DEFAULT。因此此修复本身不构成新数值优化或数学近似。编译成功仍需真实GPU验收才能声称工作正常。

## 建议root验证门

1. CPU小编译覆盖旧bool true/false、新enum三态/未知值；确认不依赖不存在的enum名字。
2. 当前Torch2.14真实header编译streamed/direct/parallel（batched实验路径至少编译或CPU覆盖），记录未变profile。
3. 小矩阵GPU对照原生BF16 matmul，在Python reduction=true/false下分别核对；explicit split-K=false应报错，不能fallback。
4. 再跑真实TP2镜像API；如数值测试不完全一致，区分已有kernel调用方式差异与本次flag修复，不用native single通过替代TP验收。
5. 修复归属accel隔离wt，manifest标新accel SHA；无需改HF/Demo推理。

收尾：review已完成；无源码修改、运行资源或后台任务。

## root实现66c4d07复核

已只读复核隔离accel `...-o5fc-cu130-20261014` commit `66c4d070e67b8fd091fa1549499c12cb7c99873b`。

结论：**与上述精确Torch语义一致，无阻断项；可以进入计划中的GPU小矩阵与真实TP2验收。**

- 公共header按实际Option类型做template/if constexpr，bool使用原`!option`；enum分支使用依赖名，不依赖旧版不存在的`at::CuBLASReductionOption`。
- enum0返回false，enum1返回true；enum2明确要求cuBLASLt，未知值同样抛错；未使用PEDANTIC、数值cast或静默fallback。
- streamed/direct/batched/parallel四调用点仅替换判定并include同一header。math默认值、DISALLOW flag、GemmEx参数、stream提交与handle恢复均未改；parallel仍在Job构造时保存该mode。
- pyproject package-data已增加`*.h`，避免wheel漏公共header；当前docker runtime git archive也会包含src下该header。
- CPU测试编译真实c10头/链接libc10，分成没有enum声明的bool单元和带独立enum的单元；覆盖bool双态、enum前两态、split-K与未知拒绝，内容符合本次契约。root报告PASS，本review未重复执行。
- CPU gate证明host policy与兼容写法；不等同四个GPU扩展编译或数值验收，后续由打包agent承担。现有测试使用同名独立enum刻意检查类型分派，真实at::enum匹配由当前Torch头的GPU扩展编译确认。

本次复核无代码改动、无GPU或新增测试运行。

## v9编译、GPU小矩阵、链接与manifest独立复核

2026-10-14 UTC。只读复核 `ROOT_CUBLAS_HISTORY.md`、两份grouped gate脚本与v8/v9日志、`release-input-manifest-v9.json`、`verify_final_v9.py`、file-hash log、resolved acceleration和打包Git `1247d4d3/2c07b81c`。无GPU/SSH/Docker/源码修改。

结论：**未发现必须立即中断API的缺陷；支持root合入1247d4d3/2c07b81c。** 本轮下列门成立，但不要扩大成全模型精度或真Pro5000硬件验收。

### 已证明的编译与数值范围

- `grouped_compile_gate.py`调用实际四个loader：direct、streamed、parallel、batched_candidate。v9 log均有编译/链接/加载PASS；direct `.o`由v8相同源码编译完成，v9补链接成功，其余v9完整compile+link，未见绕开失败的fallback。
- `grouped_gpu_gate.py`在cuda0/cuda1各自创建BF16输入`a[33,64]`、`b[4,64,48]`（由`[4,48,64].transpose`得到），expert分段长度2/0/14/17，包含空expert；随机种子7。
- reference是同一Python reduction policy下的逐expert `torch.mm`；direct/streamed/parallel各自true/false，共12次`torch.equal=True`、`max_abs=0`。streamed/parallel传4 lanes。小矩阵的跨device输入是各自生成，不是模型TP通信测试；恰好适合局部扩展语义验收。
- `(False,False)`在3实现×2卡共6次调用中明确抛预期split-K错误；没有静默切换或执行不受支持policy。
- batched仅编译/load，没有执行数值门且没有启用。不能据此重新宣布该已舍弃候选精度通过。
- **限制**：K=64的这一组矩阵不能迫使cuBLAS一定选择不同reduction算法，也没有直接查询运行中的math flag。因此“实际两policy输出均与torch.mm一致”成立；“覆盖所有K/shape的reduction舍入效果”不成立。flag映射的保证来自已审核helper源码+CPU真假/enum门，GPU门补充实际构建/调用正确性。
- 脚本当前original policy记录True，finally恢复True正确；它只保存reduction bool，没有保存split_k独立值，因此不应称为任意外部policy状态的通用fixture。对本次默认True独立测试进程没有影响，未来如复用到同进程状态测试，可一起保存split_k。
- 此GPU gate不独立检查math/pointer handle是否恢复；该代码本次未改，仍以源码RAII审查为依据。

### 2c07b81c是否替换了数学库

- 修复只创建CUDA_HOME/lib64中两个未版本化开发链接名：`libcublas.so`和`libcublasLt.so`，目标分别是原已冻结venv下`nvidia/cu13/lib/libcublas.so.13`与`libcublasLt.so.13`。
- 没有pip安装/下载新cuBLAS、复制其他库、替换Torch、修改GemmEx算法或compute type。v8错误明确`cannot find -lcublas`，v9相同链接参数增加可解析开发名后成功。因此源码层面是让linker使用现有相同冻结运行库，不是换库规避。
- `cublas-library-paths.txt`记录目标文件存在；当前file hash gate验证源码/package/mobile，不验证这两个ELF或`/proc/self/maps`。**没有进程实际映射路径/库hash独立产物**，故只能说“链接目标明确指向既有冻结库，构建和实际GPU调用通过”，不要写成“已独立核验运行时ELF逐字节身份”。这不阻断当前修复；若最终交付要证明实际同库，可在packager已有容器用readlink+sha256或进程maps留证，reviewer未自行占资源。

### manifest真实性与覆盖

- 已把manifest列出的 **package12文件、application overlay2文件、accel overlay7文件**逐一与标明Git commit的文件SHA256比对，全都0 mismatch；package tree hash也与files map重算一致。
- `application_sources`保留HF5cccbf7、accel6e33931、Demo08c22855；`application_refresh`明确列core/deploy/modes.py与test，来源2c07b81c；`acceleration_refresh`明确列66c4d07兼容补丁。这是准确的base+overlay描述，没有把base冒充最终完整checkout。
- overlay工程的1247d4d3只新增显式accel Git tar解包；2c07b81c新增同库链接。未把HF建模代码复制入Demo，也没有改HF源码。
- `verify_final_v9.py`的实际PASS覆盖：两个mobile输出树完整hash、manifest列出的package/app/accel文件hash。它不重验所有继承HF/Demo/accel文件，不核对runtime ELF/ckpt，也不拒绝package额外未列文件。全量继承源码准确性仍依赖base镜像身份与上一轮tar/blob核对，证据口径应保留这个层次。
- release输入manifest的base记录local image ID与registry digest；v9实际build log使用本地tag v6，未以digest传给FROM，所以这是有ID记录的本次构建，不能说recipe已经强制digest pin。最终v9输出ID `sha256:93aa6ee9df9abc96669d09235bcbd499be6a369c4a4fa63ec3507e3402afaaf2`与build log、image-v9-identity一致。
- v9本轮没有新的registry发布声明，不能沿用v6 registry digest作为v9最终镜像。

### resolved acceleration及API边界

- `tp2-v9-resolved-acceleration.json`与startup-full.log engine逐项一致：hardware=pro5000、mode=tp、cuda0/1；LLM/TTS/T2W graph、merge_unit_feed、compact_logits、canvas exact chain、TTS whole FA2均true。
- 这份信息来自`accel.attach`后`model._acceleration_info`，不是简单打印ENV；LLM warmup、TTS16 captures及rng_exact都有实际初始化记录。证明已安装/初始化，不能据此断言每个未来请求都命中graph而无形状fallback。
- startup log模型加载成功、worker注册200；image identity明确两卡物理型号Pro6000。当前口径正确：**Pro5000 preset在双Pro6000上的镜像运行**。
- review时已有文字/语音stream两项和ordinary音视频各2unit产物，后续FC/完整API由packager继续。此处不替代其最终API报告，更不把模块12次exact外推为端对端hidden逐位对齐。

收尾：本轮review完成，无紧急阻断消息、无源码/资源操作。结论可直接用于root收尾合入；少量证据补强建议均不要求中断正在运行的API。

# CodeReason-PT：基于可执行验证信号的代码推理大模型后训练

**English:** CodeReason-PT: Verifier-Driven Post-Training for Code Reasoning

**Project Type:** LLM Post-training / Code Reasoning / Preference Optimization / RLVR

**Target Roles:** LLM 算法工程师、Post-training 算法实习、大模型训练算法工程师

# 0. 项目原则

本项目不是为了展示“我跑过 SFT、DPO、GRPO”，而围绕唯一核心对象：

[
\boxed{\text{Verifier-derived Training Signal}}
]

研究同一个代码执行 Verifier 产生的 correctness feedback，在转化成：

1. **Offline Preference Signal**
2. **Online RL Reward**

时，其：

[
\boxed{
\text{Preference Hardness}
+
\text{Reward Granularity}
+
\text{Signal Reliability}
}
]

如何影响模型训练行为与跨题泛化。

项目最终固定为：

[
\boxed{
Base
\rightarrow
Reasoning\ SFT
\rightarrow
\begin{cases}
DPO_{Easy}\
DPO_{Hard}\
GRPO_{Binary}\
GRPO_{Dense}
\end{cases}
\rightarrow
Independent\ Evaluation
}
]

DPO 和 GRPO 为**并行实验分支**。

默认不执行：

[
SFT\rightarrow DPO\rightarrow GRPO
]

除非全部主实验完成后仍有充足算力，再将：

[
SFT\rightarrow DPO_H\rightarrow GRPO_D
]

作为附加 warm-start 实验。

# 1. 三个核心研究问题

## RQ1：Preference Hardness

固定：

[
(x,y^+)
]

只改变 rejected response：

[
y_E^- \quad vs \quad y_H^-
]

比较：

[
DPO_E \quad vs \quad DPO_H
]

研究：

> large-margin、明显错误的 preference 与 small-margin、接近正确边界的 preference，对 DPO 训练行为和最终代码正确率有什么不同？

不预设：

[
DPO_H>DPO_E
]

任何结果都以真实实验为准。

## RQ2：Reward Granularity

固定：

- SFT checkpoint；
- GRPO prompt pool；
- rollout budget；
- optimization recipe。

只改变 reward：

### Binary

[
R_B(y)=
\mathbb{1}(\text{all reward tests passed})
]

### Dense

[
R_D(y)=
\frac{N_{\text{passed reward tests}}}
{N_{\text{reward tests}}}
]

比较：

[
GRPO_B \quad vs \quad GRPO_D
]

研究 sparse outcome signal 与 higher-resolution executable feedback 对：

- reward variance；
- zero-variance group；
- convergence；
- final Pass@1；
- hidden-test correctness；

的影响。

不预设 Dense 一定优于 Binary。

## RQ3：Verifier Signal Reliability

对于同一道 Post-training problem：

[
T_i=T_i^{reward}\cup T_i^{heldout}
]

并满足：

[
T_i^{reward}\cap T_i^{heldout}=\varnothing
]

Preference construction 和 GRPO reward **只能访问**：

[
T^{reward}
]

而：

[
T^{heldout}
]

永远只由独立 evaluator 使用。

研究：

[
R_{reward}\uparrow
]

时：

[
R_{heldout}
]

是否同步改善。

这里默认使用术语：

> **Verifier Generalization Gap**

而不是直接声称：

> verifier overfitting。

只有当该 gap 在多个 checkpoint / prompt，最好多个 seed 上持续扩大时，才描述为：

> evidence of verifier overoptimization。

# 2. Backbone

默认：

[
\boxed{\text{Qwen2.5-Coder-3B Base}}
]

若实际硬件不足：

[
\boxed{\text{Qwen2.5-Coder-1.5B Base}}
]

模型规模不是研究变量。

原则：

> 保证完整实验矩阵优先于模型参数规模。

# 3. Prompt Serialization

由于使用 Base model，不依赖已有 Instruct chat template。

全项目固定一种 causal-LM serialization，例如：

```
### Problem
{problem_statement}

### Solution
{reasoning_and_code}
```

推理时：

```
### Problem
{problem_statement}

### Solution
```

SFT 时：

- `Problem` 部分 mask；
- 只对 `Solution` response token 计算 loss。

所有 Base/SFT/DPO/GRPO evaluation 必须使用相同 serialization。

除非经过正式版本升级，否则禁止中途修改 template。

# 4. 数据划分

数据分成：

[
D_{SFT},D_{PT},D_{Dev},D_{Eval}
]

建议规模：

| Dataset       | Target           |
| ------------- | ---------------- |
| SFT           | 8K–15K problems  |
| PT            | 1K–2K problems   |
| Dev           | 300–500 problems |
| External Eval | LiveCodeBench    |

若只能获得 5K–8K 个高质量可执行映射样本：

> 接受较少数据，不使用模糊匹配强行扩充。

# 5. 数据来源

## SFT Reasoning

优先：

> OpenCodeReasoning-2

## Problem / Testcase

优先：

- TACO
- APPS

最终样本必须形成：

[
(problem,\ reasoning,\ reference\ code,\ executable\ tests)
]

只允许：

- source ID；
- question ID；
- 可验证 metadata；

等高置信方式完成映射。

**禁止使用 embedding / fuzzy semantic join 把无法确认的数据强行拼接。**

无法可靠映射：

> drop。

# 6. 数据隔离与 Decontamination

必须同时做：

### 6.1 SFT ↔ PT

保证：

[
D_{SFT}\cap D_{PT}=\varnothing
]

包括：

1. exact problem ID；
2. normalized text hash；
3. n-gram / MinHash near duplicate。

### 6.2 Train ↔ External Eval

同样执行：

- exact normalized match；
- near duplicate；
- source / contest / URL / ID overlap。

必须输出真实报告，例如：

```
SFT raw:                     9,482
PT raw:                      1,634
Exact SFT/PT overlap:           21
Near-duplicate overlap:         37
Final SFT:                   9,424
Final PT:                    1,576
```

未经证明严格日期 cutoff 前，External Eval 统一称：

> **Decontaminated LiveCodeBench Evaluation**

不要称 Temporal Holdout。

只有真正能证明：

[
Date(eval)>Date(all\ training\ problems)
]

时才升级这个 claim。

# 7. 数据 Schema

统一以 JSONL / Parquet 保存。

## Problem Schema

```
problem_id
source
source_id
difficulty

prompt
reasoning
reference_code

reward_tests
heldout_tests

split
```

## Rollout Schema

```
problem_id
checkpoint
sampling_seed

temperature
top_p
max_new_tokens

response
generated_reasoning
generated_code

status
compile_success
runtime_success
timeout

reward_passed
reward_total
reward_pass_rate

heldout_passed
heldout_total
heldout_pass_rate

response_tokens
code_tokens
latency
```

## Preference Schema

```
problem_id

chosen
easy_rejected
hard_rejected

chosen_reward
easy_reward
hard_reward

easy_margin
hard_margin

chosen_tokens
easy_tokens
hard_tokens
```

所有实验配置必须额外保存：

```
run_id
git_commit
config_file
model_checkpoint
random_seed
gpu_info
start_time
end_time
```

# 8. Reward / Held-out Test Split

这是 P0，而不是可选增强。

用于核心 PT 实验的问题必须满足：

[
|T|\ge T_{min}
]

初始：

[
T_{min}=10
]

根据数据实际分布可修改。

推荐：

[
T^{reward}^{heldout}=70:30
]

或：

[
80:20
]

一次 split 后持久化，固定 seed。

**GRPO reward callback 禁止加载 held-out tests。**

**Preference builder 禁止读取 held-out results。**

Held-out evaluation 必须作为独立脚本运行，避免训练代码无意中访问 hidden signal。

# 9. Executable Verifier

核心 API：

```
verify(response, tests) -> VerificationResult
```

流程：

[
Response
\rightarrow
ExtractCode
\rightarrow
SyntaxCheck
\rightarrow
Execute
\rightarrow
Judge
]

结构化状态：

```
CE
RE
TLE
WA
AC
```

Python 可使用：

```
ast.parse / py_compile
→ isolated execution
→ output comparison
```

返回：

```
status
compile_success
runtime_success
timeout
passed
total
pass_rate
exit_code
runtime_ms
stdout_size
stderr_size
```

# 10. Sandbox

第一版只做 RL 必须的可靠隔离：

- wall-time；
- CPU；
- memory；
- process count；
- network disabled；
- restricted filesystem；
- output size；
- deterministic image/environment。

优先：

> Docker/container worker。

不研究：

- namespace 创新；
- sandbox benchmark；
- 极致安全机制；
- 极限吞吐。

只有 profiling 证明 container startup 是主要瓶颈后，再考虑 persistent workers。

# 11. Verifier 验收 Gate

训练开始前必须运行：

```
python -m verifier.validate_reference_solutions ...
```

在 reference solutions 上要求：

[
AcceptanceRate\approx100%
]

不能轻易把官方 solution 的 WA/TLE 归因于数据。

需要人工检查：

- 输入格式；
- output parser；
- testcase；
- timeout；
- Python version；
- runner。

Verifier 未通过验收：

[
\boxed{\text{禁止进入 DPO / GRPO}}
]

# 12. Verifier Reliability Metrics

## Generalization Gap

[
G_V=
R_{reward}-R_{heldout}
]

## Verifier False-positive Rate

[
\boxed{
FPR_V=
P(R_{heldout}<1\mid R_{reward}=1)
}
]

它衡量：

> reward verifier 判定为完全正确的 solution 中，有多少无法通过 unseen tests。

注意：

FPR 只作为 verifier limitation / signal noise 指标。

不要将一个单独 FPR 数值直接解释成：

> “训练数据有同等比例错误”。

# 13. Phase 0：Base Baseline

任何训练前，先固定 baseline。

输出：

```
results/base/
```

记录：

- Dev Pass@1；
- executable rate；
- mean test pass rate；
- avg output tokens；
- generation latency。

同时冻结：

```
eval_config.yaml
```

以后所有六个模型必须遵守同一 evaluation protocol。

# 14. Phase 1：Reasoning SFT

训练：

[
Base\rightarrow SFT
]

目标：

> 建立可用于 Preference / RLVR 的 reasoning cold-start policy。

Loss：

-\sum_{t\in response}
\log P_\theta(y_t|x,y_{<t})
]

prompt token：

[
label=-100
]

# 15. SFT 初始 Recipe

默认起点：

```
Backbone: Qwen2.5-Coder-3B Base
Samples: actual high-confidence mapped set
Max length: 4096
LoRA rank: 16 or 32
LoRA dropout: 0.05
Epoch: 1
LR: 1e-4 ~ 2e-4
Warmup: 3%
Optimizer: AdamW
Precision: bf16
Gradient checkpointing: enabled if needed
```

这些只是 initial recipe。

不做 LoRA rank ablation。

# 16. SFT 验收 Gate

必须重新跑：

```
Base
vs
SFT
```

检查：

- train loss 正常；
- generation 格式稳定；
- executable rate；
- Dev Pass@1；
- mean test pass；
- avg response length。

若 SFT 后明显异常：

> 不进入后续训练。

优先排查：

1. prompt serialization；
2. assistant loss mask；
3. EOS；
4. tokenizer；
5. code extraction；
6. LR；
7. sequence truncation；
8. 数据 mapping。

# 17. Phase 2：Policy-Sampled Preference Data

正式名称：

> **Policy-Sampled Preference Construction**

不是 Online DPO。

流程：

[
\pi_{SFT}
\rightarrow
K\text{-way sampling}
\rightarrow
Verifier(T^{reward})
\rightarrow
Preference
]

初始：

```
K = 8
temperature = 0.8
top_p = 0.95
```

资源有限：

```
K = 4
```

Hard negative 太少：

```
K = 16
```

不要首先修改 hard threshold。

# 18. Preference Pair 定义

## Chosen

[
R_{reward}(y^+)=1
]

## Easy Negative

必须：

[
Executable(y_E^-)=1
]

并：

[
0\le R_{reward}(y_E^-)\le\tau_E
]

初始：

[
\tau_E=0.2
]

## Hard Negative

必须：

[
Executable(y_H^-)=1
]

并：

[
\tau_H\le R_{reward}(y_H^-)<1
]

初始：

[
\tau_H=0.5
]

CE / RE / TLE 不进入核心 Easy/Hard comparison。

保留到 failure analysis。

# 19. Matched DPO Dataset

核心要求：

同一 problem 尽可能存在：

[
(x,y^+,y_E^-,y_H^-)
]

这样：

### DPO-E

[
(x,y^+,y_E^-)
]

### DPO-H

[
(x,y^+,y_H^-)
]

固定：

- same prompt IDs；
- same chosen；
- same number of pairs；
- same training steps；
- same LR；
- same beta；
- same seed；
- same LoRA config。

主要实验变量：

[
\boxed{Preference\ Margin}
]

# 20. Preference Margin

定义：

R(y^+)-R(y^-)
]

输出：

```
results/preference/easy_margin_distribution.*
results/preference/hard_margin_distribution.*
```

同时记录：

- chosen/easy/hard length；
- difficulty；
- source；
- reward distribution。

# 21. Length Confound

Easy/Hard 两组必须进行 length matching。

至少保证：

[
P_E(L)\approx P_H(L)
]

可使用：

- length bucket matching；
- nearest length matching；
- filtering。

不要求完全相同。

但若 Hard completion 普遍比 Easy 长很多，则 DPO comparison 不能直接归因于 hardness。

必须在结果中报告 response-length distributions。

# 22. DPO From Scratch

必须实现：

```
dpo/loss_from_scratch.py
```

提供：

```
sequence_logprob()
implicit_reward()
dpo_loss()
```

处理：

- prompt/response concat；
- shifted logits；
- response mask；
- padding；
- chosen/rejected；
- frozen reference policy。

Sequence logprob：

\sum_{t\in response}
\log\pi_\theta(y_t|x,y_{<t})
]

DPO：

## \log\pi_\theta(y^+|x)

\log\pi_\theta(y^-|x)
]

## \log\pi_{ref}(y^+|x)

\log\pi_{ref}(y^-|x)
]

-\log\sigma[
\beta(\Delta_\theta-\Delta_{ref})
]
]

# 23. DPO Numerical Validation

构造固定 toy batch：

```
python -m dpo.validate_loss
```

比较：

```
custom implementation
vs
TRL
```

要求：

[
|L_{ours}-L_{framework}|<\epsilon
]

如果不一致，禁止进入正式 DPO training。

# 24. DPO Preference Accuracy

定义 implicit reward：

\log\pi_{ref}(y|x)
]
]

Preference Accuracy：

P[
r_\theta(x,y^+)

> 

r_\theta(x,y^-)
]
}
]

不要使用：

[
P[\log\pi(y^+)>\log\pi(y^-)]
]

冒充 DPO preference accuracy。

# 25. DPO Metrics

必须保存：

- training loss；
- chosen implicit reward；
- rejected implicit reward；
- implicit reward margin；
- preference accuracy；
- response length；
- Dev Pass@1；
- reward-test score；
- held-out score；
- external benchmark score。

重点分析：

[
Preference\ Fitting
\stackrel{?}{\Longrightarrow}
Executable\ Correctness
]

# 26. Phase 3：GRPO

两个 branch 都从：

[
\boxed{\pi_{SFT}}
]

开始。

不从 DPO 初始化。

每组：

[
G=8
]

资源有限：

[
G=4
]

# 27. GRPO Reward

## Binary

\mathbb 1(\text{all reward tests pass})
]

## Dense

\frac{N_{passed}}
{N_{reward\ tests}}
]

CE / RE / TLE：

[
R=0
]

不加入：

- format reward；
- compile reward；
- arbitrary weights；
- LLM judge。

确保唯一核心变量是：

[
\boxed{Reward\ Resolution}
]

# 28. Dense Reward Limitation

README 必须主动说明：

[
99/100\ test\ cases
]

不等于：

> 算法在语义空间中“99% 正确”。

可能唯一失败 case 对应：

- complexity；
- overflow；
- critical boundary；
- fundamentally wrong algorithm。

因此 Dense reward 定位为：

> **higher-resolution executable proxy**

而不是：

> semantic distance。

# 29. GRPO Prompt Pool

不提前删除所有：

- all-correct；
- all-wrong；

问题。

利用 SFT initial rollout success rate：

[
p_i=
\frac{#success}{K}
]

按数据分位点划分：

- Easy；
- Medium；
- Hard。

GRPO 训练使用 stratified sampling。

这样才能真实观察训练过程中：

[
Z^+,Z^-
]

如何变化。

# 30. Group-relative Advantage

教学 / 验收脚本：

```
grpo/advantage_toy.py
```

计算：

\frac{r_i-\bar r}
{\sigma_r+\epsilon}
]

测试：

```
[0,0,1,0]
[1,1,1,1]
[0,0,0,0]
```

README 明确：

> group-relative normalization 只解释 advantage signal construction，不是完整 GRPO objective。

面试知识必须额外理解：

- old/current policy；
- probability ratio；
- clipping；
- KL；
- loss aggregation。

# 31. GRPO Diagnostics

P0 指标：

[
MeanReward
]

[
RewardStd
]

[
KL
]

[
ResponseLength
]

[
ValidationPass@1
]

若框架直接支持：

[
ClipFraction
]

顺手记录。

# 32. Zero-Variance Metrics

总体：

[
Z=
\frac{
N[\operatorname{Var}(r)=0]
}{
N_{groups}
}
]

拆分：

### All-correct

\frac{
N[r_1=\cdots=r_G=1]
}{
N_{groups}
}
]

### All-wrong

\frac{
N[r_1=\cdots=r_G=0]
}{
N_{groups}
}
]

解释：

### (Z^+\uparrow)

可能意味着：

> prompt 对当前 policy 已经太容易，rollout compute 的有效性下降。

### (Z^-\uparrow)

可能意味着：

> 当前 policy 很难探索到 successful trajectory。

这些只作为 observation。

项目不继续加入 curriculum / dynamic sampling 实验。

# 33. GRPO Held-out Evaluation

训练 reward callback：

```
只能访问 reward_tests
```

在独立 evaluation interval 执行：

```
current checkpoint
→ fixed dev/PT sample
→ reward tests
→ heldout tests
```

得到：

[
R_{reward}(t)
]

[
R_{heldout}(t)
]

[
G_V(t)
]

[
FPR_V(t)
]

不得将 held-out score 回传给 optimizer。

# 34. External Evaluation

主 benchmark：

[
\boxed{LiveCodeBench}
]

名称默认：

> **Decontaminated LiveCodeBench Evaluation**

辅助 benchmark 只选择：

- HumanEval
  **或**
- MBPP

一个即可。

不堆 benchmark。

# 35. Evaluation Protocol

所有模型：

```
Base
SFT
DPO-E
DPO-H
GRPO-B
GRPO-D
```

必须固定：

- prompt serialization；
- temperature；
- top-p；
- max_new_tokens；
- samples/problem；
- verifier；
- benchmark version。

配置保存在：

```
configs/eval.yaml
```

# 36. Pass@k

若每题采样：

[
n
]

个 completion，其中：

[
c
]

个成功：

1-
\frac{
\binom{n-c}{k}
}{
\binom nk
}
}
]

不得使用：

[
c/n
]

冒充 Pass@k。

# 37. 最终 Main Table

必须生成：

| Model  | Internal Heldout | LCB Pass@1 | Pass@k | Executable Rate | Avg Tokens |
| ------ | ---------------- | ---------- | ------ | --------------- | ---------- |
| Base   |                  |            |        |                 |            |
| SFT    |                  |            |        |                 |            |
| DPO-E  |                  |            |        |                 |            |
| DPO-H  |                  |            |        |                 |            |
| GRPO-B |                  |            |        |                 |            |
| GRPO-D |                  |            |        |                 |            |

# 38. Difficulty Breakdown

| Model  | Easy | Medium | Hard |
| ------ | ---- | ------ | ---- |
| Base   |      |        |      |
| SFT    |      |        |      |
| DPO-E  |      |        |      |
| DPO-H  |      |        |      |
| GRPO-B |      |        |      |
| GRPO-D |      |        |      |

Difficulty definition 固定后禁止在看到结果以后修改。

# 39. DPO Analysis Artifacts

必须产出：

1. Easy/Hard reward-margin distribution；
2. Easy/Hard response-length distribution；
3. preference accuracy curve；
4. implicit reward margin；
5. Dev/LCB performance；
6. representative failures。

# 40. GRPO Analysis Artifacts

必须产出：

[
step\rightarrow MeanReward
]

[
step\rightarrow RewardStd
]

[
step\rightarrow KL
]

[
step\rightarrow AvgLength
]

[
step\rightarrow Z^+,Z^-
]

以及最重要的一张：

[
step
\rightarrow
\begin{cases}
RewardTest\
HeldoutTest\
ValidationPass@1
\end{cases}
]

如果有：

[
ClipFraction
]

同时保存。

# 41. Randomness / Statistical Control

不要因为：

[
27.0>26.5
]

就声明某方法优越。

## 默认

对 benchmark problems 做 bootstrap：

[
95%CI
]

## 若关键 pair 差距很小且算力允许

补第二 seed：

```
DPO-E vs DPO-H
GRPO-B vs GRPO-D
```

若 CI 大量重叠：

> 只报告 observed trend。

禁止使用：

> significantly better

之类措辞，除非真实证据支持。

# 42. Failure Analysis

至少人工检查：

[
10\sim20
]

个 case。

分类：

```
wrong algorithm
boundary condition
off-by-one
complexity
overflow
I/O
runtime
reasoning-code mismatch
verifier false positive
post-training regression
```

必须包含：

> post-training 后反而变差的 case。

# 43. 最重要的 Failure Patterns

## A. DPO-H < DPO-E

分析：

- margin 太小；
- preference ambiguity；
- verifier noise；
- insufficient capacity；
- data size。

## B. Reward ↑，Heldout ≈ constant

描述：

> verifier generalization gap。

不要直接叫：

> verifier overfitting。

## C. (Z^-) 长期很高

说明：

> 当前 policy 几乎无法探索 successful solution。

## D. (Z^+) 持续提高

说明：

> 大量 rollout compute 可能消耗在已经掌握的 prompt 上。

# 44. 真实资源记录

每次正式 run 都必须保存：

| Item                | Value |
| ------------------- | ----- |
| Backbone            |       |
| GPU model           |       |
| GPU count           |       |
| GPU memory          |       |
| SFT problems        |       |
| SFT tokens          |       |
| PT problems         |       |
| Preference rollouts |       |
| DPO matched pairs   |       |
| GRPO group size     |       |
| GRPO rollout tokens |       |
| SFT wall time       |       |
| DPO wall time       |       |
| GRPO wall time      |       |
| Peak memory         |       |

这些数据属于项目真实性证据。

# 45. 框架使用原则

## SFT / DPO

优先：

```
Transformers
PEFT
TRL
```

## Preference Sampling / Evaluation

算力需要时使用：

```
vLLM
```

## GRPO

第一目标：

> TRL 跑通完整闭环。

若环境稳定且时间允许：

> 再迁移到 veRL。

没有真实使用：

> 简历禁止写 veRL / vLLM。

# 46. Repository Structure

```
CodeReason-PT/
│
├── PLAN.md
├── README.md
│
├── data/
│   ├── prepare.py
│   ├── join_reasoning.py
│   ├── split_tests.py
│   ├── deduplicate.py
│   ├── schemas.py
│   └── validate.py
│
├── verifier/
│   ├── extract_code.py
│   ├── sandbox.py
│   ├── executor.py
│   ├── judge.py
│   ├── result.py
│   └── validate_reference_solutions.py
│
├── sft/
│   ├── train.py
│   ├── evaluate.py
│   └── config.yaml
│
├── preference/
│   ├── sample.py
│   ├── build_pairs.py
│   ├── length_match.py
│   ├── analyze_margin.py
│   └── validate_pairs.py
│
├── dpo/
│   ├── train.py
│   ├── loss_from_scratch.py
│   ├── validate_loss.py
│   └── evaluate.py
│
├── grpo/
│   ├── rewards.py
│   ├── train.py
│   ├── advantage_toy.py
│   ├── diagnostics.py
│   └── evaluate_checkpoints.py
│
├── eval/
│   ├── internal_hidden.py
│   ├── livecodebench.py
│   ├── pass_at_k.py
│   ├── bootstrap.py
│   └── metrics.py
│
├── analysis/
│   ├── build_tables.py
│   ├── plot_dpo.py
│   ├── plot_grpo.py
│   └── failure_cases.md
│
├── configs/
│   ├── base_eval.yaml
│   ├── sft.yaml
│   ├── dpo_easy.yaml
│   ├── dpo_hard.yaml
│   ├── grpo_binary.yaml
│   ├── grpo_dense.yaml
│   └── eval.yaml
│
├── results/
│
├── scripts/
│   ├── 00_prepare_data.sh
│   ├── 01_validate_verifier.sh
│   ├── 02_eval_base.sh
│   ├── 03_train_sft.sh
│   ├── 04_sample_preferences.sh
│   ├── 05_train_dpo.sh
│   ├── 06_train_grpo.sh
│   ├── 07_eval_all.sh
│   └── 08_analyze.sh
│
└── tests/
```

# 47. Codex 执行规则

Codex 必须按照阶段实现。

禁止：

> 一次性把整个仓库自动生成完再调。

每阶段必须先通过验收，再进入下一阶段。

# 48. Phase 0：环境与数据

实现：

- repo skeleton；
- config system；
- logging；
- dataset schema；
- high-confidence dataset mapping；
- SFT/PT split；
- SFT↔PT dedup；
- reward/heldout split。

**Gate**

```
数据可加载
Schema validation 通过
SFT/PT 无 ID overlap
SFT/PT near-duplicate report 已生成
reward/heldout split 固定
```

# 49. Phase 1：Verifier

实现：

```
code extraction
execution
test judging
sandbox
structured result
```

**Gate**

Reference solutions：

[
AcceptanceRate\approx100%
]

随机错误 solution 能正确分类：

```
CE / RE / TLE / WA
```

# 50. Phase 2：Base Baseline

完成固定：

```
Base → Dev / LCB
```

**Gate**

必须已有真实 baseline table。

从此以后 evaluation config 不随意改动。

# 51. Phase 3：SFT

完成：

[
Base\rightarrow SFT
]

**Gate**

必须：

- checkpoint 可加载；
- output 格式稳定；
- executable rate 无严重 collapse；
- Dev 有正式数字；
- loss curve 已保存。

# 52. Phase 4：Preference Dataset

完成：

[
K\text{-way sampling}
]

并生成：

[
(x,y^+,y_E^-,y_H^-)
]

matched dataset。

**Gate**

至少有数百个 matched pairs。

必须输出：

- margin distribution；
- length distribution；
- FPR；
- sample examples。

# 53. Phase 5：DPO

先：

```
loss_from_scratch
```

再：

```
TRL numerical validation
```

最后训练：

```
DPO-E
DPO-H
```

**Gate**

- custom loss 数值对齐；
- 两组使用相同 matched prompts；
- training curves；
- Dev results。

# 54. Phase 6：GRPO-B

首先只实现 Binary RLVR。

验证：

[
generate
\rightarrow
verify
\rightarrow
reward
\rightarrow
advantage
\rightarrow
policy\ update
]

闭环。

**Gate**

- reward 非恒定；
- policy parameters 真正更新；
- KL / reward / variance 正常记录；
- 无 NaN / explosion。

# 55. Phase 7：GRPO-D

在 GRPO-B 已通过 Gate 后，只替换 reward。

保持其余配置尽可能一致。

**Gate**

Binary/Dense 都有：

- checkpoint；
- training logs；
- diagnostic curves。

# 56. Phase 8：Unified Evaluation

统一评测：

```
Base
SFT
DPO-E
DPO-H
GRPO-B
GRPO-D
```

生成：

- main table；
- difficulty breakdown；
- bootstrap CI；
- executable metrics。

# 57. Phase 9：Analysis

必须生成：

### DPO

```
preference margin
length distribution
implicit reward
preference accuracy
performance
```

### GRPO

```
mean reward
reward std
KL
length
Z+
Z-
reward vs heldout
validation Pass@1
```

### Verifier

```
G_V
FPR_V
```

### Failure

至少 10–20 cases。

# 58. 项目优先级

## P0：必须完成

- data mapping；
- Base benchmark；
- verifier；
- reward/heldout split；
- SFT；
- policy sampling；
- matched Easy/Hard DPO；
- Binary/Dense GRPO；
- GRPO diagnostics；
- LiveCodeBench；
- basic decontamination；
  -真实实验表。

## P1：尽量完成

- FPR trend；
- bootstrap CI；
- second seed；
- detailed failure analysis；
- more robust near-dedup。

## P2：时间充足再做

- veRL migration；
- vLLM optimization；
- DPO-H → GRPO-D warm start。

# 59. 明确禁止继续加的模块

冻结：

```
PPO experiment
DAPO experiment
GSPO experiment
IPO
ORPO
SimPO
KTO
RAG
Agent
multiple backbones
LoRA rank ablation
sandbox research
FSDP benchmark
extra benchmarks
```

这些内容可以学习。

但：

[
\boxed{Project\ Scope<Interview\ Knowledge\ Scope}
]

# 60. 降级策略

## 3B 显存不够

[
3B\rightarrow1.5B
]

保持实验结构不变。

## GRPO OOM / 太慢

依次：

```
G 8 → 4
completion length ↓
PT prompts ↓
3B → 1.5B
```

不要先删除 GRPO branch。

## Hard negatives 太少

依次：

```
K 8 → 16
τ_H 0.5 → 0.4
```

报告真实 margin distribution。

## OCR Mapping 数量太少

接受小数据。

不要 fuzzy join。

## DPO-H 没提升

不补造实验结果。

分析：

```
margin
noise
length
capacity
distribution
```

## Dense GRPO 没赢

作为真实结论。

重点分析：

```
reward proxy quality
variance
heldout gap
FPR
```

# 61. 推荐执行节奏

不是硬性 deadline。

## Day 1–2

```
data
mapping
schema
dedup
base evaluation
```

## Day 3

```
verifier
sandbox
reward/heldout
reference validation
```

## Day 4

```
SFT
```

## Day 5

```
policy sampling
preference construction
```

## Day 6

```
scratch DPO
DPO-E/H
```

## Day 7–8

```
GRPO-B
GRPO-D
```

## Day 9

```
LCB / unified eval
```

## Day 10–11

```
diagnostics
hidden-test analysis
decontamination
```

## Day 12

```
failure analysis
bootstrap
```

## Day 13

```
second seed / failed experiment fix
```

## Day 14

```
README
figures
resume
interview review
```

实际优先级永远高于日程。

# 62. README 最终结构

```
1. Motivation
2. Research Questions
3. System Overview
4. Dataset & Provenance
5. Executable Verifier
6. Reasoning SFT
7. Policy-Sampled Preference Construction
8. DPO
9. GRPO / RLVR
10. Hidden-test Evaluation
11. LiveCodeBench Evaluation
12. Training Dynamics
13. Results
14. Failure Analysis
15. Limitations
16. Reproduction
```

README 应该像：

> 小型 research-engineering report。

不是单纯安装教程。

# 63. 最终简历模板

真实完成后再填写数字。

### CodeReason-PT｜基于可执行验证信号的代码推理大模型后训练

*个人项目｜LLM Post-training / DPO / GRPO / RLVR*

- 基于 **Qwen2.5-Coder** 搭建代码推理 Post-training 流水线，完成 Reasoning SFT、DPO 与 GRPO/RLVR；实现隔离代码执行 Verifier，通过单元测试构造可验证训练反馈，并跟踪 reward、KL、response length 与 rollout reward variance 等训练动态。
- 基于 **SFT-policy sampled responses** 自动构建 matched preference pairs，在固定 prompt/chosen 与训练预算条件下比较 executable easy/hard-negative DPO；进一步比较 binary outcome 与 testcase-pass-rate dense reward 下的 GRPO，分析 preference margin、reward sparsity 与 zero-variance rollout 对训练行为的影响。
- 将训练 testcase 拆分为 **reward / held-out tests**，通过 verifier generalization gap 与 false-positive rate 分析训练信号可靠性，并在去污染 **LiveCodeBench** 上统一评测 Base/SFT/DPO/GRPO；最佳 Post-training variant 将 Pass@1 从 **XX.X% 提升至 YY.Y%**，结合 bootstrap CI、难度分层及失败样例分析跨题泛化。

没有真实使用的技术名词必须删除。

# 64. 90 秒面试介绍

我这个项目主要研究的是代码执行反馈应该怎样转化成有效的 LLM Post-training signal，而不是单纯把 SFT、DPO 和 GRPO 都跑一遍。

我首先基于 Qwen-Coder 做 Reasoning SFT，然后实现了一个隔离代码执行 Verifier，并把每道 Post-training 题目的 testcase 拆成 reward tests 和 held-out tests。

Preference Optimization 这一支中，我从 SFT policy 对没有进入 SFT 的题进行多路采样，通过 reward tests 自动构造 matched preference pairs。固定 prompt 和 chosen，只改变 rejected 的 correctness margin，对比 executable easy negative 和大部分测试通过但仍有错误的 hard negative，从而研究 preference hardness 对 DPO 的影响。

在线 RL 这一支从同一个 SFT checkpoint 出发，用 GRPO 比较 all-tests-pass binary reward 和 testcase pass-rate dense reward，同时跟踪 reward variance、KL 和 zero-variance groups 等训练动态。

我没有直接把 training reward 上升当成能力提升，而是使用 held-out tests 观察 verifier generalization gap，并最终在去污染的 LiveCodeBench 上检查这些训练 signal 是否真的转化成跨问题的代码推理泛化。

# 65. Codex 的最高优先级指令

执行过程中始终遵守以下规则：

1. **不要增加本 PLAN 未定义的新算法。**
2. **每个 Phase 先通过 Gate，再进入下一阶段。**
3. **数据质量优先于数量。**
4. **禁止 fuzzy join 强行扩大 SFT 数据。**
5. **Held-out tests 禁止进入任何训练 signal。**
6. **DPO-E/H 必须基于 matched prompts。**
7. **GRPO-B/D 必须从同一个 SFT checkpoint 开始。**
8. **所有模型使用统一 evaluation protocol。**
9. **任何实验结果不得提前假设。**
10. **必须保存真实 config、seed、日志、GPU、wall time 和失败案例。**
11. **遇到实验失败优先定位原因，不得偷偷修改实验定义让结果变好。**
12. **P0 没完成之前禁止投入时间进行 P2 优化。**

# Final Scope

最终项目严格冻结为：

[
\boxed{
Executable\ Verifier
\rightarrow
Reasoning\ SFT
\rightarrow
\begin{cases}
Controlled\ Preference\ DPO\
Verifiable\ GRPO
\end{cases}
\rightarrow
Hidden\ Test
+
Decontaminated\ External\ Evaluation
}
]

核心研究对象：

[
\boxed{
Preference\ Hardness
+
Reward\ Granularity
+
Verifier\ Signal\ Reliability
}
]

最终评价标准不是“用了多少框架”，而是是否真实产生：

- model checkpoints；
- training logs；
- rollout data；
- preference data；
- DPO / GRPO curves；
- verifier diagnostics；
- benchmark numbers；
- failure cases；
- reproducible configs。

如果这些全部存在，这个项目即达到可以作为 **LLM / Post-training 算法实习核心简历项目** 的完成标准。
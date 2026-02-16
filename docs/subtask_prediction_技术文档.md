# π0.5 Subtask Prediction 完整技术文档

## 1. 背景与动机

### 1.1 论文描述

根据 π0.5 论文，推理时模型执行**两阶段**过程：

1. **Stage 1 — Subtask Prediction**: 用户给出高层指令（如 "clean the bedroom"），VLM 结合当前图像观测，**自回归生成**子任务文本（如 "pick up the pillow"）
2. **Stage 2 — Action Generation**: 将生成的子任务作为 low-level command，送入 Action Expert 通过 Flow Matching 生成连续动作序列

### 1.2 openpi 现状

官方 openpi 代码**只实现了 Stage 2**（给定 prompt → 生成动作），缺失 Stage 1。
- [GitHub Issue #790](https://github.com/Physical-Intelligence/openpi/issues/790): 确认了 compute_loss 仅计算 action loss，缺少 VLM language loss
- [GitHub Issue #635](https://github.com/Physical-Intelligence/openpi/issues/635): 确认了只支持 flow matching head，不支持 subtask generation via NTP

### 1.3 参考实现

| 来源 | 借鉴内容 |
|------|---------|
| [BrunoFANG1/openpi_subtask_generation](https://github.com/BrunoFANG1/openpi_subtask_generation) | tokenizer 层的 `tokenize_high_level_prompt()`、`detokenize()` 方法设计思路；高低层 prompt 格式化方案 |
| π0.5 论文 Figure 3 | 两阶段推理架构设计 |
| openpi 现有 `pi0_fast.py` | 自回归生成循环的模式参考（该文件用于生成 action tokens，我们改为生成 text tokens） |

## 2. 核心设计原理

### 2.1 为什么能 work

PaliGemma 是一个 prefix-LM 架构的 VLM，其 `Embedder` 类天然具备双向映射能力：

```python
# gemma.py 中的 Embedder 类
class Embedder:
    def encode(self, x):    # token_ids → embeddings
        return self.input_embedding_table[(x,)] * sqrt(embed_dim)
    
    def decode(self, x):    # hidden_states → logits
        return jnp.dot(x, self.input_embedding_table.T)
```

在 π0.5 的 pre-training 阶段，VLM 被**联合训练**了 language subtask 预测目标（论文 Figure 3 的 "language subtasks" 部分）。因此预训练权重中已经编码了子任务分解的能力。我们只需要提供一个自回归解码循环来激活这个能力。

### 2.2 两阶段推理架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 1: Subtask Generation                  │
│                                                                 │
│  输入: Images + "Task: clean the bedroom. Subtask: "            │
│                                                                 │
│  ┌──────────┐    ┌──────────┐                                   │
│  │  SigLIP   │    │ Embed    │                                   │
│  │  Encoder   │    │ Prompt   │                                   │
│  └────┬──────┘    └────┬─────┘                                   │
│       ↓                ↓                                         │
│  image_tokens     prompt_tokens                                  │
│  [b,~768,2048]    [b, 200, 2048]                                │
│       ↓                ↓                                         │
│  prefix_tokens = concat → [b, ~968, 2048]                       │
│       ↓                                                         │
│  VLM Forward Pass (Expert 0 only, Expert 1 = None)              │
│       ↓                                                         │
│  prefix_out [b, ~968, 2048] + KV Cache                          │
│       ↓                                                         │
│  ┌─ Autoregressive Loop (max 50 steps) ─────────────────────┐   │
│  │ 1. logits = Embedder.decode(hidden) → [b, 257152]        │   │
│  │ 2. token = argmax(logits) → int32[b]                      │   │
│  │ 3. if EOS: break                                          │   │
│  │ 4. embed → [b,1,2048] → VLM forward + KV cache update    │   │
│  └───────────────────────────────────────────────────────────┘   │
│       ↓                                                         │
│  detokenize → "pick up the pillow"                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Stage 2: Action Generation                    │
│                                                                 │
│  输入: Images + "Task: pick up the pillow, State: 128 130...;   │
│                  \nAction: "                                     │
│       ↓                                                         │
│  标准 π0.5 sample_actions() 流程                                │
│  (prefix 填充 KV cache → while_loop 去噪)                      │
│       ↓                                                         │
│  actions: float32[batch, 50, 32]                                │
└─────────────────────────────────────────────────────────────────┘
```

## 3. 修改的文件清单

| 文件 | 修改类型 | 新增内容 |
|------|---------|---------|
| `src/openpi/models/gemma.py` | 新增方法 | `Module.decode_to_logits()` — LM head |
| `src/openpi/models/tokenizer.py` | 新增方法 | `tokenize_subtask_prompt()`, `detokenize()` |
| `src/openpi/models/pi0.py` | 新增方法 | `generate_subtask()`, `sample_actions_with_subtask()` |
| `src/openpi/transforms.py` | 新增类 | `TokenizeSubtaskPrompt` |
| `src/openpi/policies/policy.py` | 新增类 | `SubtaskPolicy` |
| `src/openpi/policies/policy_config.py` | 新增函数 | `create_subtask_policy()` |
| `scripts/test_subtask_inference.py` | 新建文件 | 测试脚本 |

## 4. 各模块新增代码详解

---

### 4.1 `gemma.py` — `Module.decode_to_logits()`

**位置**: `_gemma.Module` 类中，`embed()` 方法之后

**代码**:
```python
def decode_to_logits(self, hidden_states):
    """将 hidden states 映射回词表 logits。
    等价于标准 LM 的 language model head。"""
    return self.embedder.decode(hidden_states)
```

**输入/输出形状**:
| 参数 | 形状 | 说明 |
|------|------|------|
| `hidden_states` (输入) | `float32[batch, seq_len, 2048]` | Transformer 最后一层 + final_norm 的输出 |
| `logits` (输出) | `float32[batch, seq_len, 257152]` | 词表上的 unnormalized log-probabilities |

**原理**: `Embedder.decode(x)` = `jnp.dot(x, embedding_table.T)` = 线性映射。与标准 LM 共享 embedding-unembed 权重。

**调用方式**: 通过 linen bridge — `self.PaliGemma.llm(hidden, method="decode_to_logits")`

---

### 4.2 `tokenizer.py` — 新增方法

#### 4.2.1 `tokenize_subtask_prompt()`

**目的**: 将高层指令格式化为 subtask 生成的 prefix prompt

**格式**: `"Task: clean the bedroom. Subtask: "` + padding

**输入/输出**:
| 参数 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `high_level_prompt` (输入) | `str` | — | 高层指令，如 "clean the bedroom" |
| `tokens` (输出) | `int32` | `[max_token_len]` | SentencePiece token IDs，右侧 padding 为 0 |
| `mask` (输出) | `bool` | `[max_token_len]` | True = 有效 token，False = padding |

**处理流程**:
1. 清理文本（lowercase, strip, 替换 `_` 和 `\n`）
2. 规范化末尾标点，添加 `.` 分隔符
3. 拼接为 `"Task: X. Subtask: "` 格式
4. SentencePiece encode (add_bos=True)
5. Padding 到 max_token_len

**借鉴**: BrunoFANG1 的 `tokenize_high_level_prompt()` 格式设计

#### 4.2.2 `detokenize()`

**目的**: 将生成的 token IDs 解码回文本

**输入/输出**:
| 参数 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `tokens` (输入) | `int32` | `[seq_len]` | token IDs，可能包含 padding(0) 和 EOS(1) |
| 返回值 | `str` | — | 解码后的文本 |

**处理**: 过滤掉 id ≤ 1 的 token（padding=0, EOS=1），然后用 SentencePiece decode

**借鉴**: BrunoFANG1 的 `detokenize()` 方法

---

### 4.3 `pi0.py` — 核心自回归生成

#### 4.3.1 `generate_subtask()`

**位置**: `Pi0` 类中，`sample_actions()` 之后

**目的**: Stage 1 — VLM 自回归生成子任务文本

**函数签名**:
```python
def generate_subtask(
    self,
    rng: KeyArray,
    observation: Observation,
    *,
    max_gen_steps: int = 50,
    temperature: float = 0.0,
) -> jnp.ndarray:  # int32[batch, max_gen_steps]
```

**参数说明**:
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `rng` | `KeyArray` | — | JAX 随机 key（温度>0时使用） |
| `observation` | `Observation` | — | 包含图像和已格式化的高层 prompt tokens |
| `max_gen_steps` | `int` | 50 | 最大生成 token 数 |
| `temperature` | `float` | 0.0 | 采样温度，0=greedy argmax |

**完整数据流**:

```
Step 1: embed_prefix(observation)
  ├─ 图像: SigLIP(images) → image_tokens [batch, ~768, 2048]
  ├─ 语言: Embed(tokenized_prompt) → prompt_tokens [batch, 200, 2048]
  └─ concat → prefix_tokens [batch, ~968, 2048]
       prefix_mask [batch, ~968]  (True for valid, False for padding)
       prefix_ar_mask [~968]      (全 False = 双向 attention)

Step 2: VLM Forward Pass
  PaliGemma.llm([prefix_tokens, None], mask, positions)
  → (prefix_out [batch, ~968, 2048], None), kv_cache
  注意: 传入 [prefix_tokens, None] — 只运行 VLM，不运行 Action Expert

Step 3: 获取最后有效位置的 logits
  prefix_len = sum(prefix_mask) → [batch]  (实际有效长度, e.g. 783)
  last_valid_idx = prefix_len - 1 → [batch]
  last_hidden = prefix_out[arange(batch), last_valid_idx, :] → [batch, 2048]
  logits = decode_to_logits(last_hidden[:, None, :]) → [batch, 1, 257152]
  last_logit = logits[:, 0, :] → [batch, 257152]

Step 4-7: 自回归循环 (Python for-loop)
  for step_i in range(max_gen_steps):
    ┌── 采样: token = argmax(last_logit) → int32[batch]
    ├── 存储: generated_tokens[:, step_i] = token
    ├── EOS 检查: if all(token == 1): break
    ├── 嵌入: token_emb = Embed(token[:, None]) → [batch, 1, 2048]
    ├── 位置: next_pos = prefix_len + step_i → [batch, 1]
    ├── 掩码: gen_mask [batch, 1, kv_len+1] (mask padding positions)
    ├── VLM Forward: PaliGemma.llm([token_emb, None], kv_cache=kv_cache)
    │     → vlm_out [batch, 1, 2048], kv_cache updated
    └── 下一步 logits: decode_to_logits(vlm_out) → [batch, 1, 257152]

返回: generated_tokens int32[batch, max_gen_steps]
```

**关键设计决策**:
1. **Python for-loop 而非 `jax.lax.while_loop`**: 因为 KV cache 的 seq_len 每步增长，while_loop 要求固定形状
2. **只运行 VLM (Expert 0)**: 传入 `[tokens, None]`，Action Expert 被跳过
3. **从最后有效位置取 logits**: 避免从 padding 位置取值（critical bug fix）
4. **EOS token = 1**: 与 PaliGemma 标准一致

#### 4.3.2 `sample_actions_with_subtask()`

**目的**: 完整的两阶段推理

**函数签名**:
```python
def sample_actions_with_subtask(
    self,
    rng: KeyArray,
    observation: Observation,
    *,
    high_level_prompt: str,
    tokenizer: PaligemmaTokenizer,
    max_gen_steps: int = 50,
    temperature: float = 0.0,
    num_steps: int = 10,
    noise: Array | None = None,
) -> tuple[Actions, str]:
```

**数据流**:
```
输入: observation (images + state) + "clean the bedroom"

Stage 1:
  tokenize_subtask_prompt("clean the bedroom")
  → "Task: clean the bedroom. Subtask: " → tokens [200]
  → 广播到 [batch, 200]
  → 构造 subtask_obs (覆盖 tokenized_prompt)
  → generate_subtask(subtask_obs) → int32[batch, 50]
  → detokenize → "pick up the pillow"

Stage 2:
  tokenizer.tokenize("pick up the pillow", state=state)
  → "Task: pick up the pillow, State: 128 130 ...;\nAction: "
  → tokens [200], mask [200]
  → 广播到 [batch, 200]
  → 构造 action_obs (覆盖 tokenized_prompt)
  → sample_actions(action_obs) → float32[batch, 50, 32]

输出: (actions, "pick up the pillow")
```

---

### 4.4 `transforms.py` — `TokenizeSubtaskPrompt`

**目的**: 作为数据 transform，将 high-level prompt 格式化为 subtask 生成 prefix

**输入**: data dict 中的 `"prompt"` key (str)
**输出**: data dict 中添加 `"tokenized_prompt"` int32[max_len] 和 `"tokenized_prompt_mask"` bool[max_len]

**注意**: 这个 transform 主要用于**独立测试** subtask generation。在 `SubtaskPolicy` 中，tokenization 由 `sample_actions_with_subtask()` 内部处理。

---

### 4.5 `policy.py` — `SubtaskPolicy`

**目的**: 封装两阶段推理逻辑，对外提供统一的 `infer()` 接口

**初始化参数**:
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | `BaseModel` | — | Pi0 模型实例 |
| `transforms` | `list` | `()` | 输入变换（**不含** TokenizePrompt） |
| `output_transforms` | `list` | `()` | 输出变换 |
| `subtask_max_gen_steps` | `int` | 50 | subtask 最大生成长度 |
| `subtask_temperature` | `float` | 0.0 | subtask 采样温度 |
| `tokenizer_max_len` | `int` | 200 | tokenizer 最大 token 长度 |

**infer() 流程**:
```
1. 提取 high_level_prompt（在 transforms 消费 prompt 前）
2. 应用 input transforms（图像处理、状态归一化等）
3. 构造 Observation
4. 调用 model.sample_actions_with_subtask()
5. 应用 output transforms
6. 返回 {actions, state, subtask_text, policy_timing}
```

---

### 4.6 `policy_config.py` — `create_subtask_policy()`

**目的**: 工厂函数，正确配置 SubtaskPolicy

**关键逻辑**: 自动从 transform 链中**过滤掉** `TokenizePrompt`，因为 SubtaskPolicy 内部处理 tokenization：
```python
for t in all_transforms:
    if not isinstance(t, transforms.TokenizePrompt):
        input_transforms_no_tokenize.append(t)
```

## 5. 数据形状验证矩阵

以 `pi05_droid` (Gemma 2B) 为例：

| 变量 | 形状 | 类型 | 产生位置 |
|------|------|------|---------|
| images (per camera) | `[1, 224, 224, 3]` | float32 | transforms |
| image_tokens (per camera) | `[1, 256, 2048]` | float32 | SigLIP encoder |
| tokenized_prompt | `[1, 200]` | int32 | tokenizer |
| tokenized_prompt_mask | `[1, 200]` | bool | tokenizer |
| prompt_embeddings | `[1, 200, 2048]` | bf16 | Gemma embed |
| prefix_tokens | `[1, 968, 2048]` | bf16 | embed_prefix concat |
| prefix_mask | `[1, 968]` | bool | embed_prefix concat |
| prefix_ar_mask | `[968]` | bool | embed_prefix (全 False) |
| prefix_out (VLM) | `[1, 968, 2048]` | bf16 | Gemma Module |
| kv_cache K | `[18, 1, 968, 1, 256]` | bf16 | Attention layers |
| kv_cache V | `[18, 1, 968, 1, 256]` | bf16 | Attention layers |
| last_hidden | `[1, 1, 2048]` | bf16 | gather from prefix_out |
| logits | `[1, 1, 257152]` | float32 | decode_to_logits |
| generated_token | `[1]` | int32 | argmax |
| token_embedding | `[1, 1, 2048]` | bf16 | Gemma embed |
| gen_mask (step 0) | `[1, 1, 969]` | bool | constructed |
| gen_mask (step i) | `[1, 1, 968+i+1]` | bool | constructed |
| generated_tokens | `[1, 50]` | int32 | accumulated |
| actions (final) | `[1, 50, 32]` | float32 | flow matching |

**KV Cache 增长轨迹**:
```
初始 prefix pass:  K/V shape = [18, 1, 968, 1, 256]
Gen step 0:        K/V shape = [18, 1, 969, 1, 256]
Gen step 1:        K/V shape = [18, 1, 970, 1, 256]
...
Gen step n:        K/V shape = [18, 1, 968+n+1, 1, 256]
```

## 6. Attention Mask 详解

### Stage 1 (subtask generation) 中的 mask 构造

**Prefix pass**: 标准双向 attention（ar_mask 全 False）
```
prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
→ [batch, prefix_len, prefix_len]
→ 有效位之间全部为 True，padding 位为 False
```

**Generation step i**: 新 token 可以 attend 到所有有效的 previous tokens + 自身
```python
# kv_cache 中有 968+i 个位置 (prefix + i个已生成token)
# 当前新 token 是第 968+i+1 个
kv_seq_len = kv_cache[0].shape[2]  # = 968 + i
gen_mask = ones(batch, 1, kv_seq_len + 1)  # 全 1

# 但 prefix 的 padding 位不应该被 attend 到
prefix_pad_mask = concat(
    prefix_mask,                           # [batch, 968] — 有效 prefix 位
    ones(batch, i + 1)                     # [batch, i+1] — 所有生成的 token 都有效
)                                          # [batch, 968 + i + 1] = [batch, kv_seq_len + 1]

gen_mask = gen_mask * prefix_pad_mask[:, None, :]  # 遮掩 padding
→ [batch, 1, 968+i+1]
```

## 7. 与现有代码的兼容性

| 方面 | 影响 |
|------|------|
| `sample_actions()` | ✅ 完全不变 |
| `compute_loss()` | ✅ 完全不变 |
| `Policy.infer()` | ✅ 完全不变 |
| `train.py` | ✅ 完全不变 |
| 现有 config | ✅ 完全不变 |
| checkpoint 兼容 | ✅ 新方法只使用现有权重，不需要额外参数 |

## 8. 运行命令

### 本地环境 (已验证)

由于已将 `test_subtask_inference.py` 的默认配置修改为适配本地 `pi05_libero`，你可以直接运行：

```bash
# 使用本地 pi05_libero checkpoint 进行测试
uv run scripts/test_subtask_inference.py --prompt "pick up the red cup"
```

或者显式指定参数（推荐在文档中记录完整命令）：

```bash
# 显式指定 config 和 checkpoint（单卡运行，推荐）
TF_CPP_MIN_LOG_LEVEL=2 uv run scripts/test_subtask_inference.py \
    --config pi05_libero \
    --checkpoint /storages/liweile/Robo/openpi/checkpoints/pi05_libero/reproduce_libero/29999 \
    --prompt "pick up the red cup" \
    --temperature 0.0
```

### 其他常用命令 (需下载对应模型)

> **注意**: `pi05_base` 不是有效的 config 名称。它仅作为预训练 checkpoint 路径存在，
> 被 fine-tuning 配置引用。有效的 pi05 config 为：`pi05_libero`、`pi05_droid`、`pi05_aloha`。

```bash
# 使用 pi05_droid（需要从 GCS 下载 checkpoint）
TF_CPP_MIN_LOG_LEVEL=2 uv run scripts/test_subtask_inference.py \
    --config pi05_droid \
    --prompt "clean the table"

# 带温度采样 (增加生成的多样性)
TF_CPP_MIN_LOG_LEVEL=2 uv run scripts/test_subtask_inference.py \
    --config pi05_droid \
    --prompt "organize the workspace" \
    --temperature 0.7
```

# 4. 在 Python 代码中使用 SubtaskPolicy
from openpi.training import config as _config
from openpi.policies import policy_config

# 使用 pi05_libero 配置
config = _config.get_config("pi05_libero")
policy = policy_config.create_subtask_policy(
    config,
    "/storages/liweile/Robo/openpi/checkpoints/pi05_libero/reproduce_libero/29999",
    subtask_max_gen_steps=50,
    subtask_temperature=0.0,
)

# 注意：Libero 需要 observation/image 等 key
result = policy.infer({
    "observation/image": image_array,       # [224, 224, 3]
    "observation/wrist_image": wrist_array, # [224, 224, 3]
    "observation/state": state_array,       # [8]
    "prompt": "clean the table",
})
print(f"Subtask: {result['subtask_text']}")
print(f"Actions: {result['actions'].shape}")
```

## 9. 局限性与未来改进方向

1. **subtask generation 不可 JIT**: 由于 KV cache 动态增长，使用 Python for-loop，每步都有 dispatch overhead
2. **缺少 subtask training loss**: 当前只实现了推理。训练时需要在 `compute_loss` 中添加 language modeling loss
3. **batch_size > 1 的 EOS 处理**: 当前使用 `all(token == EOS)` 全 batch 判断，部分 batch 提前结束时会继续生成
4. **fine-tuned checkpoint 的 subtask 能力**: 经过 action-only fine-tuning 后，subtask generation 能力可能退化
5. **未实现 knowledge insulation**: 论文中的知识隔离训练策略未包含在此实现中

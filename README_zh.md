<p align="center">
  <img src="assets/nanoclaw.png" alt="Nano Claw Code" width="128" height="128">
  <h1 align="center">Nano-Claw-Code</h1>
  <p align="center">
    <em>蒸馏后的编程智能体 — 更少工具，同等性能，~5,800 行 Python。</em>
  </p>
  <p align="center">
    <a href="https://github.com/OpenLAIR/dr-claw/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-GPL--3.0%20%2B%20AGPL--3.0-blue?style=for-the-badge" alt="License: GPL-3.0 + AGPL-3.0" /></a>
  </p>
  <p align="center">
    <a href="README.md">English</a> | 中文
  </p>
</p>

---

## 目录

- [这是什么？](#这是什么)
- [快速开始](#快速开始)
- [路线图](#路线图)
- [主要结果](#主要结果)
- [贡献](#贡献)
- [蒸馏流程](#蒸馏流程)
- [仓库结构](#仓库结构)
- [安装](#安装)
- [使用](#使用)
- [SWE-bench 评测](#swe-bench-评测)
- [许可](#许可)

---

## 这是什么？

Nano-Claw-Code 是一个从完整版 [Claude Code](https://github.com/anthropics/claude-code) 框架中**系统性蒸馏**出来的轻量级 Python 编程智能体。蒸馏分两步：

1. **TypeScript 裁剪** — 分析 SWE-bench 上的工具使用情况，从原始 Claude Code 中移除 29 个未使用的工具和 4 组服务（~405,500 → ~378,100 核心代码行）。
2. **Python 重写** — 将核心 Agent 循环、工具实现和 CLI 用纯 Python 重写，把 ~378,100 行 TypeScript 压缩为 **~5,800 行 Python**，同时保留相同的工具调用接口和 Agent 能力。

最终结果在 [SWE-bench Lite](https://www.swebench.com/) 上**达到**完整版 TypeScript Agent 的表现。

<p align="center">
  <img src="assets/screenshot.png" width="700" alt="Nano-Claw-Code — 终端截图" />
</p>

---

## 快速开始

```bash
git clone https://github.com/OpenLAIR/nano-claw-code.git   # 或你的 fork
cd nano-claw-code
pip install -e .                    # 或：uv sync && source .venv/bin/activate
cp .env.example .env                # 可选；编辑密钥（或使用下方 export）
./start.sh                          # 与安装后的 nano-claw-code 等价
```

---

## 路线图

- [x] 从 Claude Code 蒸馏（42 → 13 工具，TypeScript 裁剪）
- [x] Python 重写 — nano-claw-code（~5,800 行，12 工具）
- [x] SWE-bench 评测框架（含完整 trace 日志，已包含在仓库中）
- [x] SWE-bench Lite 对比评测（50/300 实例）
- [ ] SWE-bench Lite 全量运行（300 实例）
- [ ] SWE-bench Verified 运行（500 实例）
- [ ] 通过 OpenRouter 评测第三方模型（Kimi、MiniMax）
- [ ] 从 Agent trace 构建蒸馏数据集

---

## 主要结果

在 SWE-bench Lite 前 50 个实例上评测，模型为 `claude-sonnet-4-20250514`：

| 版本 | 语言 | 工具数 | 核心代码行数 | 提交 | 解决 | 解决率 |
|------|------|--------|------------|------|------|--------|
| **Claude Code**（完整版） | TypeScript | 42 | ~405,500 | 50 | 33 | 66.0% |
| **Nano-Claw-Code**（本仓库） | Python | 12 | **~5,800** | 50 | 31 | 62.0% |

> 代码量减少 ~70 倍，解决率接近。全量测试（300 实例）正在进行中。

---

## 贡献

### 1. 基于工具使用的蒸馏

完整版 Claude Code 定义了 **~56 个工具**，涵盖 Shell 执行、文件读写、网页访问、多 Agent 协作、计划模式、定时任务、MCP 集成等。我们分析了 Agent 在 SWE-bench 任务中实际调用了哪些工具，移除所有非必要部分：

<details>
<summary><b>移除 29 个工具</b>（点击展开完整列表）</summary>

| 移除的工具 | 行数 | 移除原因 |
|-----------|------|---------|
| `PowerShellTool` | 8,959 | 仅 Windows；`BashTool` 覆盖 Unix |
| `LSPTool` | 2,005 | 实验性语言服务器集成 |
| `SendMessageTool` | 997 | Agent 间消息传递（团队/群） |
| `EnterPlanModeTool` / `ExitPlanModeTool` | 934 | 计划模式 UI（SWE-bench 未使用） |
| `ConfigTool` | 809 | Anthropic 内部设置 |
| `BriefTool` | 610 | 输出格式化模式 |
| `ToolSearchTool` | 593 | 动态工具发现 |
| `EnterWorktreeTool` / `ExitWorktreeTool` | 563 | Git worktree 隔离 |
| `ScheduleCronTool` / `CronDelete` / `CronList` | 543 | 定时任务调度 |
| `TeamCreateTool` / `TeamDeleteTool` | 534 | 多 Agent 群体协作 |
| `TaskCreate` / `TaskGet` / `TaskUpdate` / `TaskList` / `TaskStop` / `TaskOutput` | 1,761 | V2 任务管理系统 |
| `ListMcpResourcesTool` / `ReadMcpResourceTool` | 381 | MCP 资源访问 |
| `AskUserQuestionTool` | 309 | 结构化提问 UI |
| `McpAuthTool` | 215 | MCP 认证 |
| `RemoteTriggerTool` | 192 | 远程 Agent 触发器 |
| `SyntheticOutputTool` | 163 | 结构化 JSON 输出 |
| `REPLTool` | 85 | REPL 模式包装器 |
| `SleepTool` | 17 | 睡眠工具 |
| `TungstenTool` | 5 | Anthropic 内部 |
| `WorkflowTool` | 2 | 工作流占位符 |

</details>

- **移除 4 组服务**（~7,400 行）— 团队记忆同步、语音转文字、LSP 服务器管理、插件生命周期
- **共裁剪 ~27,400 行**（核心框架的 6.8%），**性能无损**

### 2. Python 重写

将裁剪后的 Agent 重写为纯 Python — **~5,800 行**，15 个模块，**12 个工具**：

<details>
<summary><b>保留 12 个工具</b>（点击展开工具映射）</summary>

| 工具 | 功能 | 原 Claude Code 对应工具 |
|------|------|----------------------|
| `Read` | 文件读取，支持图片/目录 | `FileReadTool` |
| `Write` | 文件创建/覆写 | `FileWriteTool` |
| `Edit` | 字符串替换编辑 + diff 预览 | `FileEditTool` |
| `Bash` | 持久化工作目录的 Shell + 沙盒模式 | `BashTool` |
| `Glob` | 模式匹配，自动添加 `**/` 前缀 | `GlobTool` |
| `Grep` | 正则搜索，ripgrep 优先，Python 兜底 | `GrepTool` |
| `WebFetch` | URL 抓取 + HTML→文本转换 | `WebFetchTool` |
| `WebSearch` | DuckDuckGo HTML 搜索 | `WebSearchTool` |
| `NotebookEdit` | Jupyter 单元格创建/编辑 | `NotebookEditTool` |
| `TodoWrite` | 内存任务追踪，支持合并 | `TodoWriteTool` |
| `Agent` | 子 Agent 生成 + 工具过滤 | `AgentTool` |
| `Skill` | 从 `.claude/skills/` 加载技能 | `SkillTool` |

</details>

除工具外，Agent 还保留了完整版的关键基础设施：

<details>
<summary><b>保留 9 项基础设施能力</b>（点击展开）</summary>

| 能力 | 模块 | 功能 |
|------|------|------|
| 子 Agent 系统 | `agents.py` | 3 个内置配置（通用、探索、规划）+ 自定义 Agent（`.claude/agents/*.md`） |
| 技能系统 | `skills.py` | 从 `~/.claude/skills/` 发现技能，支持 frontmatter 元数据（内联/fork 执行） |
| 记忆层次 | `memory.py` | 分层加载 `CLAUDE.md` 上下文（全局 → 逐目录），支持 `@include` |
| 上下文压缩 | `agent.py` | 监控 token 预算（~200K），超过 75% 阈值时摘要压缩旧消息 |
| Prompt 缓存 | `agent.py` | Anthropic `cache_control: ephemeral` 断点，降低 token 消耗 |
| 权限系统 | `permissions.py` | 3 种模式（全部接受 / 手动 / 自动）+ 安全命令分类 |
| 会话持久化 | `session.py` | 对话保存/加载/恢复，自动保存和搜索 |
| API 重试 | `agent.py` | 429/5xx 指数退避 + 抖动，支持 `Retry-After` 头 |
| OpenAI 兼容 | `openai_compat.py` | 非 Anthropic 供应商的替代后端（Kimi、MiniMax 等） |

</details>

### 3. 多供应商模型支持

原始 Claude Code 仅支持 Anthropic API。Nano-Claw-Code 新增了对**任意 OpenAI 兼容端点**的原生支持，支持使用第三方模型进行评测和部署：

<details>
<summary><b>支持 4 类供应商</b>（点击展开）</summary>

| 供应商 | 环境变量 | 示例 |
|--------|----------|------|
| **Anthropic**（直连） | `ANTHROPIC_API_KEY` | Claude Sonnet、Claude Opus |
| **OpenRouter** | `OPENROUTER_API_KEY` + `OPENROUTER_MODEL` | OpenRouter 目录中的任意模型 |
| **OpenAI 兼容** | `OPENAI_COMPAT_BASE_URL` + `OPENAI_COMPAT_API_KEY` | Azure AI、Kimi（月之暗面）、MiniMax、DeepSeek、本地 vLLM/Ollama |
| **LiteLLM Proxy** | `ANTHROPIC_BASE_URL` + `ANTHROPIC_API_KEY` | 统一网关，支持 100+ 供应商 |

</details>

`openai_compat.py` 模块（~600 行）将 Agent 的 Anthropic 原生工具调用协议转换为标准 OpenAI Chat Completions 格式 — 处理工具 schema、流式增量和多轮工具调用/结果对。供应商检测基于环境变量自动完成，切换模型无需修改代码。

### 4. SWE-bench 对比评测

两个版本在相同条件下评测，评测框架记录完整的 trace 日志 — 包括每个工具调用、模型回复和思考过程。

---

## 蒸馏流程

```
┌─────────────────────┐      prune 29 tools     ┌─────────────────────┐        rewrite in       ┌─────────────────────┐
│  Claude Code        │ ──────────────────────▶ │  (intermediate)     │ ──────────────────────▶ │  Nano-Claw-Code     │
│  TypeScript         │     4 service groups    │  TypeScript         │          Python         │  Python             │
│  ~405,500 lines     │      -27,400 lines      │  ~378,100 lines     │                         │  ~5,800 lines       │
│  42 tools           │                         │  13 tools           │                         │  12 tools           │
└─────────────────────┘                         └─────────────────────┘                         └─────────────────────┘
```

---

## 仓库结构

```
nano-claw-code/
├── nano_claw_code/            # Agent 源码
│   ├── cli.py                 #   交互式 REPL、CLI、启动界面（1,639 行）
│   ├── tools_impl.py          #   12 个核心工具实现（1,066 行）
│   ├── agent.py               #   Agent 循环、压缩、Prompt 缓存、重试（659 行）
│   ├── openai_compat.py       #   OpenAI 兼容 API 适配器（599 行）
│   ├── agents.py              #   子 Agent 配置 & 自定义 Agent 加载（302 行）
│   ├── skills.py              #   技能发现 & 执行（294 行）
│   ├── config.py              #   配置管理（279 行）
│   ├── session.py             #   会话持久化（233 行）
│   ├── prompts.py             #   系统提示词（189 行）
│   ├── stream_json.py         #   Stream-JSON 输出协议（185 行）
│   ├── frontmatter.py         #   CLAUDE.md 前置解析（137 行）
│   ├── permissions.py         #   权限处理（133 行）
│   └── memory.py              #   记忆管理（111 行）
├── swebench_harness/          # SWE-bench 评测框架
│   ├── run_swebench_claude_code.py  # 主评测脚本（推理 + 评测）
│   ├── run.sh                 #   一键启动（安装、预测、评测）
│   ├── compare_results.py     #   跨版本结果对比
│   ├── requirements.txt       #   评测依赖（datasets、swebench）
│   ├── instance_ids_pilot_8.txt   # 8 实例试点子集
│   ├── instance_ids_full_50.txt   # 50 实例子集
│   └── results/               #   预测结果 & 评测报告
├── start.sh                   # 启动脚本（封装 CLI）
├── pyproject.toml             # Python 包配置
├── uv.lock                    # 锁定依赖（供 uv 使用）
├── .env.example               # API / 模型环境变量示例
└── assets/                    # 截图 & 图片
```

---

## 安装

### 前置依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| **Python** | >= 3.10 | Agent 运行时 |
| **Docker** | latest | SWE-bench 测试执行（可选） |

### 第 1 步 — 安装

```bash
pip install -e .
```

#### 用 uv 安装

[uv](https://docs.astral.sh/uv/) 会根据 `pyproject.toml` 与仓库中的 `uv.lock` 安装依赖，环境可复现：

```bash
uv sync                    # 仅运行时依赖
uv sync --extra dev        # 额外包含 pytest、ruff（开发）
```

会在项目根目录创建 `.venv/`（已加入 `.gitignore`）。可用 `uv run nano-claw-code …`、`uv run pytest`，或激活虚拟环境后照常使用 `./start.sh`。

可选安装 Rich 终端样式：

```bash
uv sync --extra dev --extra rich
```

若修改了 `pyproject.toml` 里的依赖，请执行 `uv lock` 并将更新后的 `uv.lock` 一并提交。

### 第 2 步 — 配置 API

可将 [`.env.example`](.env.example) 复制为 `.env` 并编辑（会沿项目目录自动加载），或在 shell 中 `export`：

```bash
# 方式 A：直接使用 Anthropic API
export ANTHROPIC_API_KEY="sk-ant-xxx"

# 方式 B：通过 OpenRouter（支持 Kimi、MiniMax 等）
export OPENROUTER_API_KEY="sk-or-xxx"
export OPENROUTER_MODEL="moonshotai/kimi-k2"

# 方式 C：通过 LiteLLM Proxy
export ANTHROPIC_BASE_URL="http://127.0.0.1:4000"
export ANTHROPIC_API_KEY="sk-anything"
export MODEL="moonshotai/kimi-k2"
```

### 第 3 步 — 运行

[`start.sh`](start.sh) 与安装后的控制台命令 **`nano-claw-code`** 指向同一入口：

```bash
./start.sh
# 等价：
nano-claw-code
```

### 开发

```bash
pytest                              # 默认仅单元测试（不发 API）
# 须先清空 addopts，否则默认的 -m 过滤仍会生效：
pytest --override-ini addopts= -m e2e
pytest --override-ini addopts= -m integration
pytest --override-ini addopts= -m "integration or e2e"
# 一次性跑完全部用例：
pytest --override-ini addopts=
```

E2e / integration 需要 `ANTHROPIC_API_KEY`（`sk-ant-*`），仅 `test_e2e_cli_version`（测 `--version`）例外。若用 uv，可写 `uv run pytest …`。

---

## 使用

### 单次提问

```bash
./start.sh -p "解释这个代码库"
# 或：nano-claw-code -p "解释这个代码库"
```

### 通过 OpenRouter 使用第三方模型

```bash
export OPENROUTER_API_KEY="sk-or-xxx"
export OPENROUTER_MODEL="moonshotai/kimi-k2"
./start.sh
```

也可以通过 [LiteLLM Proxy](https://docs.litellm.ai/) 统一管理供应商：

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:4000"
export ANTHROPIC_API_KEY="sk-anything"
export MODEL="moonshotai/kimi-k2"
```

---

## SWE-bench 评测

仓库内置了完整的评测框架 `swebench_harness/`，支持**推理**（生成补丁）和**评测**（运行 SWE-bench 评分）。

### 前置条件

```bash
pip install -e .                          # 安装 nano-claw-code
pip install -r swebench_harness/requirements.txt  # 评测依赖（datasets、swebench）
# 使用 uv 时：
# uv pip install -e . && uv pip install -r swebench_harness/requirements.txt
```

Docker 必须运行中 — SWE-bench 使用 Docker 容器执行和评分补丁。

### 快速开始（一条命令）

```bash
cd swebench_harness
./run.sh --max-instances 10
```

这条命令会：
1. 自动安装 `nano-claw-code`（如尚未安装）
2. 在 SWE-bench Lite 实例上生成预测
3. 运行 SWE-bench 评测并生成 JSON 报告

### 分步执行

**第 1 步 — 生成预测：**

```bash
cd swebench_harness

# 运行前 N 个实例
python run_swebench_claude_code.py --max-instances 10

# 运行指定子集
python run_swebench_claude_code.py --instance-ids instance_ids_pilot_8.txt

# 从指定实例恢复
python run_swebench_claude_code.py --resume-from django__django-11099
```

预测结果保存在 `results/nano-claw-code/predictions.jsonl`，完整 trace（工具调用、模型回复、思考过程）保存在 `results/nano-claw-code/traces/`。

**第 2 步 — 评测预测结果：**

```bash
python run_swebench_claude_code.py --evaluate
```

运行官方 SWE-bench Docker 评测，生成 JSON 报告（如 `claude-sonnet-4-20250514.nano-claw-code-swebench.json`）。

**第 3 步 — 查看结果：**

```bash
# 摘要输出到终端；详细报告在 JSON 文件中
cat claude-sonnet-4-20250514.nano-claw-code-swebench.json | python -m json.tool
```

### 配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--max-instances N` | 限制评测实例数 | 全部 |
| `--instance-ids FILE` | 指定实例 ID 列表文件 | — |
| `--model MODEL` | 使用的模型 | `claude-sonnet-4-20250514` |
| `--dataset DATASET` | SWE-bench 数据集 | `princeton-nlp/SWE-bench_Lite` |
| `--split SPLIT` | 数据集切分 | `test` |
| `--max-turns N` | 每个实例最大 Agent 轮次 | 30 |
| `--resume-from ID` | 从指定实例恢复 | — |
| `--evaluate` | 仅运行评测（跳过推理） | — |
| `--predictions FILE` | 自定义预测文件（用于评测） | 自动检测 |
| `--bare` | 跳过 hooks/LSP 加速推理 | — |
| `-v, --verbose` | 启用调试日志 | — |

### 使用 OpenRouter / LiteLLM

```bash
export OPENROUTER_API_KEY="sk-or-xxx"
export OPENROUTER_MODEL="moonshotai/kimi-k2"
cd swebench_harness && ./run.sh --max-instances 5
```

---

## 许可

本项目采用 **GPL-3.0 + AGPL-3.0** 许可证。详见 [LICENSE](https://github.com/OpenLAIR/dr-claw/blob/main/LICENSE)。

本项目基于 Anthropic 的 [Claude Code](https://github.com/anthropics/claude-code) 构建。

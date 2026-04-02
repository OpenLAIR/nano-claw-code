<p align="center">
  <h1 align="center">Nano-Claw-Code</h1>
  <p align="center">
    <em>蒸馏后的编程智能体 — 更少工具，同等性能，~5,800 行 Python。</em>
  </p>
  <p align="center">
    <a href="README.md">English</a> | 中文
  </p>
</p>

---

## 这是什么？

Nano-Claw-Code 是一个从完整版 [Claude Code](https://github.com/anthropics/claude-code) 框架中**系统性蒸馏**出来的轻量级 Python 编程智能体。蒸馏分两步：

1. **TypeScript 裁剪** — 分析 SWE-bench 上的工具使用情况，从原始 Claude Code 中移除 29 个未使用的工具和 4 组服务（~405,500 → ~378,100 核心代码行）。
2. **Python 重写** — 将核心 Agent 循环、工具实现和 CLI 用纯 Python 重写，把 ~378,100 行 TypeScript 压缩为 **~5,800 行 Python**，同时保留相同的工具调用接口和 Agent 能力。

最终结果在 [SWE-bench Lite](https://www.swebench.com/) 上**达到**完整版 TypeScript Agent 的表现。

---

## 路线图

- [x] 从 Claude Code 蒸馏（42 → 13 工具，TypeScript 裁剪）
- [x] Python 重写 — nano-claw-code（~5,800 行，12 工具）
- [x] SWE-bench 评测框架（含完整 trace 日志）
- [x] SWE-bench Lite 对比评测（40/300 实例）
- [ ] SWE-bench Lite 全量运行（300 实例）
- [ ] SWE-bench Verified 运行（500 实例）
- [ ] 通过 OpenRouter 评测第三方模型（Kimi、MiniMax）
- [ ] 从 Agent trace 构建蒸馏数据集

---

## 主要结果

在 SWE-bench Lite 前 40 个实例上评测，模型为 `claude-sonnet-4-20250514`：

| 版本 | 语言 | 工具数 | 核心代码行数 | 提交 | 解决 | 解决率 |
|------|------|--------|------------|------|------|--------|
| **Claude Code**（完整版） | TypeScript | 42 | ~405,500 | 38 | 22 | 57.9% |
| **Nano-Claw-Code**（本仓库） | Python | 12 | **~5,800** | 38 | 22 | 57.9% |

> 代码量减少 ~70 倍，解决率完全一致。全量测试（300 实例）正在进行中。

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

除工具外，Agent 还保留了完整版的关键基础设施：

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

### 3. SWE-bench 对比评测

两个版本在相同条件下评测，评测框架记录完整的 trace 日志 — 包括每个工具调用、模型回复和思考过程。

---

## 蒸馏流程

```
┌─────────────────────┐     裁剪 29 个工具      ┌─────────────────────┐    Python 重写     ┌─────────────────────┐
│  Claude Code         │  ──────────────────▶   │  （中间版本）         │  ──────────────▶  │   Nano-Claw-Code    │
│  TypeScript          │     4 组服务            │  TypeScript          │                    │   Python            │
│  ~405,500 行         │     -27,400 行          │  ~378,100 行         │                    │   ~5,800 行         │
│  42 个工具           │                         │  13 个工具           │                    │   12 个工具         │
└─────────────────────┘                          └─────────────────────┘                    └─────────────────────┘
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
├── start.sh                   # 启动脚本
├── pyproject.toml             # Python 包配置
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

### 第 2 步 — 配置 API

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

```bash
./start.sh
```

---

## 使用

### 交互模式

```bash
./start.sh
```

### 单次提问

```bash
./start.sh -p "解释这个代码库"
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

## 许可

本项目基于 Anthropic 的 [Claude Code](https://github.com/anthropics/claude-code) 构建。

# 30day-agent

30 天 AI Agent 学习计划实践项目，包含 LangGraph 学习笔记和一个基于 LangGraph 构建的 code agent（LangCode）。

## 项目结构

```
30day-agent/
├── src/                  # Jupyter 学习笔记本
│   ├── day1-2.ipynb      # LangGraph 基础（状态、节点、边、HITL）
│   └── day3.ipynb        # 工具调用与 Pydantic schema
├── LangCode/             # 基于 LangGraph 的 code agent（Git 子模块）
│   └── src/LangCode/
│       ├── main.py       # 程序入口
│       ├── shared/       # 共享模块（state、llm、config、tools）
│       └── agents/       # agent 实现（supervisor 等）
└── doc/                  # 学习计划文档
```

## 快速开始

```bash
# 安装依赖
uv sync

# 运行 LangCode agent
uv run python LangCode/src/LangCode/main.py

# 运行学习笔记本
uv run jupyter notebook src/day1-2.ipynb
```

## LangCode Agent

基于 LangGraph 构建的 code agent，提供以下工具能力：

| 工具 | 功能 |
|------|------|
| `read_file` | 读取文件内容 |
| `fetch_api` | 异步 HTTP GET 请求 |
| `execute_shell` | 执行 shell 命令 |
| `run_python` | 沙箱内执行 Python 代码 |

通过环境变量配置 LLM：

```bash
export LC_MODEL_NAME="your-model"
export LC_API_KEY="your-api-key"
export LC_BASE_URL="https://your-api-endpoint/v1"
```

## 学习计划

详见 [doc/learning_plan_detail.md](doc/learning_plan_detail.md)，覆盖 LangGraph、RAG、LoRA 微调、vLLM 等主题。

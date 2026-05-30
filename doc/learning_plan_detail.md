# 30天 AI Agent 学习计划 — 详细学习内容

---

## Week 1：从LangChain升级到工业级Agent（Day 1–7）

### Day 1–2：LangGraph 状态机深入

**学习内容**

1. **StateGraph vs Chain 的本质区别**
   - Chain 是线性的 `A → B → C`，无法表达分支、循环、并行
   - StateGraph 是有向图，节点是处理函数，边是流转规则，State 是全局共享数据
   - 类比：Chain 像流水线，StateGraph 像地铁线路图——你可以换乘、回路、跳站

2. **核心三要素**
   - **Node**：一个 Python 函数，接收 State，返回更新后的 State
   - **Edge**：连接 Node 的有向边，可以是普通边或条件边
   - **State**：TypedDict 定义的全局状态容器，所有 Node 共享读写

3. **条件边（Conditional Edge）**
   - 在边上绑定一个路由函数，根据当前 State 决定下一个 Node
   - 典型场景：LLM 输出是否包含工具调用？是→执行工具，否→返回结果
   - 实现：`add_conditional_edges(source_node, routing_fn, {path_map})`

4. **Human-in-the-loop**
   - `interrupt_before` / `interrupt_after`：在指定节点前/后暂停，等待人工输入
   - 典型场景：Agent 要执行危险操作（删除文件、发邮件）前暂停确认
   - 用 `Checkpointer` 保存暂停时的状态，恢复时从断点继续

5. **LangGraph 的持久化（Persistence）**
   - MemorySaver / SqliteSaver：将 State 持久化到内存或数据库
   - 支持 Time Travel：回退到任意历史状态重新执行
   - 对话多轮时必须用持久化，否则 Agent 失忆

**动手任务**

- 把之前用 LangChain 写的任意 Agent（如简单问答 Agent）用 LangGraph 重写
- 实现条件边：工具调用失败→自动重试→超过 3 次→报错退出
- 代码骨架参考：
  ```python
  from langgraph.graph import StateGraph, END
  from typing import TypedDict, Annotated

  class AgentState(TypedDict):
      messages: list
      retry_count: int

  def call_llm(state): ...
  def call_tool(state): ...
  def should_retry(state): return state["retry_count"] < 3

  graph = StateGraph(AgentState)
  graph.add_node("llm", call_llm)
  graph.add_node("tool", call_tool)
  graph.add_edge("llm", "tool")
  graph.add_conditional_edges("tool", should_retry, {True: "llm", False: END})
  ```

**给你的提示**
你有 LangChain 基础，迁移成本很低。重点感受 StateGraph 在处理复杂分支时比 Chain 清晰多少。

---

### Day 3：Tool Calling 鲁棒性

**学习内容**

1. **用 Pydantic 定义严格的 JSON Schema**
   - 工具的输入/输出必须有类型约束，不能让 LLM 自由发挥
   - Pydantic 的 `Field(description=...)` 会自动转为 JSON Schema 注入 Prompt
   - 示例：
     ```python
     from pydantic import BaseModel, Field

     class FileReadInput(BaseModel):
         path: str = Field(description="文件的绝对路径")
         encoding: str = Field(default="utf-8", description="文件编码")
     ```

2. **结构化输出方案对比**
   - **Function Calling**（推荐）：OpenAI / Anthropic 原生支持，最稳定
   - **JSON Mode**：强制输出合法 JSON，但不保证字段完整
   - **Instructor 库**：用 Pydantic 模型约束 LLM 输出，自动重试
   - **正则提取**：最后手段，脆弱但万能

3. **优雅降级策略**
   - 超时：设置 30s 超时，超时后返回错误信息让 LLM 决定下一步
   - 格式错误：自动检测 JSON parse 失败，触发重新生成（最多 2 次）
   - 参数缺失：检查必填字段，缺失时让 LLM 补充而不是直接报错
   - 工具不存在：返回可用工具列表，引导 LLM 选择正确的工具

4. **中间件设计模式**
   - 在工具调用前后各加一层拦截器
   - 前置拦截：参数校验、权限检查
   - 后置拦截：结果格式化、错误分类、日志记录

**动手任务**

- 写一个 Tool Call 中间件：自动解析结果，格式错误时触发重新生成，最多重试 2 次
- 实现三个基础工具：`file_read`、`shell_exec`、`python_run`
- 每个工具都要有 Pydantic Schema 约束

**给你的提示**
这是 Agent 面试必考点。面试官最喜欢问："你的 Agent 调工具失败了怎么办？"

---

### Day 4：MCP 协议入门

**学习内容**

1. **MCP 是什么**
   - Model Context Protocol，Anthropic 推出的 Agent 工具协议标准
   - 目标：统一 LLM 与外部工具/数据源的通信方式，类似 USB 之于外设
   - 核心思想：把工具、资源、Prompt 模板标准化为可发现、可调用的服务

2. **Client/Server 架构**
   - **MCP Client**：集成在 LLM 应用中，负责发起工具调用请求
   - **MCP Server**：暴露工具/资源的服务端，可以是本地进程或远程服务
   - 通信方式：stdio（本地进程）或 SSE（远程 HTTP）
   - 工具定义：Server 用 JSON Schema 描述每个工具的名称、参数、功能

3. **MCP vs LangChain Tool**
   - LangChain Tool 是代码级别的函数绑定，与 LangChain 生态强耦合
   - MCP 是协议级别的标准，任何 LLM 框架都能接入
   - MCP 支持动态发现：Client 启动时自动获取 Server 暴露的工具列表
   - MCP 支持 Resources（数据源）和 Prompts（模板），不只是工具

4. **MCP 的三大原语**
   - **Tools**：可调用的函数（如文件读写、数据库查询）
   - **Resources**：可读取的数据源（如文件内容、数据库记录）
   - **Prompts**：预定义的 Prompt 模板，可带参数

**动手任务**

- 跑通官方 filesystem MCP 示例
- 尝试接入一个 git MCP server，让 Agent 能读取代码仓库信息
- 理解 MCP Server 的工具注册流程

**给你的提示**
MCP 正在成为大厂 Agent 基础设施的标准协议，写在简历上很加分。1 天入门足够。

---

### Day 5–6：Context 与 Memory 管理

**学习内容**

1. **Context Window 的硬限制**
   - 每个模型有最大 Token 数（如 GPT-4: 128K, Claude: 200K）
   - "Lost in the Middle" 现象：模型对上下文中间位置的信息关注度最低
   - 实际影响：对话超过 20 轮后，早期信息会被"遗忘"或混淆

2. **Memory 方案对比**
   - **滑动窗口**：只保留最近 N 轮对话，简单但丢失历史信息
   - **摘要压缩**：每 N 轮让 LLM 总结历史，用摘要替代原始对话
   - **向量存储**：将历史对话 Embedding 存入向量数据库，按语义检索
   - **混合方案**（推荐）：近期用滑动窗口 + 远期用向量检索

3. **Multi-turn 状态持久化**
   - 不只是保存对话历史，还要保存 Agent 的中间状态
   - LangGraph 的 Checkpointer 可以自动持久化整个 State
   - 需要考虑：并发对话隔离、状态过期清理、跨设备同步

4. **Prompt 设计中的 Memory 利用**
   - 每次思考时先回顾关键历史信息的 Prompt 模板：
     ```
     以下是本次对话的关键信息摘要：
     {memory_summary}

     最近 3 轮对话：
     {recent_messages}

     请基于以上信息回答用户的问题。
     ```
   - 让 Agent 自己判断哪些信息值得记住

**动手任务**

- 给 Week 1 的 Agent 加入 Memory 模块：对话超过 N 轮时自动压缩历史
- 设计一个 Prompt：让 Agent 每次思考时先回顾关键历史信息
- 测试：连续对话 30 轮，观察 Agent 是否还记得第 1 轮的内容

**给你的提示**
很多人做 Agent 只管当前轮，完全不考虑长对话退化。加上 Memory 设计，简历立刻高一档。

---

### Day 7：整合与复盘

**学习内容**

1. **回顾本周知识组合**
   - StateGraph 管理流程 → Tool Calling 保证鲁棒性 → MCP 标准化工具 → Memory 维持长对话
   - 这四者组合才是一个"工业级"Agent 的基础

2. **思考 Agent 的边界与失败模式**
   - LLM 幻觉导致调用不存在的工具
   - 无限循环：Agent 反复执行同一个失败的操作
   - 上下文溢出：长对话后 State 膨胀超出 Token 限制
   - 权限问题：Agent 尝试执行无权限的操作

3. **Agent 设计的核心原则**
   - 最小权限：Agent 只能访问必要的工具和数据
   - 可观测性：每一步都要有日志，失败时能回溯原因
   - 优雅降级：主路径失败时有备选方案

**动手任务**

- 整合本周所有组件，跑通一个完整的多工具 Agent
- 在 GitHub 建好项目仓库，写好基础 README
- 记录：Agent 能成功完成哪些任务？在哪些情况下会失败？

**给你的提示**
今天不要学新东西。把本周写的代码整理好，这是你简历项目的地基。

---

### Week 1 推荐资料

**LangGraph**：直接看官方文档 `docs.langchain.com/oss/python/langgraph/overview`，重点看 Concepts 里的 State、Nodes、Edges、Memory 四节，然后看 Guides 里的 `Use the graph API`、`Human-in-the-loop`、`Multi-agent`、`MCP`。

> **注意**：当前文档处于新旧交替期，v1.0 里 `create_react_agent` 预置组件已被废弃，推荐改用 LangChain 的 `create_agent`。以 `docs.langchain.com` 下的新版为准。

**Tool Calling / Structured Output**：`docs.langchain.com/oss/python/langchain/tools`，在同一个文档站左侧导航里。概念讲解可参考 OpenAI 官方文档的 Function Calling 章节（即使你不用 OpenAI，这是最清晰的），再看 Pydantic 的 `model_validator` 和 `field_validator` 文档。

**MCP**：`modelcontextprotocol.io` 官网的 Introduction + Quickstart，一个下午能读完。

---

### Week 1 周末里程碑

> 手写一个含状态流转的命令行 Agent，支持工具调用失败后自动重试，能跑通"思考→调用→检查→纠错"完整闭环

---

## Week 2：高级RAG + 底层原理护城河（Day 8–14）

### Day 8–9：高级 RAG 架构

**学习内容**

1. **Chunking 策略详解**
   - **固定分块**（如 512 Token）：简单但会切断句子/段落，破坏语义完整性
   - **Sentence-level 分块**：按句子切分，保证语义完整，但块大小不均匀
   - **Recursive 分块**（LangChain 默认）：先按段落，再按句子，最后按字符
   - **语义分块**：用 Embedding 相似度判断切分点，相似度骤降的地方切开

2. **Parent-Child Chunking**
   - 核心思想：检索时用小块（Child），返回时用大块（Parent）
   - Child Chunk：50-100 Token，Embedding 语义精准，提高召回率
   - Parent Chunk：500-1000 Token，包含完整上下文，给 LLM 足够信息
   - 实现：索引时同时存储 Parent 和 Child，检索命中 Child 后返回对应的 Parent

3. **Sentence Window Retrieval**
   - 检索时用单句 Embedding，返回时带上前后 N 句
   - 比 Parent-Child 更细粒度，适合长文档中的精确定位
   - 窗口大小的选择：太大→噪声多，太小→上下文不足，通常 3-5 句

4. **元数据过滤**
   - 给每个 Chunk 附加元数据：来源文件、页码、章节标题
   - 检索时可以先按元数据过滤，再做语义搜索
   - 示例：只搜索"2024年"的文档，或只搜索"技术文档"类别

**动手任务**

- 用 LlamaIndex 实现一个带 Parent-Child 分块的本地知识库
- 对比：普通分块 vs Parent-Child，在长文档问答上的效果差异
- 准备 10 个测试问题，记录两种方案的回答质量

**给你的提示**
面试问"你怎么做 RAG"时，大多数人只说向量检索。你能说出分块策略的取舍，直接拉开差距。

---

### Day 10–11：Rerank 与混合检索

**学习内容**

1. **Bi-Encoder vs Cross-Encoder**
   - **Bi-Encoder**（向量检索）：Query 和 Document 分别编码，计算余弦相似度
     - 优点：速度快（可预计算 Document 向量），适合大规模检索
     - 缺点：Query 和 Document 独立编码，无法捕捉交互信息
   - **Cross-Encoder**（Rerank）：Query 和 Document 拼接后一起编码
     - 优点：能捕捉 Query-Document 交互，精度远高于 Bi-Encoder
     - 缺点：速度慢（每个 Pair 都要过模型），不适合大规模检索

2. **Two-Stage 检索架构**
   - Stage 1（召回）：用 Bi-Encoder 快速从百万文档中检索 Top-100
   - Stage 2（精排）：用 Cross-Encoder 对 Top-100 重新打分，取 Top-5
   - 这就是搜索引擎的经典架构：粗排→精排

3. **BM25 + 向量的 Hybrid 检索**
   - BM25：基于词频的经典稀疏检索，擅长精确关键词匹配
   - 向量检索：基于语义的稠密检索，擅长语义相似但词不同的情况
   - 融合方式：
     - 分数融合：`final_score = α * bm25_score + (1-α) * vector_score`
     - RRF（Reciprocal Rank Fusion）：`score = Σ 1/(k + rank_i)`

4. **常用 Rerank 模型**
   - BGE-Reranker（BAAI）：中文效果好，开源免费
   - Cohere Rerank：API 服务，效果好但要付费
   - bge-reranker-v2-m3：支持多语言，当前最优开源方案

**动手任务**

- 集成 BGE-Reranker 到你的 RAG 系统，实现 Two-Stage 检索
- 测试：10 个问题，对比加 Rerank 前后的答案质量
- 记录 Recall@K 和 MRR 指标的变化

**给你的提示**
Rerank 是 RAG 提质最有效的单点优化。跑通过一次，面试就能说出真实的效果数字。

---

### Day 12：vLLM 推理加速原理

**学习内容**

1. **KV Cache 是什么**
   - Transformer 推理时，每个 Token 的 Key 和 Value 需要被缓存
   - 因为生成新 Token 时需要和所有历史 Token 的 K/V 做 Attention
   - KV Cache 避免了重复计算，但占用大量显存

2. **为什么推理是 Memory-bound**
   - 生成阶段：每步只生成 1 个 Token，但要读取整个 KV Cache
   - 计算量小（矩阵-向量乘法），但显存带宽是瓶颈
   - 这就是为什么 GPU 算力利用率只有 5-10%

3. **PagedAttention**
   - 核心思想：借鉴 OS 虚拟内存的分页机制
   - 传统方式：每个请求预分配最大长度的连续显存 → 大量碎片浪费
   - PagedAttention：将 KV Cache 分成固定大小的"页"，按需分配，非连续存储
   - 效果：显存利用率从 50-60% 提升到 90%+

4. **Continuous Batching**
   - 传统 Static Batching：一个 batch 中所有请求必须同时开始、同时结束
   - 问题：短请求要等长请求完成，GPU 空闲等待
   - Continuous Batching：请求完成后立即插入新请求，GPU 持续满载
   - 又叫 Iteration-level Scheduling

5. **KV Cache 空间计算公式**
   ```
   KV Cache 大小 = 2 × num_layers × num_heads × head_dim × seq_len × dtype_bytes
   ```
   - 2：Key 和 Value 各一份
   - dtype_bytes：FP16=2, FP8=1, INT4=0.5
   - 示例：Qwen2.5-7B，seq_len=4096，FP16：
     ```
     2 × 28 × 28 × 128 × 4096 × 2 = 1.64 GB
     ```

**动手任务**

- 在本地用 vLLM 启动一个小模型（如 Qwen-1.5B），观察并发吞吐变化
- 能在纸上写出 KV Cache 的空间计算过程
- 对比：vLLM vs 原生 HuggingFace 推理的吞吐差异

**给你的提示**
你不需要会写 vLLM 源码。但面试官问原理时，你能讲出"分页"类比 OS 的底层逻辑，含金量极高。

---

### Day 13：Ollama + 本地模型部署

**学习内容**

1. **GGUF 量化格式**
   - GGML → GGUF：llama.cpp 的模型格式，支持 CPU 和 GPU 混合推理
   - 量化级别：Q2_K（最小，质量差）→ Q4_K_M（平衡）→ Q8_0（接近 FP16）
   - Q4_K_M 通常是最佳选择：模型大小减半，质量损失很小

2. **量化的基本原理**
   - 将 FP16（16bit）权重压缩为 INT4（4bit）或 INT8（8bit）
   - 需要校准数据集来确定量化参数
   - 精度损失主要来自：异常值处理、分组量化误差

3. **Ollama 的模型管理**
   - `ollama pull qwen2.5:7b`：下载模型
   - `ollama list`：查看已下载的模型
   - `ollama run qwen2.5:7b`：启动交互式对话
   - API 端点：`http://localhost:11434/api/generate`

4. **本地部署的实际意义**
   - 面试时证明你"真的动手做过"，不只是 API 调用者
   - 了解推理延迟的真实瓶颈：CPU vs GPU、量化级别、上下文长度
   - 能回答"为什么不直接用 API"：隐私、成本、离线场景

**动手任务**

- 用 Ollama 在本地跑 Qwen2.5-7B 或 LLaMA3-8B
- 把你的 RAG 系统的 LLM 部分替换成本地 Ollama 模型，跑通端到端
- 测试不同量化级别的推理速度和回答质量

**给你的提示**
本地部署能力在面试中证明你"真的动手做过"，不只是 API 调用者。

---

### Day 14：Week 2 整合

**学习内容**

1. **RAG 系统评测方案**
   - **Recall@K**：在 Top-K 个检索结果中，包含正确答案的比例
   - **MRR（Mean Reciprocal Rank）**：正确答案排名的倒数的均值
   - **Answer Correctness**：生成答案与标准答案的匹配度
   - 评测数据集准备：手动构造 50 个 QA Pair，覆盖常见问题和边界情况

2. **完整 RAG 流程串联**
   ```
   文档 → 分块（Parent-Child）→ Embedding → 向量存储
                                                    ↓
   用户 Query → Embedding → 向量检索 + BM25 → Rerank → Top-K Context
                                                              ↓
                                               LLM 生成答案 ← Context
   ```

3. **常见 RAG 问题与优化方向**
   - 检索不到相关内容 → 优化分块策略、增加 BM25
   - 检索到但答案不准 → 加 Rerank、优化 Prompt
   - 答案有幻觉 → 强制引用来源、降低温度
   - 延迟太高 → 用更小的模型、减少 Top-K

**动手任务**

- 完成一个完整的 RAG 系统：支持 Parent-Child 分块、向量+BM25 混合检索、Rerank、本地模型生成
- 用 10 个测试问题跑一遍，记录结果，写进 README
- 计算 Recall@K 和 MRR 指标

**给你的提示**
今天的产出就是你简历项目二。哪怕还不完善，有量化结果就比"搭了个 RAG 聊天框"高一个量级。

---

### Week 2 推荐资料

**高级 RAG**：LlamaIndex 官方文档 `docs.llamaindex.ai`，左侧找 Understanding → RAG → Advanced RAG 章节，Parent-Child 和 Sentence Window 都在里面。Reranker 部分看 `huggingface.co/BAAI/bge-reranker-v2-m3` 的模型卡，里面有使用示例。

**vLLM 原理**：不要看文档，看两个东西。第一个是 vLLM 原始论文 "Efficient Memory Management for Large Language Model Serving with PagedAttention"（arxiv 搜 "PagedAttention vLLM"），只需要读 Abstract + Section 3（核心机制），大概 20 分钟。第二个是 vLLM 官方博客 `blog.vllm.ai/2023/06/20/vllm.html`，比论文更好懂，配图清晰。这两个看完，PagedAttention 的面试八股就掌握了。

**Ollama**：直接看 `ollama.com` 官网的 README 和 API 文档，不需要其他资料。

---

### Week 2 周末里程碑

> 搭出带 Rerank 的本地知识库系统，且能口述 PagedAttention 原理与 KV Cache 空间计算公式（面试加分项）

---

## Week 3：核心项目：硬核 Coding Agent（Day 15–21）

### Day 15–16：Coding Agent 架构设计

**学习内容**

1. **Planner-Executor-Reviewer 三角架构**
   - **Planner**：接收用户任务，分解为可执行的子任务列表
   - **Executor**：逐个执行子任务，调用工具完成具体操作
   - **Reviewer**：检查执行结果，判断是否符合预期，决定是否需要重做
   - 三者形成闭环：Planner → Executor → Reviewer → Planner（如果需要修正）

2. **为什么需要多 Agent**
   - 单 Agent 要同时负责规划、执行、检查，Prompt 膨胀且角色混乱
   - 多 Agent 各司其职：Planner 的 Prompt 专注于任务分解，Executor 专注于工具调用
   - 类比：项目经理（Planner）+ 开发者（Executor）+ 测试（Reviewer）

3. **工具集设计规范**
   ```python
   # 文件读取
   def file_read(path: str, line_range: str = None) -> str: ...

   # 文件编辑（Search-Replace）
   def file_edit(path: str, old_code: str, new_code: str) -> str: ...

   # Shell 执行
   def shell_exec(command: str, timeout: int = 30) -> dict: ...

   # 代码运行
   def code_run(code: str, language: str = "python") -> dict: ...
   ```
   - 每个工具都要有清晰的输入 Schema 和返回格式
   - 返回值包含：成功/失败标志、输出内容、错误信息

4. **状态机设计**
   ```python
   class CodingAgentState(TypedDict):
       task: str                    # 用户原始任务
       plan: list[str]              # 分解后的子任务列表
       current_step: int            # 当前执行到第几步
       files_modified: dict         # 修改过的文件内容
       execution_log: list          # 执行日志
       review_result: str           # 审查结果
       retry_count: int             # 当前步骤重试次数
   ```

**动手任务**

- 用 LangGraph 设计状态机：Planner 节点分解任务 → Executor 节点调工具 → Reviewer 节点检查结果
- 把状态机的图（用 mermaid 或手绘）画出来，这就是你以后面试讲的架构图
- 定义好每个节点的输入输出接口

**给你的提示**
先画图再写代码。架构清晰了，代码自然不乱。

---

### Day 17：文件编辑与 Diff 实现

**学习内容**

1. **为什么不能让模型输出完整文件**
   - Token 浪费：一个 500 行的文件，改 1 行也要输出 500 行
   - 错误率高：模型可能在不相关的地方引入错误
   - 成本高：输出 Token 比输入 Token 贵 3-5 倍

2. **Diff/Patch 方案**
   - Unified Diff 格式：`-` 表示删除，`+` 表示添加
   - 优点：只输出修改部分，Token 使用最少
   - 缺点：行号定位不精确，上下文匹配容易出错

3. **Search-Replace 方案（推荐）**
   - 模型输出：`<search>要查找的代码</search><replace>替换后的代码</replace>`
   - 优点：精确定位，不怕行号变化
   - 缺点：如果搜索内容有重复，可能匹配错误
   - Claude Code、Cursor 等主流 Coding Agent 都用类似方案

4. **实现细节**
   - 搜索时用精确字符串匹配，不要用正则
   - 如果匹配到多个位置，返回错误让模型更精确地指定
   - 修改后要验证文件语法是否正确（如 Python 的 `ast.parse`）

**动手任务**

- 实现一个 `file_edit` 工具：接收 search/replace 对，精准修改文件，失败时返回错误信息
- 测试：让 Agent 修改一个 Python 函数的实现，验证 diff 应用是否准确
- 处理边界情况：搜索内容不存在、匹配到多处、文件不存在

**给你的提示**
这是 Claude Code、Cursor 等 Coding Agent 的核心技术点。你实现过，面试时就能聊得很深。

---

### Day 18：Sandbox 执行隔离

**学习内容**

1. **为什么需要 Sandbox**
   - Agent 生成的代码可能执行危险操作：删除文件、发送网络请求、消耗大量资源
   - 没有隔离的代码执行等同于给 Agent root 权限

2. **subprocess 的安全使用**
   ```python
   import subprocess

   result = subprocess.run(
       ["python", "-c", code],
       capture_output=True,
       text=True,
       timeout=30,           # 超时限制
       cwd="/tmp/sandbox",   # 限制工作目录
       env={"PATH": "/usr/bin"},  # 限制环境变量
   )
   ```
   - `timeout`：防止死循环和长时间运行
   - `cwd`：限制文件访问范围
   - `capture_output`：捕获 stdout 和 stderr

3. **输出截断**
   - Agent 可能输出大量内容（如 print 循环），需要截断
   - 限制输出长度：超过 10000 字符时截断并提示

4. **进阶隔离方案**
   - Docker 容器：完整隔离，但启动慢
   - gVisor/Firecracker：轻量级 VM，安全性更高
   - seccomp/AppArmor：限制系统调用
   - 面试时说"考虑了安全隔离"就够了，不需要真的上 Docker

**动手任务**

- 实现一个安全的 `code_runner` 工具：支持超时、输出截断、错误捕获
- 让 Agent 能运行生成的代码并把 stdout/stderr 返回给自己
- 测试：故意让 Agent 运行死循环、访问不存在的文件，验证隔离效果

**给你的提示**
不需要上 Docker。用 subprocess 加好限制，面试说"考虑了安全隔离"就够了。

---

### Day 19–20：Reflection 与自我修正

**学习内容**

1. **Reflection 机制**
   - Agent 执行完一步后，不急着进入下一步，先"反思"结果质量
   - Prompt 模板：
     ```
     你刚刚执行了以下操作：{action}
     结果是：{result}
     请判断：
     1. 结果是否符合预期？
     2. 如果不符合，问题出在哪里？
     3. 应该如何修正？
     ```
   - Reflection 的输出决定下一步：继续、重做、还是换方案

2. **错误分类与修正策略**
   - **语法错误**：代码无法运行 → 直接把错误信息给 Executor 修复
   - **逻辑错误**：代码能运行但结果不对 → 需要 Planner 重新规划
   - **测试未通过**：运行测试用例失败 → 根据失败信息定位问题
   - **工具调用失败**：网络超时、权限不足 → 重试或更换工具

3. **防止无限循环**
   - 设置最大重试次数（如 3 次）
   - 设置最大总步数（如 20 步）
   - 检测重复行为：如果连续 3 次执行相同操作且失败，强制退出
   - 退出时输出当前状态和失败原因，方便人工介入

4. **Reviewer 节点的实现**
   ```python
   def reviewer(state):
       # 1. 运行测试用例
       test_result = run_tests(state["files_modified"])
       # 2. 如果测试通过，检查代码质量
       if test_result["passed"]:
           return {"review_result": "pass"}
       # 3. 如果测试失败，分析原因
       return {
           "review_result": "fail",
           "error_info": test_result["errors"],
           "retry_count": state["retry_count"] + 1
       }
   ```

**动手任务**

- 在 Reviewer 节点加入：运行测试用例 → 失败 → 把错误信息喂给 Planner 重新规划
- 测试：给一个有 bug 的需求，观察 Agent 能否自己发现并修复
- 实现重试上限和退出条件

**给你的提示**
这是区别于普通 Agent 项目最核心的功能。能自我修正的 Agent，才是真正的 Agent。

---

### Day 21：项目收尾与基础 UI

**学习内容**

1. **FastAPI 基础路由**
   ```python
   from fastapi import FastAPI
   from pydantic import BaseModel

   app = FastAPI()

   class TaskRequest(BaseModel):
       task: str
       max_steps: int = 20

   @app.post("/run")
   async def run_agent(req: TaskRequest):
       result = await agent.execute(req.task, req.max_steps)
       return {"status": "success", "log": result}
   ```

2. **Streamlit 最小前端**
   - 输入框：接收任务描述
   - 输出区：显示执行日志
   - 状态指示：正在执行 / 成功 / 失败
   - 代码量：约 30 行

3. **README 模板**
   ```
   # Coding Agent
   ## 简介
   基于 LangGraph 的自主编程 Agent...
   ## 架构图
   ![架构图](./docs/architecture.png)
   ## Quick Start
   pip install -r requirements.txt
   python main.py "写一个快速排序"
   ## Demo
   ![Demo](./docs/demo.gif)
   ```

**动手任务**

- 用 FastAPI 暴露一个 `/run` 接口，输入任务描述，输出执行日志
- 写好项目 README：背景、架构图、Quick Start、Demo 截图
- 录一个 10 秒的 GIF 演示主流程

**给你的提示**
UI 不用做好看，能跑就行。面试官看的是架构和逻辑，不是前端。

---

### Week 3 推荐资料

**Diff/Patch 实现**：不需要专门找资料，直接看 Python 标准库 `difflib` 的文档，然后去读 Aider（开源 Coding Agent）的源码里 `aider/coders/editblock_coder.py`，它实现的 search/replace 机制就是你要做的东西，代码很清晰。

**Reflection 机制**：看 LangGraph 文档里的 "Reflection" 示例，官方有完整的 notebook，直接跑一遍。

**多 Agent 架构**：看 LangGraph 的 "Multi-agent" 章节，重点是 Supervisor 模式那个示例。

---

### Week 3 周末里程碑

> 项目核心功能跑通，能演示"接收任务→分解→执行→反思修正"完整链路，准备好架构图和 README

---

## Week 4：微调实操 + 简历包装 + 海投（Day 22–30）

### Day 22–24：LoRA 微调实操

**学习内容**

1. **LoRA 原理**
   - 核心公式：`W = W₀ + B·A`
     - `W₀`：预训练权重（冻结，不更新）
     - `A`：低秩矩阵（d × r），`B`：低秩矩阵（r × d）
     - `r`（Rank）远小于 `d`，所以参数量大幅减少
   - 为什么有效：预训练模型的任务适应只需要低秩更新
   - 参数效率：7B 模型全量微调需要 ~28GB 显存，LoRA (r=16) 只需要 ~8GB

2. **Rank 和 Alpha 的含义**
   - **Rank（r）**：低秩矩阵的秩，控制表达能力
     - r=8：适合简单任务（格式调整、风格迁移）
     - r=16-32：适合复杂任务（工具调用、代码生成）
     - r=64+：接近全量微调，显存优势减小
   - **Alpha（α）**：缩放因子，控制 LoRA 更新的权重
     - 实际更新 = `(α/r) × B·A`
     - 通常设 α = 2×r，如 r=16, α=32

3. **SFT 的 Loss Mask**
   - 训练数据格式：`[Input] [Target]`
   - Loss 只对 Target 部分计算，Input 部分的 Loss 权重设为 0
   - 为什么：模型不需要"学习"输入，只需要学习如何从输入生成目标输出
   - 实现：在 labels 中将 Input 部分设为 -100（PyTorch 的 ignore_index）

4. **LLaMA-Factory 使用**
   - 一站式微调框架，支持 LoRA/QLoRA/全量微调
   - 配置文件方式：YAML 定义模型、数据、训练参数
   - 支持多种模型：LLaMA、Qwen、Mistral、ChatGLM 等

**动手任务**

- 在 AutoDL 租一张 RTX 4090（约 10-15 元/天）
- 用 LLaMA-Factory 跑通 Qwen2.5-7B 的 LoRA 微调
- 数据集：用你项目中工具调用的 bad case（格式错误、参数错误等）
- 跑完后对比微调前后 Tool Calling 的格式准确率

**给你的提示**
哪怕只跑通一次，面试时就能说："我在 4090 上微调过 Qwen2.5-7B，用了 200 条 bad case 数据，LoRA rank=16，工具调用准确率提升了 X%。"这和没跑过完全不是一个层次。

---

### Day 25–26：面试八股突击

**学习内容**

1. **Transformer 核心：Pre-LN vs Post-LN**
   - **Post-LN**（原始 Transformer）：LayerNorm 在残差连接之后
     - 问题：深层网络训练不稳定，需要 Warmup
   - **Pre-LN**（GPT-2 及之后）：LayerNorm 在残差连接之前
     - 优点：训练更稳定，不需要 Warmup，梯度流更顺畅
     - 现在几乎所有大模型都用 Pre-LN

2. **RoPE 旋转位置编码**
   - 核心思想：用旋转矩阵编码位置信息
   - 优势：
     - 相对位置编码：两个 Token 的 Attention 只取决于它们的相对距离
     - 天然支持外推：训练 4K 长度，推理时可以扩展到更长
   - 实现：对 Q 和 K 向量的每对相邻维度应用旋转矩阵
   - 为什么比绝对位置编码好：绝对位置编码无法泛化到训练时未见过的位置

3. **MoE 路由机制**
   - **Mixture of Experts**：每个 Token 只激活部分 Expert（如 8 个中选 2 个）
   - **Top-K Routing**：根据 Router 分数选择 Top-K 个 Expert
   - **负载均衡 Loss**：防止所有 Token 都路由到同一个 Expert
   - 优势：参数量大但计算量小（如 Mixtral 8x7B：47B 参数，但每个 Token 只用 13B）

4. **RLHF vs DPO**
   - **RLHF**：用人类偏好训练 Reward Model，再用 PPO 优化
     - 优点：理论上可以优化任意目标
     - 缺点：训练不稳定，需要额外的 Reward Model
   - **DPO**：直接从偏好数据优化，不需要 Reward Model
     - 优点：训练更稳定，实现更简单
     - 缺点：可能不如 RLHF 灵活

5. **Multi-Head Attention 矩阵维度变化**
   ```
   输入: [batch, seq_len, d_model]
   → Q = X·Wq: [batch, seq_len, d_model]
   → reshape: [batch, seq_len, num_heads, head_dim]
   → transpose: [batch, num_heads, seq_len, head_dim]
   → Attention: softmax(Q·K^T / √head_dim) · V
   → output: [batch, num_heads, seq_len, head_dim]
   → reshape: [batch, seq_len, d_model]
   → 输出: [batch, seq_len, d_model]
   ```

**动手任务**

- 找 GitHub 上"大模型面试题"相关仓库，把上面 4 个知识点能流畅口述
- 手推 Multi-Head Attention 的矩阵维度变化

**给你的提示**
你懂 Attention 原理，这部分不是从零学，是把已知的东西转化成面试语言。

---

### Day 27：简历算法化改写

**学习内容**

1. **工程视角 vs 算法视角**
   - 工程视角："用了 LangGraph 实现 Agent，支持工具调用和多轮对话"
   - 算法视角："基于状态机设计反思修正闭环，实现 Tool Calling 格式准确率从 72% 提升至 94%，通过 LoRA 轻量化微调优化工具选择策略"
   - 核心区别：算法视角有"做了什么→怎么做的→效果如何"

2. **量化结果的方法**
   - Tool Calling 准确率：微调前 72% → 微调后 94%
   - RAG 检索质量：Recall@5 从 0.65 提升到 0.82
   - 推理延迟：从 3.2s 降低到 1.8s
   - 系统吞吐：从 10 req/s 提升到 50 req/s

3. **简历核心技能区**
   ```
   - LangGraph 多智能体编排：状态机设计、条件路由、反思修正闭环
   - RAG 检索增强生成：Parent-Child 分块、混合检索、BGE-Rerank
   - LoRA 轻量化微调：基于 LLaMA-Factory 的 Qwen2.5 微调，工具调用准确率提升 22%
   - vLLM 推理优化：PagedAttention、Continuous Batching、KV Cache 管理
   ```

4. **项目描述模板**
   ```
   项目名称：基于 LangGraph 的自主编程 Agent
   技术栈：LangGraph, Qwen2.5, vLLM, LlamaIndex
   项目描述：
   - 设计 Planner-Executor-Reviewer 三节点状态机，实现任务分解→执行→反思修正的闭环
   - 实现 Search-Replace 文件编辑方案，支持精准代码修改，Token 消耗降低 60%
   - 集成 Sandbox 执行隔离，支持超时控制和输出截断，防止危险操作
   - 基于 LoRA 微调 Qwen2.5-7B，工具调用格式准确率从 72% 提升至 94%
   ```

**动手任务**

- 把 Coding Agent 项目的简历描述改成算法化语言
- 核心技能区加入：LangGraph 多智能体编排、LoRA 轻量化微调、vLLM 推理优化、RAG 检索重排

**给你的提示**
同一个项目，工程视角写是 60 分，算法视角写是 85 分。你这一天的时间 ROI 极高。

---

### Day 28–30：海投与模拟面试

**学习内容**

1. **目标岗位类型**
   - 大模型应用工程师：偏应用层，Agent/RAG 开发
   - AI Agent 研发：偏 Agent 架构设计
   - LLM 工程师：偏模型部署、推理优化
   - AI 平台工程师：偏基础设施、训练框架

2. **投递策略**
   - 北航学历 → 先投大厂实习直通车（字节、阿里、腾讯、百度）
   - 同时投独角兽：智谱、月之暗面、阶跃星辰、MiniMax
   - 不要只投一家，同时推进多个流程

3. **10 个必须准备的面试问题**
   - Tool Calling 失败怎么处理？→ 重试机制、格式校验、降级策略
   - RAG 效果差怎么优化？→ 分块策略、Rerank、混合检索、Prompt 优化
   - KV Cache 怎么计算？→ 公式 + 具体数值示例
   - LoRA 的原理是什么？→ 低秩分解、Rank/Alpha 的含义
   - Agent 怎么防止无限循环？→ 最大步数、重复检测、退出条件
   - 多 Agent 通信怎么设计？→ 消息传递、共享状态、事件驱动
   - 如何评估 RAG 系统？→ Recall@K、MRR、Answer Correctness
   - vLLM 的 PagedAttention 是什么？→ OS 分页类比
   - RLHF 和 DPO 的区别？→ 训练流程、优缺点对比
   - 你的项目中最大的技术挑战是什么？→ 准备一个具体案例

4. **模拟面试准备**
   - 自己对着录像练：控制语速、逻辑清晰
   - 准备"项目介绍"的 2 分钟版本
   - 准备"遇到过什么 bug，怎么解决的"的真实案例

**动手任务**

- 整理 10 个必须准备的面试问题并写好回答框架
- 投递不少于 15 个岗位
- 找人做一次模拟面试，或者自己对着录像练

**给你的提示**
简历投出去后不要停。继续完善项目，继续刷八股。第一个面试邀约可能比你想象的来得快。

---

### Week 4 推荐资料

**LoRA 原理**：直接读原论文 "LoRA: Low-Rank Adaptation of Large Language Models"，只需要读 Section 1（Introduction）和 Section 4（方法），两页纸，半小时搞定。数学部分就是 `W = W₀ + BA` 这一个公式，理解为什么可以用低秩矩阵近似即可。

**LLaMA-Factory 实操**：看它的 GitHub README 和 `examples/` 目录下的配置文件，不需要其他教程，README 写得非常详细。

**面试八股**：搜 GitHub，关键词 `LLM interview`，找 star 数最高的几个仓库，重点看 Transformer 结构、位置编码、KV Cache 计算这三块。不需要全背，把你简历上写的技术点对应的问题吃透就够了。

---

### Week 4 月末里程碑

> GitHub 有 1 个完整 Agent 项目，简历用"算法化语言"重写，已投递 10+ 岗位，准备好核心面试问题的回答框架

---

## 总原则

遇到不懂的概念，先搜官方文档，没有再搜论文，最后才搜博客。博客质量参差不齐，很多是复制粘贴，官方文档和论文是第一手资料。碰到具体的代码报错或实现细节，直接来问我比搜索快。

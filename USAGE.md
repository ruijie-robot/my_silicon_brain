# 投资研究支持系统使用指南

## 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 或使用uv (推荐)
uv sync
```

### 2. 配置API密钥

创建`.env`文件：
```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### 3. 启动系统

```bash
# 完整系统启动
python main.py

# 仅启动知识库监控
python main.py --knowledge-only

# 查看本地LLM配置
python main.py --check-llm

# 运行演示
python demo.py
```

## 功能介绍

### 📚 智能知识库

- **实时监控**: 自动监控`documents/`文件夹，文件变更时自动更新向量数据库
- **支持格式**: PDF, MD, TXT, DOCX, HTML
- **向量搜索**: 基于Milvus的高效相似度搜索
- **增量更新**: 只处理变更的文件，提高效率

**使用方法**:
1. 将投资相关文档放入`documents/`文件夹
2. 系统会自动处理并建立索引
3. 在对话中会自动搜索相关信息

### 🌐 网络搜索工具

**可用命令**:
- `web_search`: 通用网络搜索
- `search_finance_news`: 搜索金融新闻
- `search_stock_info`: 搜索股票信息
- `search_market_sector`: 搜索板块信息
- `get_webpage_content`: 获取网页内容

### 🤖 本地LLM支持

支持的模型服务:
- **Ollama** (推荐)
- **LM Studio**
- **Text Generation WebUI**

**推荐模型**:
- Qwen2.5-14B: 中文金融分析最佳
- Qwen2.5-7B: 平衡性能
- DeepSeek-Coder: 代码和逻辑分析

**安装Ollama**:
```bash
# macOS
brew install ollama

# 启动服务
ollama serve

# 下载模型
ollama pull qwen2.5:14b
```

## 核心功能

### 🏦 板块分析

**命令格式**: `sector <板块名>`

**示例**:
```
sector 新能源汽车
sector 医药生物
sector 人工智能
```

**功能**:
- 分析板块基本情况和发展趋势
- 识别板块内优质公司
- 结合用户经验推荐熟悉标的
- 提供风险提示和投资建议
- 获取最新市场信息和新闻

### 📊 交易复盘检查

**命令格式**: `review <内容>`

**示例**:
```
review 今天买入了比亚迪，卖出了宁德时代，感觉新能源板块还有上涨空间
review 本周复盘：持仓医药股表现不佳，考虑减仓转向科技股
```

**功能**:
- 分析交易记录和思考过程
- 检查是否遗漏重要投资机会
- 识别违背投资原则的操作
- 提供建设性改进建议
- 总结经验教训

### 💡 通用投研问答

**直接提问即可**，系统会：
- 搜索相关知识库内容
- 调用网络搜索获取最新信息
- 结合历史对话上下文
- 提供专业分析和建议

## 系统架构

```
投资研究支持系统
├── 知识库管理 (knowledge_base.py)
│   ├── 文档处理 (Unstructured)
│   ├── 向量存储 (Milvus) 
│   └── 实时监控 (Watchdog)
│
├── MCP工具集 (mcp_web_search.py)
│   ├── 网络搜索 (DuckDuckGo)
│   ├── 金融新闻 (RSS)
│   └── 股票信息
│
├── AI核心 (investment_research_system.py)
│   ├── LLM调用 (Claude/本地模型)
│   ├── 工具编排 (MCP)
│   └── 记忆管理 (LangChain)
│
└── 本地模型 (local_llm_config.py)
    ├── Ollama集成
    ├── 模型管理
    └── API调用
```

## 最佳实践

### 📖 知识库建设

1. **文档分类**:
   ```
   documents/
   ├── research/          # 研究报告
   ├── strategy/          # 投资策略
   ├── analysis/          # 分析文章
   └── trading_rules/     # 交易原则
   ```

2. **文档命名**: 使用清晰的命名规则
   - `2024_Q3_新能源板块分析.pdf`
   - `价值投资策略_巴菲特理念.md`

3. **内容质量**: 确保文档内容高质量、结构化

### 🔍 搜索优化

1. **精确查询**: 使用具体的关键词
   - 好: "贵州茅台 估值分析"
   - 差: "白酒股票"

2. **多维度分析**: 结合基本面和技术面
3. **时效性**: 关注最新市场动态

### 💼 投研工作流

1. **晨间准备**:
   ```
   sector 今日热点板块
   search_finance_news 市场重要消息
   ```

2. **日间分析**:
   ```
   search_stock_info 关注标的
   web_search 相关新闻事件
   ```

3. **晚间复盘**:
   ```
   review 今日交易记录
   总结投资机会和风险点
   ```

## 故障排除

### 常见问题

1. **知识库搜索无结果**:
   - 检查documents目录是否有文档
   - 确认文档格式支持
   - 查看向量数据库是否正常创建

2. **网络搜索失败**:
   - 检查网络连接
   - 确认搜索服务可用性

3. **本地模型无响应**:
   - 检查Ollama服务状态: `ollama list`
   - 确认模型已下载
   - 检查系统资源使用

4. **API调用失败**:
   - 检查.env文件中的API密钥
   - 确认API额度充足
   - 查看网络代理设置

### 日志查看

系统运行时会输出详细日志，注意查看：
- 连接状态信息
- 错误和警告消息
- 工具调用结果

## 进阶配置

### 自定义提示词

可以在`investment_research_system.py`中修改`system_prompts`来定制AI助手的行为。

### 扩展MCP工具

在`mcp_web_search.py`中添加新的工具函数，支持更多数据源。

### 本地模型优化

根据硬件配置选择合适的模型大小，调整推理参数。

---

**技术支持**: 查看代码注释和文档，或在GitHub上提交Issue
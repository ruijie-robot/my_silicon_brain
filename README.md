# my_silicon_brain

# 写作目的
初稿在draft.md里面，请根据初稿现在的内容，再结合我下面的要求，对draft.md进行更改。需要广泛的搜索文献，把公募基金如何让公司AI化的各种例子囊括进来。

文章的写作目的，是让公募基金的高管们能够统一思想，认同下面几个观点，不要让公司变成一个外包工具的集合体：
1. 绝大部份的Agent项目停留在demo，而没有铺开，有2个原因：首先是低估了公司业务的复杂度，没有做高级的定制；其次是迫不及待的把Agent用于主要盈利业务上，而不是最适合LLM的重复工作替代
2. 个人成为人机智能体，应该使用哪些最简易直观的工具，但是公司成为智能体平台，只使用Dify这样的“开箱即用”的平台是不足够的，企业需要使用LangChain这样更灵活的平台专业定制，才能让AI和公司业务结合得更紧密
3. 大语言模型可以成为投资人的效率倍增器，但是并不会成为决策者，投资者可以通过调用更多的信息源，构建更合理的假设，对自己的假设提出更尖锐的问题，做出更准确的判断，最终，区别成功投资者与庸常之辈的根本区别，在于其思维体系的纪律性。
4. MCP和A2A的协议发展，让智能体和工具/数据的对话成为可能，A2A的发展，让智能体和智能体对话成为可能，公司通过这些通用协议，提供tools和Agent对整个公司的业务做底层的支持，而在业务层面可以自动的调用这些工具，可以实现整体员工效率的提升。



# 读者背景
我的读者是公募基金的CEO，他完全不懂技术，他想要知道未来AI Agent要如何运用到自己的公司，他曾经探索过“第四范式”这种为大家提供金融第三方工具的公司，他由于高估LangChain的开发难度，不希望自己做团队开发，想要找一个开箱即用的东西来整体提升公司的效率

# 文献寻找
我需要的文献是，作为公募基金作为一家公司，要如何利用AI发展公司的各方各面（包括怎么让基金经理更加有竞争力，或者怎么让公司运转的成本更低等等），而不是下面几个主题， 请不要错误的去寻找下面几个主题的文章：
1. 阐释投资AI类股票值得投资
比如说：JPMorgan Asset Management. (2025). "AI investment trends 2025: Beyond the bubble | JTEK Strategy." Retrieved from https://am.jpmorgan.com/lu/en/asset-management/institutional/insights/market-insights/investment-outlook/ai-investment/

或者：Fidelity Investments. (2025). "Investing for the next stage of artificial intelligence: Agentic AI." Retrieved from https://www.fidelity.com/learning-center/trading-investing/priyanshu-bakshi-ai

2. 仅仅是宽泛的说明AI的发展
比如说：Vanguard. (2025). "Investing in the age of AI | Vanguard Strategic AI Research Partnership." Retrieved from https://corporate.vanguard.com/content/corporatesite/us/en/corp/articles/better-vantage-episode-one.html



# 段落结构
### Section 1
请阅读State_of_AI_in_Business 2025.pdf 和 Using_LLM_for_stock_valuation_analysis.pdf，并且结合可以搜索到的其他文献，写一个综述，体现出我的写作目的 （当你阐述的时候，需要清楚的标注出每一个结论的Reference，并把文件名或者链接放在Reference section里面）


### Section 2
请阅读State_of_AI_in_Business 2025.pdf 和 Using_LLM_for_stock_valuation_analysis.pdf，并且结合可以搜索到的其他文献，从投资和风控两个方向出发，通过Blackrock对于这两个方向的布局为例子，做下面的分析 （当你阐述的时候，需要清楚的标注出每一个结论的Reference，并把文件名或者链接放在Reference section里面）
1. 投资
1.1 股票投资：结合Blackrock的一些AI产品和布局来分析
- DCF建模
- Scenario Modeling
- Sensitivity Analysis
- Comprehensive Peer Analysis
- Sector Rotation Analysis

1.2 固收投资
- private debt投资
- OTC交易自动化

2. 风控
- Portfolio-Level Valuation Analysis
- Portfolio Valuation Health Check
- Rebalancing Scenarios
- 合规

3. 销售
- 投资顾问

4. 交易
- 人机结合自动化交易

### Section 3
请阅读State_of_AI_in_Business 2025.pdf 和 Using_LLM_for_stock_valuation_analysis.pdf，并且结合可以搜索到的其他文献，简述公募基金公司要如何从无到有来进行AI化的进程，比如说，首先需要积累数据，然后把数据变为一个Agent friendly的形式（也就是tools），根据tools的调用，产生不同的Agent。这是一个金字塔的结构，最底层是数据的积累，接着是tools的构建，然后是Agent的构建 （当你阐述的时候，需要清楚的标注出每一个结论的Reference，并把文件名或者链接放在Reference section里面）

然后从风控和投资2个方面，举2个例子，说明要怎么做

1. 数据清洗（比如说， 把财报自动转换为格式化数据）

2. tools构建：根据数据构建上层Agent可以调用的Tools

3. Agent的构建：可以是公司级别的复杂的Agent构建（比如说，风控Agent，或者审批Agent，这些大的Agent可以直接提供服务，或者接入不同部门提供服务），也可以是个人Agent构建 （投资经理助手Agent， 这些Agent可以访问投资经理自己的资源库，并且调用公司提供的数据&通用Agent）


### Section 4: Reference





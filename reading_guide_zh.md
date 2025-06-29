# `src` 目录阅读指南

本文档为 `src` 目录中的每个 Python 脚本提供了详细的阅读指南，旨在帮助理解每个脚本的用途、功能和实现细节。

## 目录

1.  [`src/config.py`](#srcconfigpy)
2.  [`src/context_generator_BFS.py`](#srccontext_generator_bfspy)
3.  [`src/context_generator_path.py`](#srccontext_generator_pathpy)
4.  [`src/context_generator_rag.py`](#srccontext_generator_ragpy)
5.  [`src/context_generator_rw.py`](#srccontext_generator_rwpy)
6.  [`src/context_pruner.py`](#srccontext_prunerpy)
7.  [`src/design_context_summarizer.py`](#srcdesign_context_summarizerpy)
8.  [`src/doc_KG_processor.py`](#srcdoc_kg_processorpy)
9.  [`src/dynamic_prompt_builder.py`](#srcdynamic_prompt_builderpy)
10. [`src/gen_KG_graphRAG.py`](#srcgen_kg_graphragpy)
11. [`src/gen_plan.py`](#srcgen_planpy)
12. [`src/kg_traversal.py`](#srckg_traversalpy)
13. [`src/main.py`](#srcmainpy)
14. [`src/rtl_analyzer.py`](#srcrtl_analyzerpy)
15. [`src/rtl_kg.py`](#srcrtl_kgpy)
16. [`src/rtl_parsing.py`](#srcrtl_parsingpy)
17. [`src/saver.py`](#srcsaverpy)
18. [`src/sva_extraction.py`](#srcsva_extractionpy)
19. [`src/use_KG.py`](#srcuse_kgpy)
20. [`src/utils.py`](#srcutilspy)
21. [`src/utils_LLM.py`](#srcutils_llmpy)
22. [`src/utils_LLM_client.py`](#srcutils_llm_clientpy)
23. [`src/utils_gen_plan.py`](#srcutils_gen_planpy)

---

## `src/config.py`

*   **目的：** 此脚本集中管理项目的配置设置，项目似乎涉及“生成计划” (`gen_plan`)、“构建知识图谱” (`build_KG`) 和“使用知识图谱” (`use_KG`) 等任务。它根据选定的 `task` 和 `subtask` 定义各种参数。
*   **关键变量：**
    *   `task`: 决定主要操作模式 (例如, `gen_plan`, `build_KG`)。
    *   `subtask`: 指定 `task` 内更细粒度的操作 (例如, `gen_plan` 中的 `actual_gen`, `parse_result`)。
    *   `DEBUG`:布尔标志，用于启用或禁用调试模式，这通常会影响 `max_num_signals_process` 和 `max_prompts_per_signal` 等设置。
    *   `design_name`: 指定正在处理的特定硬件设计 (例如, `apb`, `ethmac`, `uart`)。设计文件和知识图谱 (KG) 文件的路径会基于此设置。
    *   `file_path`: 设计规范文档的路径 (通常是 PDF)。
    *   `design_dir`: 包含设计 RTL 文件的目录路径。
    *   `KG_path`: 预构建知识图谱文件的路径 (GraphML 格式)。
    *   `llm_engine_type`, `llm_model`: 大型语言模型 (LLM) 的配置，指定类型和模型名称 (例如, `gpt-4o`)。
    *   `max_tokens_per_prompt`: LLM 提示的最大令牌数。
    *   `use_KG`: 布尔标志，决定是否应使用知识图谱。
    *   `prompt_builder`: 构建提示的方法 (`static` 或 `dynamic`)。
    *   `dynamic_prompt_settings`: 如果 `prompt_builder` 为 `dynamic`，则此嵌套字典配置各种上下文生成策略。包括以下设置：
        *   `rag` (检索增强生成): 块大小、重叠、块数量 (k)、启用 RTL 代码 RAG。
        *   `path_based`: KG 中路径探索的最大深度、表示样式。
        *   `motif`: 发现图基元（例如握手、流水线）的设置。
        *   `community`: 检测 KG 中社区的设置。
        *   `local_expansion`: BFS 扩展的最大深度、子图大小限制。
        *   `guided_random_walk`: KG 上随机游走的参数。
        *   `pruning`: 基于 LLM 的上下文修剪设置。
    *   `refine_with_rtl`: 布尔值，是否使用 RTL 优化信息。
    *   `gen_plan_sva_using_valid_signals`, `valid_signals`: 基于指定的有效信号生成 SystemVerilog 断言 (SVA) 的配置。
    *   `generate_SVAs`: 布尔值，是否生成 SVA。
    *   `load_dir` (`parse_result` 子任务): 从中加载结果的目录。
    *   `input_file_path` (`build_KG` 任务): 用于 KG 构建的输入文档路径。
    *   `env_source_path`, `settings_source_path`, `entity_extraction_prompt_source_path` (`build_KG` 任务): GraphRAG 的环境、设置和提示文件的路径。
    *   `KG_root`, `graphrag_method`, `query` (`use_KG` 任务): 查询 KG 的配置。
    *   `graphrag_local_dir`: 本地 GraphRAG 存储库的路径。
    *   `ROOT`, `repo_name`, `local_branch_name`, `commit_sha`: 自动确定的 Git 存储库信息。
    *   `FLAGS`:一个 `SimpleNamespace` 对象，聚合了脚本中定义的大多数变量，使其易于访问 (例如, `FLAGS.design_name`)。
*   **功能：**
    1.  设置主要 `task` (例如, 'gen\_plan')。
    2.  根据 `task` 设置进一步的配置。对于 'gen\_plan'，它涉及选择 `design_name`，然后设置路径、LLM 参数以及使用知识图谱 (KG) 和文档检索的策略。
    3.  对于 'build\_KG'，它设置输入文档和 GraphRAG 配置文件的路径。
    4.  对于 'use\_KG'，它配置 KG 访问和查询。
    5.  它使用 `pathlib.Path` 进行稳健的路径操作。
    6.  它在末尾动态创建一个 `FLAGS` 对象 (一个 `SimpleNamespace`)，其中包含脚本中定义的所有配置变量。这允许其他模块导入 `FLAGS` 并轻松访问配置设置 (例如, `from config import FLAGS`)。
    7.  包含占位符，如 `/<path>/<to>/...`，在实际环境中需要替换为实际路径。
    8.  检索用户和主机名，以及 Git 存储库详细信息（存储库名称、分支、提交 SHA）。
*   **依赖：** `types.SimpleNamespace`, `pathlib.Path`, `utils.get_user`, `utils.get_host`, `collections.OrderedDict`, `git` (可选, 如果未找到则进行错误处理)。
*   **用途：** 此文件可能被 `src` 目录中的大多数其他脚本导入，以访问全局配置设置。

---

## `src/context_generator_BFS.py`

*   **目的：** 此脚本定义了 `LocalExpansionContextGenerator` 类，该类负责通过从指定的接口信号或节点开始，在知识图谱 (KG) 上执行广度优先搜索 (BFS) 扩展来生成上下文。目标是为 KG 中这些起点的局部邻域创建文本描述。
*   **主要类和方法：**
    *   `LocalExpansionContextGenerator`:
        *   `__init__(self, kg_traversal: 'KGTraversal')`:
            *   使用 `KGTraversal` 对象进行初始化 (可能提供对图 `self.G` 和 `signal_to_node_map` 的访问)。
            *   从 `FLAGS.dynamic_prompt_settings['local_expansion']['max_depth']` 设置 BFS 的 `max_depth`。
        *   `get_contexts(self, start_nodes: List[str]) -> List[ContextResult]`:
            *   主要的公共方法。接受一个 `start_nodes` 列表 (信号名称或节点 ID)。
            *   如果可用，则使用 `self.signal_to_node_map` 将信号名称映射到实际的图节点 ID。
            *   对于每个有效的起始节点：
                *   调用 `_bfs_expansion` 获取局部子图。
                *   调用 `_calculate_subgraph_metrics` 获取此子图的度量。
                *   调用 `_describe_local_subgraph` 生成文本描述。
                *   调用 `_calculate_context_score` 对生成的上下文进行评分。
                *   将描述、分数和元数据包装到 `ContextResult` 对象中 (从 `context_pruner` 导入)。
            *   返回 `ContextResult` 对象列表。
        *   `_bfs_expansion(self, start_node: str, max_depth: int) -> Set[str]`:
            *   从 `start_node` 开始执行 BFS，最大深度为 `max_depth`。
            *   如果访问的节点数达到 `max_subgraph_size` (来自 `FLAGS`)，则限制扩展。
            *   返回扩展子图中节点 ID 的集合。
        *   `_calculate_subgraph_metrics(self, nodes: Set[str]) -> Dict`:
            *   接受一组子图节点。
            *   使用 `networkx` 函数对 `self.G.subgraph(nodes)` 计算节点数、边数、密度、节点类型分布和平均聚类系数等度量。
            *   返回这些度量的字典。
        *   `_calculate_context_score(self, nodes: Set[str], start_node: str, metrics: Dict) -> float`:
            *   根据子图大小、节点类型的多样性和密度，为生成的上下文计算相关性得分 (0.0 到 1.0)。
        *   `_describe_local_subgraph(self, nodes: Set[str], start_node: str, metrics: Dict) -> str`:
            *   生成局部子图的详细文本描述。
            *   包括有关起始节点的信息 (名称、类型、模块)。
            *   列出子图度量 (大小、密度、聚类)。
            *   显示节点类型分布。
            *   列出子图中的关键信号、连接的模块和相关寄存器 (限制为少量示例)。
            *   描述 `start_node` 的直接连接，包括从边属性派生的关系类型。
            *   根据节点名称关键字 (例如，识别控制信号如 'clk', 'reset', 'enable' 或数据信号) 和连接性 (例如，识别中心节点) 提供“分析见解”。
*   **数据结构：**
    *   `ContextResult`: 一个命名元组或类 (从 `context_pruner` 导入)，可能包含 `text`、`score`、`source_type` 和 `metadata`。
*   **功能：**
    1.  通过在指定入口点 (信号) 周围进行局部探索，识别较大知识图谱的相关部分。
    2.  使用 BFS 进行局部探索，受深度和子图大小的限制。
    3.  使用图度量量化探索子图的特征。
    4.  生成子图的可读摘要，突出显示关键元素，如连接的信号、模块以及基于其名称和连接的起始信号的潜在角色。
    5.  对生成的上下文进行评分，以评估其潜在的相关性或质量。
*   **依赖：** `kg_traversal.KGTraversal`, `networkx` (as `nx`), `typing` (List, Dict, Set, Tuple, Optional), `context_pruner.ContextResult`, `config.FLAGS`, `numpy` (as `np`, 虽然在提供的代码片段中没有明确使用，但可能被依赖项或类的其他部分使用), `saver.saver` (用于通过 `print = saver.log_info` 进行日志记录)。
*   **用途：** 此类将使用 `KGTraversal` 对象进行实例化。然后，将使用信号名称列表调用其 `get_contexts` 方法。输出上下文可用于例如向 LLM 提供相关信息或用于其他分析任务。它是一个利用 KG 完成设计理解或验证等任务的较大系统中的组件。它在很大程度上依赖于 `FLAGS` 提供的配置作为其操作参数。

---

## `src/context_generator_path.py`

*   **目的：** 此脚本定义了 `PathBasedContextGenerator` 类。其目标是从指定节点 (信号) 开始，从知识图谱 (KG) 中提取有意义的路径。然后，它描述这些路径以提供上下文，可能用于 LLM 或其他分析。可以生成不同风格的路径描述。
*   **主要类和方法：**
    *   `PathBasedContextGenerator`:
        *   `__init__(self, kg_traversal: KGTraversal)`:
            *   使用 `KGTraversal` 对象进行初始化 (提供图访问 `self.G` 和 `signal_to_node_map`)。
            *   一次性计算并存储全局图度量 (`_calculate_global_metrics`)。
        *   `get_contexts(self, start_nodes: List[str]) -> List[ContextResult]`:
            *   主要的公共方法。将输入的 `start_nodes` (信号名称) 映射到图节点 ID。
            *   对于每个有效节点：
                *   调用 `_find_significant_paths` 查找源自该节点的路径。
                *   对于每个找到的路径：
                    *   使用 `_calculate_path_importance` 计算路径重要性。
                    *   使用 `_describe_enhanced_path` 生成路径描述 (可根据 `FLAGS` 变化)。
                    *   创建一个包含描述、分数和元数据的 `ContextResult` 对象。
            *   返回 `ContextResult` 对象列表。
        *   `_calculate_global_metrics(self) -> Dict`:
            *   计算整个图的全局度量 (例如，平均度数、密度、平均聚类系数、中心性、介数中心性)。用作评估路径重要性的基准。
        *   `_find_significant_paths(self, start_node: str) -> List[Tuple[List[str], Dict]]`:
            *   使用 `_identify_potential_endpoints` 识别从 `start_node` 开始的路径的潜在端点节点。
            *   对于每个潜在端点，使用 `nx.all_simple_paths` 查找 `start_node` 和端点之间的所有简单路径 (最大深度来自 `FLAGS`)。
            *   对于找到的每个路径，计算其度量 (`_calculate_path_metrics`) 并检查其是否重要 (`_is_significant_path`)。
            *   返回 (路径, 路径度量) 元组列表。
        *   `_identify_potential_endpoints(self, start_node: str) -> Set[str]`:
            *   识别一组用于路径查找的有趣目标节点。
            *   考虑 `start_node` 的多跳邻域中的节点和一些随机节点。
            *   查找度数或聚类系数高于平均值、局部中心、特定节点类型 (端口、模块、分配) 或名称与 `start_node` 相似的节点。
            *   限制端点数量以管理性能。
        *   `_calculate_path_metrics(self, path: List[str]) -> Dict`:
            *   计算给定路径 (节点 ID 列表) 的度量，例如路径子图的长度、边密度、路径聚类以及路径中节点的度数。它还尝试从全局度量中包含路径中节点的中心性和介数中心性分数。
        *   `_is_significant_path(self, metrics: Dict) -> bool`:
            *   通过将其度量 (平均节点度数、聚类系数、边密度) 与全局图平均值进行比较来确定路径是否重要。
        *   `_calculate_path_importance(self, path: List[str], metrics: Dict) -> float`:
            *   根据路径长度 (偏好中等长度)、密度、聚类系数 (相对于全局) 及其节点的平均中心性计算路径的重要性得分。
        *   `_describe_enhanced_path(self, path: List[str], metrics: Dict) -> str`:
            *   根据 `FLAGS.dynamic_prompt_settings['path_based']['representation_style']` 作为调度程序。
            *   调用 `_generate_concise_path_description`、`_generate_detailed_path_description`、`_generate_verification_focused_path_description` 或标准的默认描述之一。
        *   `_generate_concise_path_description(...)`: 生成路径的非常简短的摘要。
        *   `_generate_detailed_path_description(...)`: 生成详细的描述，包括路径上所有节点和边的属性。
        *   `_generate_verification_focused_path_description(...)`: 生成突出显示与硬件验证相关的方面的描述 (时序、控制/数据依赖性、跨模块接口)。
        *   默认 (标准) 路径描述：
            *   提供带有源/目标、路径长度和聚类系数的标头。
            *   详细说明源/目标节点信息 (名称、类型、模块)。
            *   列出“关系链”，显示路径中的每个节点、其类型、模块以及与下一个节点的关系 (来自边属性)。
            *   包括一个“分析”部分，尝试推断路径特征 (例如，信号流、跨模块路径、输入到输出路径、信号转换)。
*   **功能：**
    1.  从给定信号开始，识别 KG 内结构上有趣的路径。
    2.  使用图算法 (`nx.all_simple_paths`) 和启发式方法查找和评估这些路径。
    3.  通过 `FLAGS` 配置，提供这些路径的不同文本表示，以满足各种下游用途 (例如，用于修剪的简洁表示、用于深入分析的详细表示、用于特定任务的面向验证的表示)。
    4.  对路径进行评分以对其潜在重要性或相关性进行排序。
*   **依赖：** `kg_traversal.KGTraversal`, `context_pruner.ContextResult`, `config.FLAGS`, `numpy` (as `np`), `networkx` (as `nx`), `saver.saver` (用于日志记录), `typing`, `dataclasses`。
*   **用途：** 此生成器将用于从 KG 中提取线性上下文信息，重点关注连接实体的序列。生成的路径描述有助于理解设计不同部分之间的关系和流程。

---

## `src/context_generator_rag.py`

*   **目的：** 此脚本定义了用于检索增强生成 (RAG) 的类。主要类 `RAGContextGenerator` 旨在基于查询从规范文档和 RTL (寄存器传输级) 代码中检索相关的文本块。它支持可配置的分块策略。
*   **主要类和方法：**
    *   `RAGContextGenerator`:
        *   `__init__(self, spec_text: str, rtl_code: Optional[str] = None, chunk_sizes: List[int] = None, overlap_ratios: List[float] = None)`:
            *   使用规范文本 (`spec_text`) 和可选的 RTL 代码 (`rtl_code`) 进行初始化。
            *   接受用于文档分块的 `chunk_sizes` 和 `overlap_ratios` 列表，或使用默认值。
            *   从 `FLAGS.dynamic_prompt_settings['rag']` 读取特定于 RTL 的 RAG 设置 (例如, `enable_rtl_rag`, `baseline_full_spec_RTL`)。
            *   如果 `baseline_full_spec_RTL` 为 true，则绕过正常的分块/检索，并准备返回完整的规范和 RTL 连接。
            *   否则，它会为 `spec_text` 创建具有不同块/重叠配置的多个 `DocRetriever` 实例。
            *   如果 `enable_rtl_rag` 为 true 且提供了 `rtl_code` (并且不是基线模式)，它还会为 `rtl_code` 创建 `DocRetriever` 实例。
        *   `get_contexts(self, query: str, k: int = None) -> List[ContextResult]`:
            *   如果 `baseline_full_spec_RTL` 处于活动状态，则返回包含完整 `spec_text` 和 `rtl_code` 的单个 `ContextResult`。
            *   否则，迭代所有已初始化的 `spec_retrievers` 和 `rtl_retrievers`。
            *   对于每个检索器，使用 `query` 和 `k` (要获取的块数) 调用其 `retrieve` 方法。
            *   根据每个检索到的块的等级和块大小对其进行评分 (对于规范，偏好约 100 个单词的块；对于 RTL，偏好约 50 个单词的块)。RTL 块会获得轻微的分数奖励。
            *   为每个块创建 `ContextResult` 对象，包括块大小、重叠、等级和内容类型 ('spec' 或 'rtl') 等元数据。
            *   按分数对所有收集的上下文进行排序，并返回最高的 `max_contexts` (默认为 10)。
    *   `DocRetriever`:
        *   `__init__(self, text, chunk_size=100, overlap=20, source_type='spec')`:
            *   存储 `chunk_size`、`overlap` 和 `source_type` ('spec' 或 'rtl')。
            *   调用 `_create_chunks` 将输入 `text` 分割成重叠的块。
            *   初始化 `TfidfVectorizer` 并计算所有块的 TF-IDF 向量。
        *   `_create_chunks(self, text, chunk_size, overlap)`:
            *   如果 `source_type` 是 'rtl'，则调用 `_create_code_aware_chunks`。
            *   否则 (对于 'spec')，按单词分割文本，并创建 `chunk_size` 个单词和 `overlap` 个单词的块。
        *   `_create_code_aware_chunks(self, code_text, chunk_size, overlap)`:
            *   尝试从 RTL 代码创建尊重代码结构 (例如，模块定义、函数、always 块) 的块。
            *   按行分割，并尝试在块内保持重要的块 (`module`, `function`, `task`, `always`, `initial`, `endmodule` 等) 完整，同时尽可能遵守 `chunk_size` (以单词为单位)。
        *   `retrieve(self, query, k=3)`:
            *   将 `query` 转换为 TF-IDF 向量。
            *   计算查询向量和所有块向量之间的余弦相似度。
            *   返回相似度最高的 `k` 个块。
    *   `KGNodeRetriever`: (此类似乎有点不适合主要关注文本文档的“RAG”上下文生成器，但它存在于文件中。)
        *   `__init__(self, kg)`:
            *   接受 KG 结构 (带有 'nodes' 键的字典)。
            *   初始化 `SentenceTransformer` 模型 (`paraphrase-MiniLM-L6-v2`)。
            *   调用 `_create_node_embeddings` 为 KG 中的所有节点生成嵌入。
        *   `_create_node_embeddings(self)`:
            *   为每个节点创建文本表示 (ID + 属性)。
            *   使用句子转换器模型将这些文本编码为嵌入。
        *   `retrieve(self, query, k=3)`:
            *   将 `query` 编码为嵌入。
            *   计算查询嵌入和所有节点嵌入之间的余弦相似度。
            *   返回最相似的 `k` 个 KG 节点的 ID。
*   **功能：**
    1.  提供一种机制，用于根据文本查询从较大的文档 (规范、RTL 代码) 中检索相关的文本片段。
    2.  在 `DocRetriever` 中使用 TF-IDF 和余弦相似度进行检索。
    3.  通过创建具有不同参数的多个检索器，允许灵活的分块策略。
    4.  为 RTL 提供“代码感知”分块方法，以尝试保持语义块的完整性。
    5.  包括一个仅返回所有文本的基线模式，以及一个启用/禁用 RTL RAG 的选项。
    6.  `KGNodeRetriever` 类使用句子嵌入来查找与查询相似的 KG 节点，这是一种不同类型的检索 (基于节点与基于文本块)。
*   **依赖：** `typing`, `context_pruner.ContextResult`, `sklearn.feature_extraction.text.TfidfVectorizer`, `sklearn.metrics.pairwise.cosine_similarity`, `sentence_transformers.SentenceTransformer`, `config.FLAGS`, `saver.saver`。
*   **用途：** `RAGContextGenerator` 使用规范的全文和可选的 RTL 代码进行实例化。然后，使用查询调用其 `get_contexts` 方法以获取相关的文本块。这些块可用于增强 LLM 的提示，提供特定的文本上下文。`KGNodeRetriever` 可单独用于根据自然语言查询在 KG 中查找相关的起点或实体。

---

## `src/context_generator_rw.py`

*   **目的：** 此脚本定义了 `GuidedRandomWalkContextGenerator` 类，该类通过执行引导式随机游走从知识图谱 (KG) 生成上下文信息。这些游走从指定的接口信号开始，并偏向于探索朝向其他接口信号的路径，旨在发现相关的子图或节点序列。
*   **主要类和方法：**
    *   `GuidedRandomWalkContextGenerator`:
        *   `__init__(self, kg_traversal: 'KGTraversal')`:
            *   使用 `KGTraversal` 对象进行初始化 (用于图访问 `self.G` 和 `signal_to_node_map`)。
            *   从 `FLAGS.dynamic_prompt_settings['guided_random_walk']` 加载随机游走的参数 (例如, `num_walks`, `walk_budget`, `teleport_probability`, 评分组件的权重 `alpha`, `beta`, `gamma`, `max_targets_per_walk`)。
            *   初始化预计算数据的占位符：`_signal_distance_map`, `_gateway_nodes`, `_node_importance`。
        *   `get_contexts(self, start_nodes: List[str]) -> List[ContextResult]`:
            *   主要的公共方法。将输入的 `start_nodes` (信号名称) 映射到图节点 ID。
            *   如果尚未完成，则预计算信号距离 (`_precompute_signal_distances`)、识别网关节点 (`_identify_gateway_nodes`) 并计算节点重要性 (`_compute_node_importance`)。这对所有 `start_nodes` 只执行一次。
            *   对于每个有效的 `focus_node` (从 `start_nodes` 映射而来)：
                *   将其他接口信号识别为潜在目标。
                *   使用 `_guided_random_walk` 执行 `self.num_walks` 次引导式随机游走。
                *   使用 `_filter_and_rank_paths` 过滤和排序结果路径。
                *   对于排名靠前的路径 (上限来自 `FLAGS`)：
                    *   计算路径度量 (`_calculate_path_metrics`)。
                    *   使用 `_describe_path` 生成路径的文本描述。
                    *   根据发现的信号和路径质量计算分数。
                    *   创建并附加一个 `ContextResult` 对象。
            *   返回 `ContextResult` 对象列表。
        *   `_precompute_signal_distances(self, focus_nodes: List[str])`:
            *   计算并存储所有接口信号对之间 (或至少与 `focus_nodes` 相关的那些信号对之间) 的最短路径距离和实际路径 (包括下一跳)。使用 `nx.single_source_shortest_path_length` 和 `nx.single_source_shortest_path`。将结果存储在 `self._signal_distance_map` 中。
        *   `_identify_gateway_nodes(self)`:
            *   识别在不同接口信号之间的最短路径中经常出现的“网关”节点。
            *   计算预计算的最短路径中的节点出现次数。
            *   将每个信号对的顶部网关节点存储在 `self._gateway_nodes` 中。
        *   `_compute_node_importance(self)`:
            *   根据图节点的类型 (例如, 'port', 'signal', 'register' 更重要) 和度数计算其重要性得分。存储在 `self._node_importance` 中。
        *   `_guided_random_walk(self, start_node: str, target_signals: List[str], budget: int) -> Tuple[List[str], Set[str]]`:
            *   执行单次随机游走。
            *   为当前游走选择 `target_signals` 的子集。
            *   从 `current_node` 迭代移动到从邻居中选择的 `next_node`。
            *   以 `self.teleport_prob` 的概率，可能会跳转到 `_select_gateway` 节点。
            *   否则，根据 `_compute_transition_probabilities` 选择 `next_node`，该函数考虑局部节点重要性、朝向目标的方向以及新节点的发现。
            *   跟踪 `discovered_signals` (到达的目标)。
            *   返回所采用的路径和发现的信号集。
        *   `_compute_transition_probabilities(...)`: 根据局部重要性 (`alpha`)、朝向目标的方向 (`beta`) 和发现未访问节点 (`gamma`) 的加权和计算移动到每个候选邻居的概率。
        *   `_select_gateway(...)`: 选择一个潜在的网关节点进行传送，从预先识别的朝向当前目标的路径上的网关中选择。
        *   `_filter_and_rank_paths(...)`: 过滤掉重复路径和未发现任何目标信号的路径。根据发现的信号数量、路径长度 (越短越好) 和路径质量度量对剩余路径进行排序。
        *   `_calculate_path_metrics(self, path: List[str]) -> Dict`: 计算给定路径的度量，包括长度、节点类型分布、多样性、平均节点重要性和边质量 (基于边上的关系类型)。将这些组合成一个整体的 `quality_score`。
        *   `_describe_path(...)`: 生成路径的可读描述，包括焦点节点详细信息、节点序列及其关系 (使用 `_get_relationship_description`)、发现的信号以及路径分析和度量的摘要。
        *   `_get_relationship_description(self, relationship: str) -> str`: 将边属性中的技术关系标识符映射为更易读的短语 (例如, "assigns_to" -> "assigns to")。
        *   `_get_signal_for_node(self, node: str) -> Optional[str]`: 用于使用 `self.signal_to_node_map` 查找与图节点 ID 对应的接口信号名称的辅助函数。
*   **功能：**
    1.  通过模拟 KG 内的智能探索 (随机游走) 来生成上下文。
    2.  引导这些游走发现起始信号和其他接口信号之间的连接。
    3.  使用预计算的图属性 (距离、网关、节点重要性) 来为游走决策和评分提供信息。
    4.  游走结合了局部重要性、目标方向和新颖性/发现等元素。
    5.  生成最有成果的游走的文本摘要，突出显示所采用的路径和发现的信号。
*   **依赖：** `networkx` (as `nx`), `numpy` (as `np`), `random`, `typing`, `collections.defaultdict`, `collections.Counter`, `context_pruner.ContextResult`, `config.FLAGS`, `saver.saver`, `heapq`。
*   **用途：** 此生成器提供了一种动态探索 KG 的方法。它不是使用固定模式 (如 BFS 或特定路径)，而是使用概率方法来查找连接多个接口信号的潜在有趣连接或子图。生成的上下文可以揭示复杂的交互。

---

## `src/context_pruner.py`

*   **目的：** 此脚本定义了 `LLMContextPruner` 类，该类使用大型语言模型 (LLM) 来评估和选择候选上下文中与特定任务最相关的部分。目标是将一组生成的上下文 (来自 RAG、路径查找等各种来源) 精炼成一个更小、更集中的集合，用于硬件验证提示生成等任务。它被描述为“简化的容错”版本，表明它试图具有包容性。
*   **主要类和方法：**
    *   `ContextResult` (dataclass):
        *   一个用于保存上下文片段的简单数据结构。
        *   属性：`text` (str)、`score` (float，可能来自生成过程)、`source_type` (str，例如 'rag', 'path')、`metadata` (Dict)。此类也在此处定义，可能是为了方便，或者为了确保在可能并非所有模块都导入中心定义的多个模块中使用时保持一致性。
    *   `LLMContextPruner`:
        *   `__init__(self, llm_agent, max_contexts_per_type=3, max_total_contexts=10)`:
            *   使用 `llm_agent` (用于进行 LLM 调用) 进行初始化。
            *   设置 `max_contexts_per_type` (每种来源类型如 'rag', 'path' 保留多少上下文) 和 `max_total_contexts` (总体限制)。
            *   设置 `min_contexts_per_type` (如果可用，每种类型至少选择多少上下文，默认为 2)。
        *   `prune(self, contexts: List[ContextResult], query: str, signal_name: str = None) -> List[ContextResult]`:
            *   主要的公共方法。接受 `ContextResult` 对象列表、原始 `query` (例如验证意图) 和可选的 `signal_name` 以进行聚焦。
            *   按 `source_type` 对输入的 `contexts` 进行分组。
            *   对于每种 `context_type`：
                *   如果该类型的上下文过多，则根据其原始 `score` 将其预过滤到 `max_eval_contexts` (例如 20)。
                *   使用 `_create_tolerant_prompt` 为 LLM 创建提示。此提示要求 LLM 选择当前 `context_type` 中与 `query` 和 `signal_name` 相关的上下文范围 (在 `min_contexts_per_type` 和 `max_contexts_per_type` 之间)。
                *   通过 `_call_llm` 调用 LLM。
                *   使用 `_parse_llm_response` 解析 LLM 的响应 (预期为索引列表)。
                *   如果 LLM 未选择任何内容或发生错误，则根据其原始分数回退到选择得分最高的上下文 (最多 `min_contexts_per_type`)。
                *   将 LLM 选择的 (或回退选择的) 上下文添加到 `selected_contexts`。
            *   如果所有类型中 `selected_contexts` 的总数超过 `max_total_contexts`，则调用 `_select_balanced_subset` 以进一步减少数量，同时尝试保持来自不同来源类型的多样性。
            *   返回最终修剪后的 `ContextResult` 对象列表。
        *   `_select_balanced_subset(self, contexts: List[ContextResult]) -> List[ContextResult]`:
            *   如果总体上选择了过多的上下文，此方法会尝试选择一个最多为 `max_total_contexts` 的子集。
            *   它确保每种类型都有最小数量 (按比例)，然后用剩余部分中得分最高的上下文填充剩余的空位。
        *   `_create_tolerant_prompt(...)`:
            *   为 LLM 构建详细的提示。
            *   指示 LLM 扮演硬件验证专家的角色。
            *   提供 `query` 和 `signal_name`。
            *   要求 LLM 选择特定 `context_type` 的 `min_selection` 和 `max_selection` 之间的上下文。
            *   强调容错性：“即使上下文看起来只是间接相关，也至少选择 `min_selection` 个上下文”，“如有疑问，宁可包含也不要排除”。
            *   列出上下文 (对于非常长的上下文进行截断) 及其部分元数据。
            *   指定输出格式：“Selected contexts: \[索引列表]”。
            *   给出对硬件验证有用的信息示例 (信号连接、时序、协议等)。
        *   `_call_llm(self, prompt: str, tag: str) -> str`: 调用 `utils_LLM.llm_inference` 的包装器。
        *   `_parse_llm_response(self, response: str, max_index: int) -> List[int]`:
            *   使用正则表达式 (`Selected contexts:\s*\[(.*?)\]`) 从 LLM 的响应字符串中提取索引列表。
            *   验证索引是否在有效范围内。
*   **功能：**
    1.  使用 LLM 判断将可能庞大且多样化的候选上下文集合精炼成一个更小、更相关的集合。
    2.  最初按上下文来源类型操作，然后全局操作，以确保高质量上下文的数量可管理。
    3.  提示中的“容错”方法旨在保留可能间接有用的上下文，这对于复杂的验证任务可能很重要。
    4.  如果 LLM 处理失败或未返回任何选择，则包括回退机制 (使用原始分数)。
    5.  平衡最终选择，以尝试包含不同类型的上下文。
*   **依赖：** `utils_LLM.llm_inference`, `typing.List`, `typing.Dict`, `dataclasses.dataclass`, `time`, `re`, `saver.saver`。
*   **用途：** 在各种上下文生成器 (RAG、基于路径、BFS、随机游走等) 生成其候选上下文之后，使用 `LLMContextPruner` 智能地过滤和选择最佳上下文，以传递到后续阶段，可能是另一次 LLM 调用以生成验证计划或 SVA。它充当利用 LLM 理解能力的智能过滤器。

---

## `src/design_context_summarizer.py`

*   **目的：** 此脚本定义了 `DesignContextSummarizer` 类，该类负责使用 LLM 生成硬件设计的各种文本摘要。这些摘要涵盖设计规范、RTL 架构、信号详细信息和设计模式。目标是创建增强的上下文，可能用于增强 SVA (SystemVerilog 断言) 生成或其他验证任务的提示。
*   **主要类和方法：**
    *   `DesignContextSummarizer`:
        *   `__init__(self, llm_agent: str = "gpt-4")`:
            *   使用 `llm_agent` 字符串 (例如, "gpt-4") 进行初始化。
            *   设置缓存：`summary_cache` (用于特定信号的摘要)、`global_summary` (用于总体设计摘要) 和 `all_signals_summary`。
        *   `generate_global_summary(self, spec_text: str, rtl_text: str, valid_signals: List[str]) -> Dict[str, Any]`:
            *   如果尚未缓存在 `self.global_summary` 中，则生成整个设计的全面一次性摘要。
            *   调用辅助方法生成：
                *   `_generate_design_specification_summary(spec_text)`
                *   `_generate_rtl_architecture_summary(rtl_text)`
                *   `_generate_comprehensive_signals_summary(spec_text, rtl_text, valid_signals)` (此结果也存储在 `self.all_signals_summary` 中)
                *   `_generate_design_patterns_summary(spec_text, rtl_text)`
            *   将这些摘要与生成时间戳一起存储在 `self.global_summary` 字典中。
            *   返回 `self.global_summary` 字典。
        *   `_generate_design_specification_summary(self, spec_text: str) -> str`:
            *   创建一个提示，要求 LLM 总结提供的 `spec_text` (3-5 句，重点关注主要功能、关键组件、架构)。
            *   调用 `_call_llm` 获取摘要。
        *   `_generate_rtl_architecture_summary(self, rtl_text: str) -> str`:
            *   创建一个提示，要求 LLM 总结 `rtl_text` (3-5 句，重点关注模块层次结构、接口、关键架构特性)。
            *   调用 `_call_llm`。
        *   `_generate_comprehensive_signals_summary(self, spec_text: str, rtl_text: str, signals: List[str]) -> str`:
            *   创建一个提示，要求 LLM 分析 `spec_text` 和 `rtl_text`，为 `signals` 列表中的每个信号提供摘要。
            *   请求每个信号的详细信息：名称、类型、位宽、功能/用途、关键交互。
            *   要求输出为列表，每个信号都有自己的段落。
            *   调用 `_call_llm`。
        *   `_generate_design_patterns_summary(self, spec_text: str, rtl_text: str) -> str`:
            *   创建一个提示，要求 LLM 从 `spec_text` 和 `rtl_text` 中识别和总结关键设计模式、协议或对验证至关重要的结构 (例如，握手、FSM、流水线、仲裁器、CDC)。
            *   请求对模式及其验证含义进行简洁的总结 (5-10 句)。
            *   调用 `_call_llm`。
        *   `get_signal_specific_summary(self, signal_name: str, spec_text: str, rtl_text: str) -> Dict[str, str]`:
            *   如果尚未在 `self.summary_cache` 中，则为特定的 `signal_name` 生成详细摘要。
            *   创建一个提示，要求 LLM 根据 `spec_text` 和 `rtl_text` 提供 `signal_name` 的详细描述。
            *   请求详细信息：精确功能、类型/宽度、时序、关键关系、对系统行为的影响、特殊条件/边界情况。
            *   要求提供 3-5 句包含全面、以验证为重点的详细信息。
            *   调用 `_call_llm`。
            *   缓存并返回结果 (一个包含 "description" 和 "generation_time" 的字典)。
        *   `add_enhanced_context(self, dynamic_context: str, target_signal_name: str) -> str`:
            *   组合各种缓存的摘要以创建“增强上下文”字符串。
            *   它将 `global_summary` (设计概述、RTL 架构)、`target_signal_name` 的特定摘要 (来自 `self.summary_cache`)、`all_signals_summary` 和关键设计模式摘要前置到提供的 `dynamic_context`。
            *   如果未生成 `global_summary`，则返回原始 `dynamic_context` 并发出警告。
        *   `_call_llm(self, prompt: str, tag: str) -> str`: 调用 `utils_LLM.llm_inference` 的包装器。
*   **功能：**
    1.  利用 LLM 从设计规范和 RTL 代码中提取和总结关键信息。
    2.  生成全局设计摘要和特定信号的详细摘要。
    3.  识别和总结常见的硬件设计模式。
    4.  缓存生成的摘要以避免冗余的 LLM 调用。
    5.  提供一种将这些摘要整合到“增强上下文”块中的方法，大概是为了向下游 LLM 任务 (如 SVA 生成) 提供丰富、结构化的信息。
*   **依赖：** `typing.Dict`, `typing.List`, `typing.Optional`, `typing.Any`, `utils_LLM.llm_inference`, `time`, `re`, `saver.saver`。
*   **用途：** 将创建一个 `DesignContextSummarizer` 实例。将使用完整的规范和 RTL 调用一次 `generate_global_summary`。然后，可能会为感兴趣的单个信号调用 `get_signal_specific_summary`。最后，可以使用 `add_enhanced_context` 将这些结构化摘要前置到其他动态生成的上下文，然后再将其提供给另一个 LLM。

---

## `src/doc_KG_processor.py`

*   **目的：** 此脚本充当各种上下文生成模块的处理器和工厂，特别是那些与知识图谱 (KG) 和文本规范交互的模块。它包括基于配置标志初始化不同上下文生成器以及将信号名称映射到 KG 节点的函数。它还定义了 `MotifContextGenerator` 和 `CommunityContextGenerator` 类。
*   **主要函数和类：**
    *   `create_context_generators(spec_text: str, kg: Optional[Dict], valid_signals: List[str], rtl_knowledge) -> Dict[str, object]`:
        *   一个工厂函数，根据 `FLAGS.dynamic_prompt_settings` 中的设置初始化并返回上下文生成器对象的字典。
        *   接受 `spec_text`、`kg` (可以是字典或 `nx.Graph`)、`valid_signals` 列表和 `rtl_knowledge` (可能包含 RTL 内容)。
        *   如果提供了 `kg` 和 `valid_signals`，则调用 `build_signal_to_node_mapping` 将信号名称映射到 KG 节点 ID。处理不同的 KG 输入类型 (字典与 `nx.Graph`)。
        *   如果 `FLAGS.dynamic_prompt_settings['rag']['enabled']` 为 true，则初始化 `RAGContextGenerator`。
        *   如果 `kg` 可用：
            *   创建一个 `KGTraversal` 对象并将 `signal_to_node_map` 附加到该对象。
            *   如果 `FLAGS` 中相应的 `enabled` 标志为 true，则初始化其他基于 KG 的生成器：
                *   `PathBasedContextGenerator`
                *   `MotifContextGenerator`
                *   `CommunityContextGenerator`
                *   `LocalExpansionContextGenerator` (来自 `context_generator_BFS.py`)
                *   `GuidedRandomWalkContextGenerator` (来自 `context_generator_rw.py`)
        *   返回一个字典，其中键是生成器类型 (例如, 'rag', 'path')，值是生成器实例。
    *   `build_signal_to_node_mapping(kg: Dict, valid_signals: List[str]) -> Dict[str, List[str]]`:
        *   将文本 `valid_signals` 名称映射到 `kg` 中的节点 ID 列表。
        *   假定 `kg` 是一个字典 (如果需要，在调用此函数之前转换 `nx.Graph`，尽管类型提示是 `Dict`)。
        *   迭代 KG 中的节点。如果节点的类型是 'port'、'signal' 或 'assignment'，则检查其 'name' 属性是否与任何 `valid_signals` 匹配。
        *   对于 'assignment' 类型，它尝试提取基本信号名称 (例如, 从 "baud\_clk\_assignment")。
        *   还使用正则表达式单词边界匹配检查 'expression' 属性中信号名称的出现情况。
        *   打印有关映射过程的大量调试信息。
        *   返回一个将信号名称映射到相应节点 ID 列表的字典。
    *   `MotifContextGenerator`:
        *   `__init__(self, kg_traversal: KGTraversal)`: 使用 `KGTraversal` 初始化并获取 `signal_to_node_map`。
        *   `get_contexts(self, start_nodes: List[str]) -> List[ContextResult]`:
            *   将 `start_nodes` (信号名称) 映射到 KG 节点 ID。
            *   调用内部方法以查找围绕这些 `valid_nodes` 锚定的不同类型的基元/模式：
                *   `_find_cycles`: 使用 `nx.simple_cycles`。
                *   `_find_hubs`: 识别度数明显高于平均值的节点。
                *   `_find_dense_subgraphs`: 使用 `nx_comm.louvain_communities` 并检查密度。
            *   对于每个发现的模式，计算重要性 (`_analyze_pattern_importance`)，生成描述 (`_describe_enhanced_pattern`)，并创建一个 `ContextResult`。
        *   `_analyze_pattern_importance(self, nodes: List[str]) -> float`: 根据结构度量、连通性和中心性对模式重要性进行评分。
        *   `_describe_enhanced_pattern(self, pattern_type: str, pattern: Dict) -> str`: 为 'cycle'、'hub' 和 'dense' 模式生成详细的文本描述，重点关注 `start_node` 并解释模式在硬件上下文中的含义。
    *   `CommunityContextGenerator`:
        *   `__init__(self, kg_traversal: KGTraversal)`: 使用 `KGTraversal` 初始化并获取 `signal_to_node_map`。
        *   `get_contexts(self, start_nodes: List[str]) -> List[ContextResult]`:
            *   将 `start_nodes` 映射到 KG 节点 ID。
            *   使用 `_detect_communities_safely` 检测社区 (该方法尝试 Louvain，如果检测到权重问题，则回退到贪婪模块化/连通分量)。
            *   对于 `valid_nodes` 中的每个 `node_id`，如果它属于某个社区：
                *   计算社区度量 (`_calculate_community_metrics`)。
                *   使用 `_describe_enhanced_community` 生成描述。
                *   对上下文进行评分并创建一个 `ContextResult`。
            *   如果节点不在任何检测到的较大社区中，则包括回退逻辑以创建小型本地社区 (邻域)。
        *   `_detect_communities_safely(self) -> List[Set[str]]`: 尝试使用 `weight=None` 的 `nx_comm.louvain_communities`。如果怀疑字符串权重或其他问题，则创建未加权图副本并尝试 `nx_comm.greedy_modularity_communities` 或连通分量。
        *   `_calculate_community_metrics(self, community: Set[str]) -> Dict`: 计算社区子图的密度、平均度数、聚类系数。
        *   `_calculate_node_centrality(self, node: str, community: Set[str]) -> float`: 计算节点在其社区子图中的度中心性。
        *   `_describe_enhanced_community(self, community: Set[str], start_node: str) -> str`: 生成社区的详细描述，重点关注 `start_node`，列出相关模块、信号、`start_node` 的关键关系以及相关分配。
*   **功能：**
    1.  **上下文生成器编排：** `create_context_generators` 函数充当根据全局配置初始化各种上下文生成组件的中心点。
    2.  **信号映射：** `build_signal_to_node_mapping` 对于将文本信号名称 (可能来自 RTL 分析或用户输入) 连接到其在 KG 中的表示至关重要。
    3.  **基元检测：** `MotifContextGenerator` 识别 KG 中涉及指定信号的常见结构模式 (环、中心、密集区域)。这些模式可以指示重要的功能块或关系。
    4.  **社区检测：** `CommunityContextGenerator` 在 KG 中查找包含指定信号的密集互连节点组 (社区)。这些社区可以表示模块或密切相关的功能单元。
    5.  基元和社区生成器都为其发现提供描述性的文本上下文。
*   **依赖：** `kg_traversal.KGTraversal`, `context_generator_rag.RAGContextGenerator`, `context_generator_path.PathBasedContextGenerator`, `context_generator_BFS.LocalExpansionContextGenerator`, `context_generator_rw.GuidedRandomWalkContextGenerator`, `context_pruner.ContextResult`, `config.FLAGS`, `numpy`, `sklearn` (TF-IDF, 余弦相似度), `networkx` (包括 `nx_comm`), `sentence_transformers`, `saver.saver`, `typing`, `dataclasses`, `re`。
*   **用途：** `create_context_generators` 在处理流程的早期被调用，以设置所有必要的上下文生成工具。`build_signal_to_node_mapping` 是此工厂使用的关键实用程序。然后，使用信号列表调用 `MotifContextGenerator` 和 `CommunityContextGenerator` (以及由工厂初始化的其他生成器)，以从 KG 生成不同类型的上下文信息。

---

## `src/dynamic_prompt_builder.py`

*   **目的：** 此脚本定义了 `DynamicPromptBuilder` 类，该类负责通过动态收集和集成各种类型的上下文来构建 LLM 的提示。它使用多个上下文生成器，修剪收集的上下文，然后将它们分布在一个或多个提示中，同时遵守令牌限制。
*   **主要类和方法：**
    *   `DynamicPromptBuilder`:
        *   `__init__(self, context_generators: Dict[str, object], pruning_config: Dict, llm_agent, context_summarizer=None)`:
            *   使用 `context_generators` 字典 (例如, RAG、基于路径、基元)、`pruning_config` 设置 (来自 `FLAGS`)、用于修剪的 `llm_agent` 和可选的 `context_summarizer` 进行初始化。
            *   如果提供了 `llm_agent`，则初始化 `LLMContextPruner` 实例。
        *   `build_prompt(self, query: str, base_prompt: str, signal_name: Optional[str] = None, enable_context_enhancement: bool = False) -> List[str]`:
            *   主要的公共方法。
            *   从 `FLAGS` 确定 `max_prompts_per_signal` 和 `max_tokens_per_prompt`。
            *   如果 `enable_context_enhancement` 为 true 且 `context_summarizer` 可用，则估计添加摘要上下文的令牌开销。
            *   计算可用于动态获取上下文的 `max_context_tokens`，为 LLM 的响应和增强开销保留空间。
            *   为基于 KG 的生成器收集 `start_nodes`，如果可用则使用 `kg_node_retriever`，否则使用 `signal_name`。
            *   迭代 `self.context_generators`：
                *   在每个启用的生成器 (RAG、路径、基元、社区、本地扩展、引导式随机游走) 上调用 `get_contexts`。
                *   将所有生成的上下文聚合到 `all_contexts` 中。
            *   **修剪：** 如果 `self.pruning_config['enabled']` 为 true：
                *   使用 `self.llm_pruner.prune()` (内部使用 LLM) 根据 `query` 和 `signal_name` 从 `all_contexts` 中选择最相关的上下文。
                *   (`_prune_contexts_similarity` 方法是使用句子相似度的旧版回退方法，但主要流程使用 LLM 修剪器)。
            *   **上下文分布：**
                *   按 `source_type` 对 `selected_contexts` 进行分组。
                *   计算每种类型的上下文应分配到 `max_prompts_per_signal` 个提示中的数量，以确保均匀分布。
            *   **提示组装：**
                *   构建 `max_prompts_per_signal` 个提示。
                *   每个提示以 `base_prompt_text` 开头 (包括 `base_prompt` 和上下文的标头)。
                *   迭代每个提示槽和上下文类型，添加格式化的上下文 (`format_context`)，同时遵守 `max_context_tokens`。
                *   `format_context` 包括上下文文本和相关元数据。
                *   如果上下文不适合当前提示，则尝试将其放入另一个提示。如果无处安放，则跳过。
                *   如果提示中存在该类型的上下文，则为每种上下文类型添加标头 (例如, "RAG Context:", "PATH Context:")。
            *   返回完全构建的提示字符串列表 (`final_prompts`)。
            *   如果 `enable_context_enhancement` 为 true，则使用 `self.context_summarizer.add_enhanced_context()` 将摘要前置到每个最终提示。
        *   `_prune_contexts_similarity(...)`: 一种旧版/回退的上下文修剪方法，基于与查询的句子嵌入相似度。如果 `LLMContextPruner` 有可用的 LLM 代理，则不是主要的修剪方法。
*   **功能：**
    1.  编排从多个异构来源 (RAG、各种 KG 遍历方法) 生成上下文。
    2.  主要使用 LLM 智能地修剪收集的上下文，为给定的查询和信号选择最相关的信息。
    3.  将选定的上下文分布在配置数量的提示中，旨在实现上下文类型的均匀分布并遵守令牌限制。
    4.  使用其来源类型和元数据格式化上下文，以便在提示中清晰显示。
    5.  可选地通过 `ContextSummarizer` 将预先生成的设计摘要前置到提示中来增强提示。
*   **依赖：** `context_pruner.LLMContextPruner`, `context_pruner.ContextResult`, `utils_LLM.count_prompt_tokens`, `config.FLAGS`, `sentence_transformers.SentenceTransformer`, `sklearn.metrics.pairwise.cosine_similarity`, `numpy`, `networkx`, `saver.saver`, `typing`, `dataclasses`。
*   **用途：** 此类是准备复杂提示的核心组件。它使用各种上下文生成器和 LLM 代理进行实例化。然后，使用基本查询和提示结构调用 `build_prompt` 方法。输出是准备就绪的提示列表，每个提示都富含多样化且相关的上下文信息，专为 LLM 执行验证计划生成等任务而定制。

---

## `src/gen_KG_graphRAG.py`

*   **目的：** 此脚本提供使用 GraphRAG 工具构建知识图谱 (KG) 的功能。它自动执行设置 GraphRAG 环境、处理输入设计文档 (PDF 或 JSONL) 以及运行 GraphRAG 索引流程的过程。
*   **主要函数：**
    *   `build_KG()`:
        *   编排 KG 构建过程的主要函数。
        *   从 `FLAGS` 检索 `input_file_path` 和其他配置。
        *   `get_base_dir()`: 从输入文件路径确定基本目录。
        *   `create_directory_structure()`: 创建一个带有 `input` 子目录的 `graph_rag_<design_name>` 目录。
        *   `clean_input_folder()`: 删除 `input` 目录中任何现有的文件。
        *   `process_files()`: 处理输入文件。此函数调用 `process_single_file`。
            *   `process_single_file()`:
                *   如果是 JSONL：调用 `get_jsonl_stats()` 和 `parse_jsonl_to_text()`。
                *   如果是 PDF：调用 `get_pdf_stats()` 和 `parse_pdf_to_text()`。
                *   `parse_pdf_to_text()`: 使用 `PyPDF2` 从 PDF 页面提取文本，并将其保存到 `graph_rag_<design_name>/input` 目录中的 .txt 文件。
                *   `parse_jsonl_to_text()`: 从 JSONL 提取文本内容，可能清理表格和类似 LaTeX 的语法，并保存到 .txt 文件。
                *   `get_pdf_stats()`/`get_jsonl_stats()`: 收集统计信息，如页数、文件大小、字数和令牌数 (对 PDF 使用 `tiktoken`)。
        *   `initialize_graphrag()`: 运行 `python -m graphrag.index --init --root <graph_rag_dir>` 来设置 GraphRAG 项目结构。处理项目已初始化的情况。
        *   将自定义的 `.env` 文件 (来自 `FLAGS.env_source_path`) 和 `settings.yaml` (来自 `FLAGS.settings_source_path`) 复制到 `graph_rag_dir`。
        *   `copy_entity_extraction_prompt()`: 将自定义的实体提取提示 (来自 `FLAGS.entity_extraction_prompt_source_path`) 复制到 `graph_rag_dir/prompts/entity_extraction.txt`。
        *   `run_graphrag_index()`: 执行主要的 GraphRAG 索引命令：`python -m graphrag.index --root <graph_rag_dir>`。它会流式传输此命令的输出。
        *   `detect_graphrag_log_folder()`: (虽然已定义，但此函数似乎在 `build_KG` 中的 `run_graphrag_index` *之后* 被调用，但其内部的 `run_graphrag_index` 调用表明它可能旨在查找由索引运行*创建*的文件夹。`build_KG` 中的当前结构首先调用 `run_graphrag_index`，然后调用 `detect_graphrag_log_folder`，而后者本身又会再次调用 `run_graphrag_index` – 这可能是多余的或轻微的逻辑疏忽。) 目标是识别由 GraphRAG 创建的带时间戳的输出目录。
        *   使用 `utils.OurTimer` 对各个步骤进行计时。
    *   支持性实用函数：
        *   `get_base_dir()`: 确定输入文件的公共父目录。
        *   `get_pdf_stats()`, `get_jsonl_stats()`: 计算输入文件的统计信息。
        *   `parse_pdf_to_text()`, `parse_jsonl_to_text()`: 将输入文档转换为纯文本。`parse_jsonl_to_text` 包括特定的表格清理逻辑。
        *   `clean_input_folder()`: 清理目标输入目录。
        *   `clean_table()`: `parse_jsonl_to_text` 的辅助函数，用于重新格式化表格内容。
        *   `create_directory_structure()`: 设置 GraphRAG 所需的文件夹结构。
        *   `initialize_graphrag()`: 初始化 GraphRAG 项目。
        *   `copy_entity_extraction_prompt()`: 复制特定的提示文件。
        *   `run_graphrag_index()`: 运行 GraphRAG 索引过程并流式传输其输出。
        *   `detect_graphrag_log_folder()`: 尝试识别 GraphRAG 运行创建的输出文件夹。
*   **功能：**
    1.  自动执行 GraphRAG 索引流程的设置和执行。
    2.  处理 PDF 和 JSONL 输入文件，将其转换为适合 GraphRAG 的文本。
    3.  通过提供自定义的 `.env`、`settings.yaml` 和实体提取提示文件来配置 GraphRAG 运行。
    4.  执行 GraphRAG 索引命令并监控其输出。
    5.  为 KG 生成过程提供基本统计信息和计时。
*   **依赖：** `os`, `subprocess`, `shutil`, `json`, `re`, `pathlib.Path`, `PyPDF2`, `logging`, `datetime`, `utils.OurTimer`, `utils.get_ts`, `saver.saver`, `config.FLAGS`, `tiktoken`。
*   **用途：** 当 `FLAGS.task` 设置为 'build\_KG' 时调用 `build_KG()` 函数。它以设计文档作为输入，并使用外部 GraphRAG 工具生成知识图谱。生成的 KG (可能是 GraphML 文件和 `graph_rag_<design_name>/output` 目录中的其他工件) 然后将由系统的其他部分使用 (例如, 由 `KGTraversal` 和各种基于 KG 的上下文生成器使用)。

---

## `src/gen_plan.py`

*   **目的：** 此脚本是涉及为硬件设计生成测试计划和 SystemVerilog 断言 (SVA) 的过程的主要协调器。它读取设计规范 (PDF)，可选地使用知识图谱 (KG) 和 RTL 信息，与 LLM 交互以生成自然语言 (NL) 计划和 SVA，然后可以通过 JasperGold 运行这些 SVA 以进行形式验证。它还包括解析和分析先前运行结果的功能。
*   **主要函数：**
    *   `gen_plan()`: 主要入口点。
        *   处理两种 `FLAGS.subtask` 模式：
            *   `'actual_gen'`: 完整的生成和验证流程。
            *   `'parse_result'`: 从目录加载和分析结果。
        *   **在 `'actual_gen'` 模式下：**
            1.  `read_pdf()`: 读取 PDF 规范文件。
            2.  如果 `FLAGS.use_KG` 为 true：
                *   `load_and_process_kg()`: 将 KG (GraphML) 加载到 NetworkX 图和 JSON 表示中。
                *   如果 `FLAGS.refine_with_rtl`：调用 `refine_kg_from_rtl()` (来自 `rtl_parsing.py`) 以使用从 RTL 文件中提取的信息来扩充 KG。
            3.  `get_llm()`: 初始化 LLM 代理。
            4.  `write_svas_to_file([])` (使用空列表调用以获取有效信号) 或使用 `FLAGS.valid_signals`：从设计的顶层 SVA 文件中提取/定义有效信号名称列表。
            5.  如果 `FLAGS.enable_context_enhancement`：初始化 `DesignContextSummarizer`，生成全局摘要，并为要处理的信号预生成摘要。
            6.  `generate_nl_plans()`: 生成自然语言测试计划。此函数根据 `FLAGS.prompt_builder` 在内部选择 `generate_dynamic_nl_plans` 或 `generate_static_nl_plans`。
            7.  如果 `FLAGS.generate_SVAs` 为 true：
                *   `generate_svas()`: 从 NL 计划生成 SVA。还根据 `FLAGS.prompt_builder` 选择动态或静态生成。
                *   `write_svas_to_file()`: 将每个 SVA 写入单独的 `.sva` 文件，并将其嵌入原始模块接口中。
                *   `generate_tcl_scripts()`: 为每个 SVA 文件创建 JasperGold 的 TCL 脚本。
                *   `run_jaspergold()`: 为每个 TCL 脚本执行 JasperGold 并收集报告。
                *   `analyze_coverage_of_proven_svas()`: (似乎在 `utils_gen_plan.py` 中定义，但除了被调用之外，其直接输出在 `analyze_results` 中并未大量使用)。
            8.  `analyze_results()`: 打印并保存整个过程的摘要，包括 PDF 统计信息、计划/SVA 计数、JasperGold 结果和覆盖率度量。
        *   使用 `utils.OurTimer` 跟踪不同步骤的执行时间。
    *   `read_pdf(file_path: Union[str, List[str]]) -> Tuple[str, dict]`: 使用 `PdfReader` 从一个或多个 PDF 文件中读取文本，并返回连接的文本和统计信息 (页数、通过 `count_tokens_in_file` 计算的令牌数、文件大小)。
    *   `load_and_process_kg(kg_path: str) -> Tuple[nx.Graph, Dict]`: 将 GraphML 文件加载到 `nx.Graph` 中，并使用 `convert_nx_to_json` 将其转换为类似 JSON 的字典。打印 KG 统计信息。
    *   `convert_nx_to_json(G: nx.Graph) -> Dict`: 将 `nx.Graph` 转换为更简单的字典表示形式 (具有属性的节点、具有属性的边)。
    *   `generate_nl_plans(...)`: 调用 `generate_dynamic_nl_plans` 或 `generate_static_nl_plans` 的调度程序。
        *   `generate_dynamic_nl_plans(...)`:
            *   使用 `create_context_generators` (来自 `doc_KG_processor.py`) 获取各种上下文提供程序。
            *   使用 `DynamicPromptBuilder` 构建带有上下文的提示 (可能由 `DesignContextSummarizer` 增强)。
            *   为每个信号和上下文组合调用 `llm_inference` 以获取 NL 计划。
            *   对计划进行去重。
        *   `generate_static_nl_plans(...)`: 使用静态构建的提示 (`construct_static_nl_prompt`) 调用 `llm_inference` 以获取 NL 计划。使用 `parse_nl_plans` 解析输出。
    *   `generate_svas(...)`: `generate_dynamic_svas` 或 `generate_static_svas` 的调度程序。
        *   `generate_dynamic_svas(...)`: 与动态 NL 计划生成类似，使用 `DynamicPromptBuilder` 获取上下文。构建包含 NL 计划和 SVA 示例 (`get_sva_icl_examples`) 的提示，然后调用 `llm_inference`。使用 `extract_svas_from_block` (来自 `sva_extraction.py`) 提取 SVA。处理跨多个上下文的计划分发。
        *   `generate_static_svas(...)`: 使用静态构建的提示 (`construct_static_sva_prompt`) 调用 `llm_inference` 并提取 SVA。
    *   `get_sva_icl_examples()`: 返回一个包含 NL 计划到 SVA 转换的 few-shot 示例的字符串。
    *   `construct_static_nl_prompt(...)` & `construct_static_sva_prompt(...)`: 用于构建大型静态提示的函数，这些提示包括规范文本、(可选的) KG JSON、有效信号列表和示例。这些提示着重强调仅使用有效信号。
    *   `parse_nl_plans(result: str) -> Dict[str, List[str]]`: 将 LLM 输出 (预期格式："SIGNAL: plan text") 解析为字典。
    *   `write_svas_to_file(svas: List[str]) -> Tuple[List[str], Set[str]]`:
        *   从原始 SVA 文件 (`property_goldmine.sva` in `FLAGS.design_dir`) 读取模块接口。
        *   使用 `extract_signal_names` 从此接口提取 `valid_signals`。
        *   对于每个生成的 SVA，创建一个新的 `.sva` 文件，其中包含原始模块接口和格式化为断言属性的 SVA。
    *   `generate_tcl_scripts(sva_file_paths: List[str]) -> List[str]`: 通过修改原始 TCL 脚本模板以指向每个新的 SVA 文件来创建 JasperGold TCL 脚本。
    *   `modify_tcl_content(original_content: str, new_sva_path: str) -> str`: 用于在 TCL 内容中替换 SVA 文件路径的辅助函数。
    *   `run_jaspergold(tcl_file_paths: List[str]) -> List[str]`: 在单独的项目目录中为每个 TCL 脚本运行 JasperGold 并保存输出报告。JasperGold 命令路径的占位符。
    *   `analyze_results(...)`: 编译并打印各种统计信息：PDF 输入大小、NL 计划计数、SVA 计数、JasperGold 证明结果 (已证明、CEX、不确定、错误) 和覆盖率度量 (由 `calculate_coverage_metric` 提取)。使用 `generate_detailed_sva_report` 创建 SVA 状态的 CSV 和表格输出。
    *   `calculate_coverage_metric(jasper_out_str)`: 解析 JasperGold 输出文本以提取各种覆盖率百分比 (语句、分支、功能、翻转、表达式的 stimuli/COI)。还根据证明结果确定基本的“功能性”度量。
    *   `generate_detailed_sva_report(...)`: 创建一个 Pandas DataFrame 和 CSV 文件，总结每个 SVA 的 ID、证明状态和错误消息 (如果有)。使用 `extract_proof_status` 和 `extract_short_error_message`。
    *   `extract_error_message(...)` & `extract_short_error_message(...)`: 从 JasperGold 报告中提取错误详细信息的辅助函数。
    *   `log_llm_interaction(...)`: 记录 LLM 提示和响应。
    *   `extract_signal_names(module_interface: str) -> Set[str]`: 使用正则表达式从 Verilog 模块接口字符串中提取信号名称。
*   **功能：** 此脚本实现了一个端到端流水线，用于：
    1.  处理设计规范。
    2.  利用 KG 和 RTL 信息获取上下文。
    3.  使用 LLM 生成 NL 测试意图，然后生成 SVA。
    4.  使用 JasperGold 自动进行这些 SVA 的形式验证。
    5.  收集、分析和报告结果，包括验证状态和覆盖率度量。
    它支持“静态”提示构建方法 (一个大型提示) 和“动态”方法 (多个较小的提示，上下文来自各种生成器，由 `DynamicPromptBuilder` 管理)。
*   **依赖：** `sva_extraction.extract_svas_from_block`, `doc_KG_processor.create_context_generators`, `dynamic_prompt_builder.DynamicPromptBuilder`, `load_result` (各种函数), `rtl_parsing.refine_kg_from_rtl`, `utils_gen_plan` (各种函数), `design_context_summarizer.DesignContextSummarizer`, `os`, `math`, `subprocess`, `config.FLAGS`, `saver.saver`, `utils.OurTimer`, `utils_LLM` (get\_llm, llm\_inference), `networkx`, `typing`, `PyPDF2`, `json`, `random`, `re`, `pandas`, `tabulate`, `pathlib.Path`, `tqdm`。
*   **用途：** 这可能是 'gen\_plan' 任务的主要可执行脚本。它使用在 `config.FLAGS` 中设置的特定配置运行，以处理设计并评估 LLM 生成的 SVA 的质量。

---

## `src/kg_traversal.py`

*   **目的：** 此脚本定义了 `KGTraversal` 类，该类提供了表示和遍历知识图谱 (KG) 的基本功能。假定 KG 以特定的 JSON 类字典格式 (节点和边) 提供，并在内部转换为 `networkx.Graph` 对象。
*   **主要类和方法：**
    *   `KGTraversal`:
        *   `__init__(self, kg)`:
            *   接受 `kg` (预期为具有 'nodes' 和 'edges' 键的字典) 作为输入。
            *   调用 `_build_graph()` 构建内部 `networkx.Graph`。
            *   (注意：在 `doc_KG_processor.py` 中，初始化后会将实例变量 `self.signal_to_node_map` 附加到 `KGTraversal` 实例，使其可供使用 `KGTraversal` 的其他生成器使用。)
        *   `_build_graph(self)`:
            *   初始化 `self.graph = nx.Graph()`。
            *   从 `self.kg['nodes']` 将节点添加到 `self.graph`，包括其属性。
            *   从 `self.kg['edges']` 将边添加到 `self.graph`，包括其属性。
        *   `traverse(self, start_node, max_depth=2)`:
            *   从 `start_node` 开始，执行图遍历 (具体为 `_dfs` 中实现的深度优先搜索)，最大深度为 `max_depth`。
            *   在此特定遍历实例中跟踪 `visited_nodes` 和 `visited_edges` 以避免循环和冗余处理。
            *   返回 `result_nodes` (DFS 顺序的已访问节点 ID 列表) 和 `result_edges` (已访问边的列表，格式为 (source, target, attributes) 元组)。
        *   `_dfs(self, node, max_depth, current_depth, visited_nodes, visited_edges, result_nodes, result_edges)`:
            *   `traverse` 的递归辅助函数。
            *   将当前 `node` 添加到 `result_nodes`。
            *   对于每个邻居，如果尚未访问该边，则将其添加到 `result_edges` 并对该邻居递归调用 `_dfs`。
        *   `get_node_info(self, node_id)`:
            *   如果图中存在具有 `node_id` 的节点，则返回其属性。
        *   `get_edge_info(self, source, target)`:
            *   如果存在 `source` 和 `target` 之间的边，则返回其属性。
*   **功能：**
    1.  提供 `networkx` 的简单包装器，以表示最初以字典格式定义的 KG。
    2.  提供基本的 DFS 遍历机制。
    3.  允许查询节点和边属性。
*   **依赖：** `config.FLAGS` (虽然在代码片段中没有直接使用，但 KG 相关文件通常可能会引用它)、`numpy` (as `np`，没有直接使用，但在图上下文中很常见)、`networkx` (as `nx`)、`saver.saver` (用于日志记录)、`typing`、`dataclasses`。
*   **用途：** 通过传递 KG 数据 (例如, 从文件加载并处理成预期的字典格式) 来创建 `KGTraversal` 实例。然后将此实例传递给各种上下文生成器类 (如 `PathBasedContextGenerator`、`MotifContextGenerator` 等，通常通过 `doc_KG_processor.create_context_generators`)。这些生成器使用 `KGTraversal` 实例的 `self.graph` 属性 (即 `networkx.Graph` 对象) 来执行其特定的图分析和上下文提取任务。外部添加的 `signal_to_node_map` 属性也经常被这些生成器使用。

---

## `src/main.py`

*   **目的：** 此脚本充当应用程序的主要入口点。它使用 `FLAGS.task` 变量 (来自 `config.py`) 来确定要执行的主要操作：生成计划 (`gen_plan`)、构建知识图谱 (`build_KG`) 或使用/查询知识图谱 (`use_KG`)。
*   **主要函数：**
    *   `main()`:
        *   读取 `FLAGS.task`。
        *   如果 `FLAGS.task == 'gen_plan'`，则调用 `gen_plan()` (来自 `gen_plan.py`)。
        *   如果 `FLAGS.task == 'build_KG'`，则调用 `build_KG()` (来自 `gen_KG_graphRAG.py`)。
        *   如果 `FLAGS.task == 'use_KG'`，则调用 `use_KG()` (来自 `use_KG.py`)。
        *   如果任务未知，则引发 `NotImplementedError`。
    *   `if __name__ == '__main__':` 块：
        *   初始化 `OurTimer` 以测量总执行时间。
        *   在 `try...except` 块中调用 `main()` 以捕获并记录任何异常。
        *   使用 `saver.log_info()` 和 `saver.save_exception_msg()` 记录错误。
        *   使用 `report_save_dir()` 打印总执行时间和日志目录路径。
        *   调用 `saver.close()` 完成日志记录。
*   **功能：**
    1.  基于配置标志充当简单的命令行调度程序。
    2.  为选定的任务提供顶层错误处理和日志记录。
    3.  测量并报告所选任务的总执行时间。
*   **依赖：** `gen_plan.gen_plan`, `gen_KG_graphRAG.build_KG`, `use_KG.use_KG`, `saver.saver`, `config.FLAGS`, `utils.OurTimer`, `utils.get_root_path`, `utils.report_save_dir`, `traceback`, `sys`。
*   **用途：** 执行此脚本以运行项目的主要功能之一。具体行为通过在运行 `python src/main.py` 之前设置 `src/config.py` 中的 `task` 变量来控制。

---

## `src/rtl_analyzer.py`

*   **目的：** 此脚本定义了 `RTLAnalyzer` 类，旨在解析和分析 Verilog/SystemVerilog RTL (寄存器传输级) 代码。它旨在从设计中提取结构和行为信息，例如模块定义、端口、信号、实例、FSM (有限状态机) 和数据流。然后，这些信息可用于增强其他过程，例如 KG 构建或 SVA 生成。
*   **主要类和方法：**
    *   `RTLAnalyzer`:
        *   `__init__(self, design_dir: str, verbose: bool = False)`:
            *   使用 `design_dir` (RTL 文件路径) 和 `verbose` 标志进行初始化。
            *   设置各种内部字典和集合以存储分析结果：`file_info`、`module_info`、`control_flow`、`fsm_info`、`signal_type_info`、`primary_signals`、`data_flow_graph` (一个 `nx.DiGraph`)、`verification_suggestions`。
        *   `analyze_design()`:
            *   开始分析的主要公共方法。
            *   `process_files_in_order()`: 获取 Verilog 文件列表，尝试按依赖关系排序 (首先包含)。
            *   将所有 Verilog 文件的内容合并到一个字符串 (`combined_content`) 中，并将其写入 `_combined_rtl.v`。这样做可能是为了帮助更好地处理包含的解析器或获得全局视图。
            *   首先尝试使用 `_process_file()` 处理 `_combined_rtl.v` 文件。
            *   如果组合方法未产生模块信息，则回退到处理单个文件。
            *   调用各种辅助方法以提取特定类型的信息：
                *   `_extract_port_info_direct()`: 如果 AST 解析失败，则回退到通过正则表达式获取端口信息。
                *   `_find_module_instances_direct()`: 实例检测的回退。
                *   `_build_cross_module_dataflow()`: 连接跨模块实例的数据流。
                *   `_identify_fsms()`: 尝试通过 AST 分析和模式匹配来检测 FSM。
                *   `_analyze_primary_signal_relationships()`: 在数据流图中查找 I/O 信号之间的路径。
            *   `_print_summary()`: 打印发现的基本摘要。
            *   `_enhanced_signal_analysis()`: 触发更详细的信号和协议模式分析。
            *   为每个模块调用进一步的提取方法：`_extract_control_flow`、`_extract_assignments`、`_extract_signal_attributes`。
            *   `_print_expanded_summary()`: 打印更详细的摘要。
            *   清理临时的组合 RTL 文件。
        *   `_process_file(self, file_path: str)`:
            *   单个 Verilog 文件的核心处理逻辑。
            *   读取文件内容，由于 Pyverilog 解析问题，使用简化模型特殊处理 `sockit` 设计。
            *   `_preprocess_includes()`: 用于内联 `include` 指令的简单预处理器。
            *   使用 `pyverilog.vparser.parser.parse()` 生成 AST。
            *   `_extract_module_info(ast_output)`: 从 Pyverilog AST 中提取模块名称、端口、信号、实例、always 块和 assign 语句。
            *   如果 AST 解析失败或未产生模块，则回退到 `_extract_module_info_from_content()` (基于正则表达式)。
            *   `_extract_dataflow_info()`: 使用 `pyverilog.dataflow.dataflow_analyzer` 构建模块内信号的数据流图。
        *   `_extract_module_info_from_content(self, content)`: 基于正则表达式提取模块名称、端口和基本 always 块。
        *   `_parse_senslist(self, senslist)`: 解析敏感列表字符串。
        *   `_extract_module_info_from_simplified(self, file_path)`: 针对特殊 `sockit` 情况的基于正则表达式的提取。
        *   `_parse_width(self, width_node)`: 将 AST 宽度节点解析为字符串。
        *   `_add_to_dataflow_graph(self, signal_name: str, term, dfg)`: 将信号及其驱动项添加到 `self.data_flow_graph`。
        *   `_build_cross_module_dataflow(self)`: 迭代模块实例及其端口连接，以向 `self.data_flow_graph` 添加边，表示父模块和实例化模块信号之间的数据流。
        *   `_find_module_instances_direct(self)` & `_extract_port_info_direct(self)`: 基于正则表达式的回退。
        *   `_identify_fsms(self)`: 通过在 AST 中查找时钟敏感的 always 块、case 语句、状态寄存器命名约定以及通过文件内容上的正则表达式查找状态参数定义来检测 FSM。作为最后手段调用 `_direct_fsm_pattern_match`。
        *   `_direct_fsm_pattern_match(self)`: 使用正则表达式列表查找常见的 FSM 相关代码结构。
        *   `_analyze_primary_signal_relationships(self)`: 检查 `data_flow_graph` 中主要 I/O 信号之间的路径。
        *   `_print_summary(self)` & `_print_expanded_summary(self)`: 将分析结果打印到控制台。
        *   `_enhanced_signal_analysis(self)`: 调用 `_analyze_protocol_patterns` 或 `_direct_rtl_analysis`。
        *   `_analyze_protocol_patterns(self)`: 根据端口命名约定识别常见的信号类型 (时钟、复位、数据、控制、握手)。调用 `_generate_verification_suggestions`。
        *   `_generate_verification_suggestions(self, module_name, patterns)`: 根据识别的协议模式 (例如, 复位行为、CDC、数据稳定性、握手) 创建通用验证建议。
        *   `_direct_rtl_analysis(self)`: 当基于 AST 的解析不足或失败时，使用正则表达式直接从文件内容中查找模块、端口、always 块、case 语句和分配的回退分析方法。
        *   `get_analysis_results(self)`: 返回包含所有收集的分析信息的字典。
        *   `_extract_control_flow(self, module_name, file_path)`: 基于正则表达式提取 if/case/loop 语句。
        *   `_extract_assignments(self, module_name, file_path)`: 基于正则表达式提取连续和过程分配。
        *   `_extract_signal_attributes(self, module_name, file_path)`: 基于正则表达式提取信号类型 (wire, reg, logic, parameters)、宽度和初始值。
    *   `process_files_in_order(design_dir)`: (类外部)
        *   尝试通过基于 `` `include`` 指令和模块实例化构建依赖图来确定 Verilog 文件的处理顺序。
        *   使用 `nx.topological_sort`。如果检测到循环依赖，则回退到未排序列表。
    *   `main()` (`if __name__ == "__main__":` 中):
        *   命令行参数解析 (`argparse`)，用于 `design_dir` 和 `verbose`。
        *   调用 `process_files_in_order`。
        *   实例化 `RTLAnalyzer` 并调用 `analyze_design()`。
*   **功能：**
    1.  解析 Verilog/SystemVerilog 文件的层次结构。
    2.  尝试使用 Pyverilog 构建抽象语法树 (AST) 并提取结构信息 (模块、端口、实例、信号、always 块、分配)。
    3.  使用 Pyverilog 的数据流分析器来理解模块内的信号依赖关系。
    4.  构建跨模块数据流图。
    5.  当 AST 解析不完整或失败时，采用基于正则表达式的回退和直接内容分析，使其能够适应某些非标准 Verilog 或解析限制。
    6.  使用 AST 特征和模式匹配的组合来识别潜在的有限状态机 (FSM)。
    7.  根据信号命名约定分析协议模式并生成通用验证建议。
    8.  提供所分析设计的详细摘要。
    9.  输出 (`get_analysis_results()`) 是一个结构化字典，可供其他工具使用 (例如, 用于优化 KG 或为 LLM 提供上下文)。
*   **依赖：** `utils_LLM.count_prompt_tokens`, `os`, `sys`, `re`, `argparse`, `typing`, `networkx` (as `nx`), `pyverilog.vparser.parser`, `pyverilog.vparser.ast`, `pyverilog.dataflow.dataflow_analyzer`, `pyverilog.dataflow.optimizer`, `pyverilog.dataflow.walker`。
*   **用途：** 此脚本/类用于直接从源代码了解 RTL 设计的结构和行为。提取的信息 (`gen_plan.py` 中的 `rtl_knowledge`) 可用于：
    *   优化知识图谱 (例如, 添加特定于 RTL 的节点和边)。
    *   在生成计划或 SVA 时为 LLM 提供直接的 RTL 上下文。
    *   帮助识别验证的关键领域。
    作为独立脚本运行时，它会将分析结果打印到控制台。

---

## `src/rtl_kg.py`

*   **目的：** 此脚本专注于处理 RTL (寄存器传输级) 设计信息，特别是使用 `RTLAnalyzer` 类 (来自 `rtl_analyzer.py`)，然后使用 `networkx` 将此信息转换为结构化知识图谱 (KG)。它还包括用于导出此 KG 的实用程序。
*   **主要函数：**
    *   `extract_rtl_knowledge(design_dir, output_dir=None, verbose=False)`:
        *   使用 `design_dir` 实例化 `RTLAnalyzer`。
        *   调用 `analyzer.analyze_design()` 执行 RTL 分析。
        *   将 `analyzer` 的结果构建为字典 `rtl_knowledge`。此字典包括：
            *   `design_info`: 基本计数 (文件、模块、主要信号)。
            *   `modules`: 来自 `analyzer.module_info` 的每个模块的详细信息。
            *   `files`: 有关已处理文件的信息。
            *   `fsm_info`: 检测到的 FSM。
            *   `protocol_patterns`: 识别的协议模式。
            *   `verification_points`: 基于分析的验证建议 (由 `extract_verification_points` 生成)。
            *   `signal_types`: 信号的详细类型信息。
            *   `primary_signals`: I/O 端口列表。
            *   `combined_content`: 所有 RTL 文件的串联内容。
        *   如果指定了 `output_dir`，则将 `analyzer.data_flow_graph` 保存为 GraphML，并将 `rtl_knowledge` 字典保存为 JSON (使用 `make_json_serializable` 处理不可序列化的类型)。
        *   返回 `rtl_knowledge` 字典。
    *   `extract_verification_points(analyzer)`:
        *   提取或生成验证点。
        *   如果存在 `analyzer.verification_suggestions` (来自 `RTLAnalyzer`)，则使用它们。
        *   否则，它会根据 `analyzer.protocol_patterns` 生成基本验证点 (例如, 复位行为、数据稳定性、握手)。
    *   `build_knowledge_graph(rtl_knowledge)`:
        *   以 `rtl_knowledge` 字典作为输入。
        *   构建一个 `nx.MultiDiGraph` (一个可以在两个节点之间具有多条边的有向图)。
        *   使用辅助函数 `get_node_id` 为 KG 节点创建唯一的整数 ID，将它们从原始字符串标识符 (例如, "module:mod\_name", "port:mod\_name.port\_name") 映射过来。
        *   向 KG 添加节点：
            *   模块 (`type="module"`)
            *   端口 (`type="port"`), 将其链接到其模块。
            *   FSM (`type="fsm"`), 将其链接到其模块。
            *   协议模式 (`type="protocol_pattern"`), 链接到模块和信号。
            *   验证点 (`type="verification_point"`), 链接到模块和信号。
            *   控制流结构 (if, case, loop) (`type="control_flow"`), 链接到模块和相关信号。
            *   赋值 (`type="assignment"`), 链接到模块和相关信号 (LHS 和 RHS)。
            *   内部信号 (`type="signal"`), 如果尚不是端口，则链接到模块。
        *   添加表示关系的边，如 "input\_to", "outputs", "instantiates", "drives", "part\_of", "found\_in", "includes", "targets", "involves", "references", "switches\_on", "assigns\_to", "used\_in"。每条边都有一个唯一的 ID (例如, "e0", "e1")。
        *   将 `node_id_map` (原始字符串 ID 到整数 ID) 和 `reverse_id_map` 存储为图属性。
        *   返回构建的 `nx.MultiDiGraph`。
    *   `export_graph_to_graphml(kg, output_path, simplify=False)`:
        *   将 `kg` 导出到 GraphML 文件。
        *   通过使用 `make_json_serializable` 将复杂属性 (字典、列表) 转换为 JSON 字符串来处理潜在的序列化问题。
        *   具有 `simplify` 选项，可创建具有较少属性的版本，可能为了更好地与某些可视化工具兼容。
        *   如果完整导出失败，则包括回退以保存最小图。
    *   `save_knowledge_graph(kg, output_path, output_dir)`: (似乎与 `export_graph_to_graphml` 有些重复，但处理方式可能略有不同，或者是早期版本)。它尝试另存为 GraphML，如果失败则回退到 JSON。
    *   `make_json_serializable(obj)`: 通过字符串化非标准类型，将对象 (字典、列表、集合等) 递归转换为 JSON 可序列化表示。
    *   `save_ultra_simplified_gephi_kg(kg, output_path)`: 创建一个非常简化的 KG 版本，专门用于 Gephi 可视化，确保基本属性如 'id', 'label', 'type'。
    *   `write_graphml_with_unique_edge_ids(G, path)`: 一个实用程序，可确保 GraphML 输出中的所有边都具有唯一的 'id' 属性，这对于 Gephi 等工具可能很重要。
    *   `main()` (`if __name__ == "__main__":` 中):
        *   `design_dir`、`output_dir`、`verbose` 的参数解析。
        *   调用 `extract_rtl_knowledge`。
        *   调用 `build_knowledge_graph`。
        *   调用 `export_graph_to_graphml` 和 `save_knowledge_graph`。
        *   尝试使用 `matplotlib` 和 `nx.spring_layout` 生成 KG 的 PNG 可视化 (对于大型图，这可能很慢或占用大量内存)。
        *   调用 `save_ultra_simplified_gephi_kg`。
*   **功能：**
    1.  使用 `RTLAnalyzer` 解析 RTL 代码并提取详细信息。
    2.  将提取的 RTL 信息转换为形式知识图谱 (一个 `networkx.MultiDiGraph`)。节点表示模块、端口、信号、FSM、赋值等，边表示它们之间的关系。
    3.  提供实用程序以 GraphML 格式导出生成的 KG，并考虑到与 Gephi 等可视化工具的兼容性。
    4.  还可以将提取的 RTL 知识另存为 JSON 文件。
*   **依赖：** `saver.saver`, `os`, `sys`, `json`, `networkx` (as `nx`), `matplotlib.pyplot` (可选, 用于可视化), `rtl_analyzer.RTLAnalyzer`, `argparse`。
*   **用途：** 此脚本可能作为独立工具运行，也可能作为更大流程的一部分运行 (例如, 如果任务涉及创建基于 RTL 的 KG，则由 `build_KG` 调用)。`extract_rtl_knowledge` 函数也由 `rtl_parsing.py` 用于获取 RTL 信息以优化基于规范的 KG。然后，输出的 KG (GraphML 或 JSON) 可供其他分析工具或上下文生成器使用。

---

## `src/rtl_parsing.py`

*   **目的：** 此脚本专注于将从 RTL (寄存器传输级) 分析中获得的知识与现有的基于规范的知识图谱 (KG) 相集成。主要目标是创建一个更丰富、更全面的 KG，该 KG 结合了来自设计规范和实际硬件实现的见解。
*   **主要函数：**
    *   `refine_kg_from_rtl(spec_kg: nx.Graph) -> Tuple[nx.Graph, dict]`:
        *   主要的公共函数。接受现有的 `spec_kg` (可能从设计文档构建而来)。
        *   调用 `link_spec_and_rtl_graphs` 执行核心链接逻辑。
        *   调用 `analyze_graph_connectivity` 报告组合图的结构。
        *   返回 `combined_kg` 和从 `extract_rtl_knowledge` 获得的 `rtl_knowledge` 字典。
    *   `link_spec_and_rtl_graphs(spec_kg: nx.Graph, design_dir: str) -> Tuple[nx.Graph, dict]`:
        *   `extract_rtl_knowledge(design_dir, ...)`: 调用 `rtl_kg.py` 中的函数来分析 `FLAGS.design_dir` 中的 RTL 文件并获取结构化的 `rtl_knowledge`。
        *   `build_knowledge_graph(rtl_knowledge)`: 调用 `rtl_kg.py` 中的函数，仅从提取的 RTL 信息构建一个新的 KG (`rtl_kg`)。
        *   通过复制 `spec_kg` 创建 `combined_kg`。
        *   将 `rtl_kg` 中的节点和边添加到 `combined_kg` 中，为 RTL 节点 ID 添加 "rtl\_" 前缀以避免冲突，并添加 'source': 'rtl' 属性。
        *   `link_modules_to_spec(combined_kg, rtl_node_mapping)`: 在 RTL 模块节点和 `combined_kg` 中相关的规范节点之间创建链接。它添加一个 "design\_root" 节点，并将规范节点和 RTL 模块节点连接到该节点。如果模块名称出现在规范文本中，它还会尝试将规范文本节点链接到 RTL 模块节点。
        *   `link_signals_to_spec(combined_kg, rtl_node_mapping)`: 如果信号名称出现在规范文本中，则在 RTL 信号/端口节点和规范文本节点之间创建链接。
        *   `ensure_graph_connectivity(combined_kg)`: 检查图是否已连接，如果未连接，则将组件连接到 "knowledge\_root" 节点。
        *   返回 `combined_kg` 和 `rtl_knowledge`。
    *   `link_modules_to_spec(...)`: (上面已详细说明) 在规范节点和 RTL 模块节点之间创建 "describes" 关系。
    *   `link_signals_to_spec(...)`: (上面已详细说明) 在规范节点和 RTL 信号节点之间创建 "references" 关系。
    *   `ensure_graph_connectivity(kg: nx.Graph)`: (上面已详细说明) 通过添加根节点并将断开连接的组件链接到该节点来确保图已连接。
    *   `analyze_graph_connectivity(kg: nx.Graph)`: 打印有关图连通性的信息，包括连通分量的数量、RTL 和规范部分之间的桥以及高度数 (中心) 节点。
*   **功能：**
    1.  使用 `rtl_kg.extract_rtl_knowledge` 编排 RTL 代码的分析。
    2.  使用 `rtl_kg.build_knowledge_graph` 从此 RTL 分析中构建单独的 KG。
    3.  将此 RTL KG 与现有的基于规范的 KG 合并。
    4.  在组合 KG 的规范部分和 RTL 部分之间建立显式链接，主要通过以下方式：
        *   创建通用的 "design\_root" 以桥接规范和 RTL 模块层次结构。
        *   将 RTL 中找到的模块名称和信号名称与规范节点中的文本进行匹配。
    5.  分析并报告生成的组合图的连通性。
*   **依赖：** `utils_gen_plan.count_tokens_in_file` (虽然在此文件的提供的代码片段中没有直接使用，但已导入)、`os`, `re`, `sys`, `networkx` (as `nx`), `config.FLAGS`, `saver.saver`, `rtl_kg.extract_rtl_knowledge`, `rtl_kg.build_knowledge_graph`。
*   **用途：** 当 `FLAGS.refine_with_rtl` 为 true 时，`gen_plan.py` 会调用此脚本。它接受从规范构建的 KG，并通过解析相应的 RTL 代码、从 RTL 创建新的 KG，然后智能地合并和链接两者来丰富它。生成的 `combined_kg` 提供了对设计的更全面的视图，将高级需求与低级实现细节联系起来。然后，这个更丰富的 KG 可供上下文生成器用于更有效的提示增强。

---

## `src/saver.py`

*   **目的：** 此脚本定义了一个 `Saver` 类，负责在实验或运行期间进行全面的日志记录、结果保存和输出目录管理。它集中管理信息 (文本日志、绘图、pickle 对象、模型信息、配置) 保存到磁盘的方式。它还包括一个 `MyTimer` 类。
*   **主要类和方法：**
    *   `MyTimer`:
        *   `__init__(self)`: 记录开始时间。
        *   `elapsed_time(self)`: 返回以整数分钟为单位的已用时间。
    *   `Saver`:
        *   `__init__(self)`:
            *   首先，它导入并调用 `parse_command_line_args` (来自 `command_line_args.py`)，该函数会更新 `config.FLAGS`。这意味着 `Saver` 的行为在实例化时会受到命令行参数的影响。
            *   使用 `FLAGS.task`、时间戳、主机名和用户构建 `self.logdir` 路径。此目录将存储运行的所有输出。
            *   创建 `logdir` 和子目录 `plotdir` (用于绘图) 和 `objdir` (用于 pickle 对象)。
            *   打开 `model_info.txt` 进行写入。
            *   调用 `_log_model_info()` 保存初始模型/配置详细信息。
            *   调用 `_save_conf_code()` 保存 `config.py` 和 `FLAGS` 对象的副本。
            *   初始化 `self.timer = MyTimer()`。
            *   初始化 `self.stats` (一个 `defaultdict(list)`) 以累积统计信息。
        *   `get_log_dir()`, `get_plot_dir()`, `get_obj_dir()`: 返回相应目录的路径。
        *   `log_list_of_lists_to_csv(...)`, `log_dict_of_dicts_to_csv(...)`: 将数据保存到 CSV 文件。
        *   `save_emb_accumulate_emb(...)`, `save_emb_save_to_disk(...)`, `save_emb_dict(...)`: 用于保存嵌入数据的方法 (可能用于机器学习模型)，可能在保存前累积它们。
        *   `log_dict_to_json(...)`: 将字典保存到 JSON 文件。
        *   `log_model_architecture(self, model)`: 将模型架构 (例如, PyTorch 模型字符串) 和估计大小写入 `model_info.txt`。
        *   `log_info(self, s, silent=False, build_str=None)`:
            *   主要的日志记录方法。将 `s` 打印到控制台 (除非 `silent`)。
            *   将 `s` 写入 `logdir` 中的 `log.txt`。
            *   可以将列表/字典漂亮地打印为 JSON。
            *   可以选择附加到 `build_str`。
        *   `log_info_once(...)`, `log_info_at_most(...)`: `log_info` 的变体，用于防止重复消息或限制消息频率。
        *   `info(...)`, `error(...)`, `warning(...)`, `debug(...)`: 定时日志记录方法，使用已用时间前缀写入 `log.txt`、`error.txt` 或 `debug.txt`。
        *   `save_dict(self, d, p, subfolder='')`: 将字典另存为 pickle 文件。
        *   `_save_conf_code(self)`: 保存 `config.py` 的内容 (通过 `utils.py` 中的 `extract_config_code` 提取) 和当前的 `FLAGS` 对象。
        *   `save_graph_as_gexf(self, g, fn)`: 将 `networkx` 图另存为 GEXF 文件。
        *   `save_overall_time(...)`, `save_exception_msg(...)`: 保存总时间和异常消息。
        *   `_log_model_info(self)`: 将详细的模型和配置信息 (来自 `FLAGS`) 写入 `model_info.txt`。
        *   `_save_to_result_file(...)`: 将各种对象/字符串附加到 `results.txt`。
        *   `save_stats(...)`, `print_stats()`: 累积命名统计信息并在末尾打印摘要 (例如, 均值、标准差)。
        *   `close(self)`: 关闭打开的文件处理程序并打印最终统计信息。
    *   `NoOpContextManager`: 一个不执行任何操作的简单上下文管理器，在内部使用。
    *   全局 `saver` 实例：`saver = Saver()` 在末尾实例化，使全局 `saver` 对象可供整个项目导入和使用。
*   **功能：**
    1.  提供一种集中和标准化的方式来记录信息和保存在程序运行期间的各种类型的数据 (文本、对象、绘图、配置)。
    2.  为每次运行自动创建一个唯一的、带时间戳的日志目录。
    3.  在初始化时处理命令行参数解析，以配置其行为和全局 `FLAGS`。
    4.  包括用于计时和基本统计报告的实用程序。
    5.  提供不同级别和样式的日志记录 (信息、错误、调试、一次、有限频率)。
*   **依赖：** `config.FLAGS`, `command_line_args.parse_command_line_args`, `utils.py` 中的各种函数 (与时间、路径、保存、绘图、系统信息相关), `json`, `collections.OrderedDict`, `collections.defaultdict`, `pprint`, `os.path.join`, `os.path.dirname`, `os.path.basename`, `torch` (有条件地, 用于张量检查), `networkx` (用于 GEXF 保存), `numpy`, `time`, `csv`。
*   **用途：** 全局 `saver` 对象被许多其他脚本导入 (例如, `from saver import saver`)。然后，其方法如 `saver.log_info(...)`、`saver.save_dict(...)` 等用于所有输出日志记录和数据保存需求，确保输出文件的一致性和组织性。`print = saver.log_info` 模式很常见，将标准打印语句重定向到日志记录机制。

---

## `src/sva_extraction.py`

*   **目的：** 此脚本提供从文本块 (通常是 LLM 的输出) 中提取 SystemVerilog 断言 (SVA) 的函数。它采用多种基于正则表达式的策略来识别和解析 SVA 语句。
*   **主要函数：**
    *   `extract_svas_strategy1(result: str) -> List[str]`: 查找包装在 "SVA: \`\`\` ... \`\`\`" 中的块，然后从这些块中提取 SVA 模式 (`@(posedge...);`)。
    *   `extract_svas_strategy2(result: str) -> List[str]`: 直接在输入 `result` 字符串中的任何位置匹配核心 SVA 模式 (`@(posedge...);`)。
    *   `extract_svas_strategy3(result: str) -> List[str]`: 查找类似 "SVA for Plan X: ... \`\`\` ... \`\`\`" 的块，并从代码块中提取 SVA。
    *   `extract_svas_strategy4(result: str) -> List[str]`: 按 "SVA for Plan X:" 标头拆分输入，然后在每个结果部分中搜索 SVA 代码块。
    *   `extract_svas_strategy5(result: str) -> List[str]`: 查找 "```systemverilog ... ```" 块，然后提取 `property ... endproperty; assert property(...);` 结构。
    *   `extract_svas_strategy6(result: str) -> List[str]`: 在常规代码块中查找注释掉的断言 (例如, `// assert property(...)`)。
    *   `extract_svas_strategy7(result: str) -> List[str]`: 在常规代码块中查找 `assert property(...);` 行，不一定是注释掉的。
    *   `extract_svas_strategy8(result: str) -> List[str]`: 匹配更结构化的 `property P; @(posedge clk) ...; endproperty assert property(P);` 模式。
    *   `extract_svas_strategy9(result: str) -> List[str]`: 匹配 `assert property (@(posedge clk) ...;)` 模式。
    *   `clean_sva(sva: str) -> str`: 一个辅助函数，用于从 SVA 字符串中删除多行 (`/* ... */`) 和单行 (`// ...`) 注释并规范化空白。
    *   `extract_svas_from_block(block: str) -> List[str]`:
        *   主要的公共函数。
        *   迭代上面定义的策略字典。
        *   对输入 `block` 调用每个策略函数。
        *   将任何策略提取的所有唯一 SVA 收集到一个 `set` 中以避免重复。
        *   使用 `clean_sva` 清理每个提取的 SVA。
        *   打印每个提取策略找到的 SVA 数量的摘要。
        *   返回唯一的、已清理的 SVA 列表。
*   **功能：**
    1.  提供一种从可能混乱或格式多变的文本 (如 LLM 输出) 中提取 SVA 的稳健机制。
    2.  采用多策略方法，即使格式不完美或不一致，也能增加正确识别 SVA 的机会。
    3.  通过删除注释和标准化空白来清理提取的 SVA。
    4.  报告每种提取策略的有效性。
*   **依赖：** `re`, `typing.List`, `typing.Dict`, `typing.Optional`, `typing.Set`, `saver.saver` (用于 `print = saver.log_info`)。
*   **用途：** 此脚本主要在 `gen_plan.py` 中使用，在 LLM 生成预期包含 SVA 的文本之后。调用 `extract_svas_from_block` 函数来解析此文本并检索干净的 SVA 字符串列表，然后可以将其写入文件以进行形式验证。

---

## `src/use_KG.py`

*   **目的：** 此脚本提供命令行界面或函数，以使用 GraphRAG 工具查询预构建的知识图谱 (KG)。它专为直接交互或测试 KG 而设计。
*   **主要函数：**
    *   `use_KG()`:
        *   此任务的主要函数。
        *   从 `FLAGS` 检索 `KG_root` (KG 工件的路径)、`graphrag_method` (例如, 'local') 和 `query` 字符串。
        *   构建运行 GraphRAG 查询的命令：`python -m graphrag.query --data <KG_root> --method <method> "<query>"`。它还将 `PYTHONPATH` 设置为包含 `FLAGS.graphrag_local_dir`。
        *   使用 `subprocess.Popen` 执行此命令并流式传输其输出。
        *   捕获完整输出，并假定输出的最后一行是查询的答案。
        *   使用 `utils.OurTimer` 进行计时。
*   **功能：**
    1.  提供一种从命令行或以编程方式对 GraphRAG KG 执行查询的方法。
    2.  处理 GraphRAG 查询命令的构建和执行。
    3.  打印查询输出并尝试提取最终答案。
*   **依赖：** `os`, `subprocess`, `shutil`, `json`, `re`, `pathlib.Path`, `PyPDF2`, `logging`, `utils.OurTimer`, `saver.saver`, `config.FLAGS`。
*   **用途：** 当 `FLAGS.task` 设置为 `'use_KG'` 时运行此脚本。它是一个通过 GraphRAG 的查询界面与生成的知识图谱进行交互并获取响应的实用程序。

---

## `src/utils.py`

*   **目的：** 此脚本是一个通用实用程序模块，包含项目各处使用的各种辅助函数和类。这些范围从路径操作和文件操作到绘图、数据结构操作和系统信息检索。
*   **主要组件：**
    *   **路径助手：** `get_root_path()`, `get_log_path()`, `get_file_path()`, `get_save_path()`, `get_src_path()`, `create_dir_if_not_exists()`, `proc_filepath()`, `append_ext_to_filepath()`。
    *   **数据处理和保存/加载：**
        *   `save(obj, filepath, ...)` & `load(filepath, ...)`: 默认使用 Klepto 的通用保存/加载函数。
        *   `save_klepto(...)`, `load_klepto(...)`: 特定的 Klepto 保存/加载。
        *   `save_pickle(obj, filepath, ...)` & `load_pickle(filepath, ...)`: 基于 Pickle 的保存/加载。
    *   **列表/序列助手：** `argsort(seq)`, `sorted_nicely(l, reverse=False)`。
    *   **执行和系统：**
        *   `exec_cmd(cmd, timeout=None, exec_print=True)`: 执行具有可选超时的 shell 命令。
        *   `get_ts()`, `set_ts(ts)`, `get_current_ts(zone='US/Pacific')`: 时间戳生成。
        *   `timeout` 类：使用信号的函数超时的上下文管理器。
        *   `get_user()`, `get_host()`: 获取用户名和主机名。
        *   `get_best_gpu(...)`, `get_gpu_info()`, `print_gpu_free_mem_info(...)`: GPU 选择和信息实用程序 (使用 `nvidia-smi`)。
        *   `format_file_size(...)`: 将字节格式化为人类可读的大小。
        *   `report_save_dir(save_dir)`: 计算目录中的总大小和文件计数。
    *   **图助手 (NetworkX)：** `assert_valid_nid()`, `assert_0_based_nids()`, `node_has_type_attrib()`, `print_g()`, `create_edge_index()`。
    *   **绘图 (Matplotlib/Seaborn)：** `plot_dist()`, `_analyze_dist()`, `plot_scatter_line()`, `plot_points()`, `multi_plot_dimension()`, `plot_scatter_with_subplot()`, `plot_scatter_with_subplot_trend()`, `plot_points_with_subplot()`。
    *   **PyTorch 相关 (简单 MLP)：**
        *   `MLP` 类：一个基本的多层感知器模型。
        *   `MLP_multi_objective` 类：一个具有多个输出头以进行多任务学习的 MLP。
        *   `create_act(act, ...)`: 创建激活函数层。
    *   **其他：**
        *   `C` 类：一个简单的计数器。
        *   `OurTimer` 类：用于对代码块计时并记录持续时间。
        *   `format_seconds(seconds)`: 将秒格式化为人类可读的字符串 (年、月、日等)。
        *   `random_w_replacement(input_list, k=1)`: 带替换的随机抽样。
        *   `get_sparse_mat(...)`: 创建 SciPy 稀疏矩阵。
        *   `prompt(str, options=None)`, `prompt_get_cpu()`, `prompt_get_computer_name()`: 用户输入提示。
        *   `parse_as_int(s)`: 将字符串安全地解析为整数。
        *   `get_model_info_as_str(FLAGS)`, `extract_config_code()`: 用于保存配置详细信息。
        *   `TopKModelMaintainer` 类：用于在训练期间根据验证损失保存前 K 个最佳模型的助手。
        *   `coo_to_sparse(...)`: 将 SciPy COO 矩阵转换为 PyTorch 稀疏张量。
        *   `estimate_model_size(model, label, saver)`: 估计并记录 PyTorch 模型的大小。
        *   `create_loss_dict_get_target_list(...)`, `update_loss_dict(...)`: 用于在训练中管理损失字典的助手。
        *   `format_loss_dict(...)`: 格式化损失字典以供打印。
*   **功能：** 此模块提供了常用函数的工具包，可防止代码重复并在文件 I/O、日志记录、路径管理、绘图和基本机器学习模型组件等领域提高一致性。
*   **依赖：** `pytz`, `datetime`, `pathlib.Path`, `sys`, `scipy`, `scipy.sparse`, `numpy`, `signal`, `pickle`, `random`, `requests` (虽然在可见代码中未使用), `re`, `time`, `threading.Timer`, `subprocess`, `klepto`, `collections.OrderedDict`, `socket.gethostname`, `os` (makedirs, system, environ, remove, path functions), `networkx`, `seaborn`, `matplotlib.pyplot`, `torch`, `torch.nn`, `scipy.stats.mstats`, `copy.deepcopy`, `config.FLAGS` (隐式地, 因为某些函数可能会使用它)。
*   **用途：** `utils.py` 中的函数和类被项目中大多数其他模块广泛导入和使用。例如，`OurTimer` 用于性能跟踪，路径函数用于定位文件，保存/加载函数用于数据持久性。

---

## `src/utils_LLM.py`

*   **目的：** 此文件旨在作为与大型语言模型 (LLM) 交互相关的通用实用函数的占位符。
*   **主要函数/类 (预期)：**
    *   `get_llm(model_name, **kwargs)`: (在 `gen_plan.py` 中引用) 预期根据 `model_name` 和其他参数初始化并返回 LLM 客户端或代理。
    *   `llm_inference(llm_agent, prompt, tag)`: (在多个文件中引用) 预期接受初始化的 `llm_agent`、`prompt` 字符串和 `tag` (用于日志记录/跟踪)，将提示发送到 LLM，并返回 LLM 的文本响应。
    *   `count_prompt_tokens(prompt_text)`: (在 `dynamic_prompt_builder.py` 中引用，通常使用 `tiktoken` 实现) 预期根据特定 LLM 的分词器计算给定 `prompt_text` 将消耗的令牌数量。
*   **当前内容：** 提供的文件当前为空，只有版权注释和占位符注释：“# Add your own LLM client”。
*   **功能 (概念性)：**
    1.  抽象 LLM API 交互的细节。
    2.  为发送提示和接收来自 LLM 的响应提供标准化接口。
    3.  提供与 LLM 交互相关的辅助函数，例如令牌计数。
*   **依赖 (概念性)：** 可能包括 `openai`、`tiktoken` 等库或其他特定于 LLM 的客户端库，并且将依赖于 `config.FLAGS` 中的配置。
*   **用途 (预期)：** `gen_plan.py`、`dynamic_prompt_builder.py` 和 `design_context_summarizer.py` 等模块导入并调用预期在此文件中定义的函数 (例如, `llm_inference`, `get_llm`) 以与 LLM 交互。

---

## `src/utils_LLM_client.py`

*   **目的：** 此文件旨在作为与大型语言模型 (LLM) API 或服务交互的特定客户端实现的占位符。
*   **当前内容：** 提供的文件当前为空，只有版权注释和占位符注释：“# Add your own LLM client”。
*   **功能 (概念性)：**
    1.  包含向 LLM 发出请求的低级代码 (例如, OpenAI API、本地 Hugging Face 模型端点等)。
    2.  处理与 LLM 通信相关的身份验证、请求格式化、响应解析和错误处理。
    3.  在 `src/utils_LLM.py` 中定义的函数 (如 `llm_inference`) 可能会调用此客户端文件中定义的函数或使用类。
*   **依赖 (概念性)：** 将在很大程度上取决于所选的 LLM 提供程序或库 (例如, `openai`, `requests`, `transformers`)。
*   **用途 (预期)：** 此模块将封装与 LLM 服务的直接通信逻辑，从而提供更清晰的关注点分离。`utils_LLM.py` 中的高级 LLM 实用函数将使用此客户端。

---

## `src/utils_gen_plan.py`

*   **目的：** 此脚本包含专门为 `gen_plan.py` 工作流程量身定制的实用函数，特别是与 SVA 生成、JasperGold 执行和形式验证结果分析相关的函数。
*   **主要函数：**
    *   `analyze_coverage_of_proven_svas(svas: List[str], jasper_reports: List[str]) -> str`:
        *   收集已证明的 SVA (基于 Jasper 报告中的 `extract_proof_status`)。
        *   创建一个组合的 SVA 文件 (`combined_proven_svas.sva`)，其中包含所有已证明的 SVA，并嵌入到设计的模块接口中。
        *   生成一个 TCL 脚本 (`coverage_analysis.tcl`) 以在这些组合的已证明 SVA 上运行 JasperGold 进行覆盖率分析。此 TCL 脚本包括初始化覆盖率、分析设计文件、详细说明、定义时钟/复位、运行证明 (有时间限制)，然后报告各种覆盖率度量 (功能、语句、翻转、表达式、分支的 stimuli/COI) 的命令。
        *   使用此 TCL 脚本执行 JasperGold。
        *   保存覆盖率报告并返回 JasperGold 运行的 stdout。
        *   包括查找原始设计 TCL 文件以提取必要命令 (如 `elaborate -top`、时钟/复位定义和设计文件名) 的逻辑。
    *   `find_and_read_original_tcl(design_dir: str) -> Tuple[str, str]`: 在 `design_dir` 中搜索 `.tcl` 文件 (可能是主要项目 TCL) 并读取其内容。
    *   `extract_top_module(tcl_content: str) -> str`: 解析 TCL 内容以查找 `elaborate -top <module_name>` 行并提取顶层模块名称。
    *   `extract_design_file(tcl_content: str) -> str`: 解析 TCL 内容 (处理行继续) 以查找 `analyze -v2k ${RTL_PATH}/<design_file.v>` 中指定的 Verilog 设计文件。
    *   `find_original_tcl_file(logdir: str) -> str`: (注意：这似乎是应该搜索 `design_dir` 的函数的重复名称。`gen_plan.py` 中的函数搜索 `design_dir`，而 `utils_gen_plan.py` 中的此函数搜索 `logdir/tcl_scripts`。这可能是疏忽或用于不同目的，也许是查找生成的 TCL)。在 `design_dir` 中搜索 `FPV_*.tcl` 的那个也存在于此文件中。
    *   `extract_clock_and_reset(tcl_content: str) -> Tuple[str, str, str]`: 解析 TCL 内容以查找 `stopat -env`、`clock` 和 `reset` 命令。
    *   `extract_proof_status(report_content: str) -> str`: 解析 JasperGold 报告文本以确定断言的证明状态 (已证明、cex、不确定、错误)。它特别关注摘要的“断言”部分。
    *   `count_tokens_in_file(file_path)`: 读取文件并使用 `tiktoken.get_encoding("cl100k_base").encode()` 计算令牌。
*   **功能：**
    1.  为使用 JasperGold 的形式验证流程提供专门的助手。
    2.  自动执行已证明 SVA 的覆盖率分析运行的设置和执行。
    3.  包括对 JasperGold 报告和 TCL 脚本的稳健解析，以提取必要的信息 (证明状态、顶层模块、设计文件、时钟/复位命令)。
    4.  提供令牌计数实用程序，这对于管理 LLM 提示大小至关重要。
*   **依赖：** `utils.get_ts`, `config.FLAGS`, `saver.saver`, `typing.Tuple`, `typing.List`, `os`, `subprocess`, `re`, `tiktoken`, `shutil`。
*   **用途：** 这些函数主要从 `gen_plan.py` 调用，以管理与 JasperGold 的交互，为其准备输入文件，并解释其输出，尤其是在覆盖率分析方面。

---

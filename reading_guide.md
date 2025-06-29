# Reading Guide for `src` Directory

This document provides a detailed reading guide for each Python script located in the `src` directory. It aims to help understand the purpose, functionality, and implementation details of each script.

## Table of Contents

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

*   **Purpose:** This script centralizes configuration settings for a project that seems to involve tasks like "generate plan" (`gen_plan`), "build Knowledge Graph" (`build_KG`), and "use Knowledge Graph" (`use_KG`). It defines various parameters based on the selected `task` and `subtask`.
*   **Key Variables:**
    *   `task`: Determines the main operation mode (e.g., `gen_plan`, `build_KG`).
    *   `subtask`: Specifies a more granular operation within a `task` (e.g., `actual_gen`, `parse_result` for `gen_plan`).
    *   `DEBUG`: A boolean flag for enabling or disabling debug mode, which often affects settings like `max_num_signals_process` and `max_prompts_per_signal`.
    *   `design_name`: Specifies the particular hardware design being processed (e.g., `apb`, `ethmac`, `uart`). Paths to design files and Knowledge Graph (KG) files are set based on this.
    *   `file_path`: Path(s) to design specification documents (often PDFs).
    *   `design_dir`: Path to the directory containing the design's RTL files.
    *   `KG_path`: Path to the pre-built Knowledge Graph file (GraphML format).
    *   `llm_engine_type`, `llm_model`: Configuration for a Large Language Model (LLM), specifying the type and model name (e.g., `gpt-4o`).
    *   `max_tokens_per_prompt`: Maximum tokens to use for LLM prompts.
    *   `use_KG`: Boolean flag to determine if the Knowledge Graph should be used.
    *   `prompt_builder`: Method for building prompts (`static` or `dynamic`).
    *   `dynamic_prompt_settings`: A nested dictionary configuring various context generation strategies if `prompt_builder` is `dynamic`. This includes settings for:
        *   `rag` (Retrieval Augmented Generation): Chunk sizes, overlaps, number of chunks (k), enabling RTL code RAG.
        *   `path_based`: Max depth for path exploration in the KG, representation style.
        *   `motif`: Settings for discovering graph motifs (e.g., handshake, pipeline).
        *   `community`: Settings for detecting communities in the KG.
        *   `local_expansion`: Max depth for BFS expansion, subgraph size limits.
        *   `guided_random_walk`: Parameters for random walks on the KG.
        *   `pruning`: Settings for LLM-based context pruning.
    *   `refine_with_rtl`: Boolean, whether to refine information using RTL.
    *   `gen_plan_sva_using_valid_signals`, `valid_signals`: Configuration for generating SystemVerilog Assertions (SVA) based on specified valid signals.
    *   `generate_SVAs`: Boolean, whether to generate SVAs.
    *   `load_dir` (for `parse_result` subtask): Directory to load results from.
    *   `input_file_path` (for `build_KG` task): Path(s) to input documents for KG construction.
    *   `env_source_path`, `settings_source_path`, `entity_extraction_prompt_source_path` (for `build_KG` task): Paths to environment, settings, and prompt files for GraphRAG.
    *   `KG_root`, `graphrag_method`, `query` (for `use_KG` task): Configuration for querying the KG.
    *   `graphrag_local_dir`: Path to a local GraphRAG repository.
    *   `ROOT`, `repo_name`, `local_branch_name`, `commit_sha`: Git repository information, automatically determined.
    *   `FLAGS`: A `SimpleNamespace` object that aggregates most of the defined variables, making them easily accessible (e.g., `FLAGS.design_name`).
*   **Functionality:**
    1.  Sets a primary `task` (e.g., 'gen\_plan').
    2.  Based on the `task`, it sets further configurations. For 'gen\_plan', it involves selecting a `design_name` and then setting up paths, LLM parameters, and strategies for using a Knowledge Graph (KG) and document retrieval.
    3.  For 'build\_KG', it sets up paths for input documents and GraphRAG configuration files.
    4.  For 'use\_KG', it configures KG access and a query.
    5.  It uses `pathlib.Path` for robust path manipulation.
    6.  It dynamically creates a `FLAGS` object (a `SimpleNamespace`) at the end, which contains all the configuration variables defined in the script. This allows other modules to import `FLAGS` and access configuration settings easily (e.g., `from config import FLAGS`).
    7.  Includes placeholders like `/<path>/<to>/...` which need to be replaced with actual paths in a real environment.
    8.  Retrieves user and hostname, and Git repository details (repo name, branch, commit SHA).
*   **Dependencies:** `types.SimpleNamespace`, `pathlib.Path`, `utils.get_user`, `utils.get_host`, `collections.OrderedDict`, `git` (optional, with error handling if not found).
*   **How it's used:** This file is likely imported by most other scripts in the `src` directory to access global configuration settings.

---

## `src/context_generator_BFS.py`

*   **Purpose:** This script defines the `LocalExpansionContextGenerator` class, which is responsible for generating context by performing a Breadth-First Search (BFS) expansion on a Knowledge Graph (KG) starting from specified interface signals or nodes. The goal is to create a textual description of the local neighborhood of these starting points in the KG.
*   **Key Classes and Methods:**
    *   `LocalExpansionContextGenerator`:
        *   `__init__(self, kg_traversal: 'KGTraversal')`:
            *   Initializes with a `KGTraversal` object (presumably providing access to the graph `self.G` and potentially a `signal_to_node_map`).
            *   Sets `max_depth` for BFS from `FLAGS.dynamic_prompt_settings['local_expansion']['max_depth']`.
        *   `get_contexts(self, start_nodes: List[str]) -> List[ContextResult]`:
            *   The main public method. Takes a list of `start_nodes` (signal names or node IDs).
            *   Maps signal names to actual graph node IDs using `self.signal_to_node_map` if available.
            *   For each valid start node:
                *   Calls `_bfs_expansion` to get a local subgraph.
                *   Calls `_calculate_subgraph_metrics` to get metrics about this subgraph.
                *   Calls `_describe_local_subgraph` to generate a textual description.
                *   Calls `_calculate_context_score` to score the generated context.
                *   Wraps the description, score, and metadata into a `ContextResult` object (imported from `context_pruner`).
            *   Returns a list of `ContextResult` objects.
        *   `_bfs_expansion(self, start_node: str, max_depth: int) -> Set[str]`:
            *   Performs a BFS starting from `start_node` up to `max_depth`.
            *   Limits the expansion if the number of visited nodes reaches `max_subgraph_size` (from `FLAGS`).
            *   Returns a set of node IDs in the expanded subgraph.
        *   `_calculate_subgraph_metrics(self, nodes: Set[str]) -> Dict`:
            *   Takes a set of subgraph nodes.
            *   Calculates metrics like number of nodes, number of edges, density, node type distribution, and average clustering coefficient using `networkx` functions on `self.G.subgraph(nodes)`.
            *   Returns a dictionary of these metrics.
        *   `_calculate_context_score(self, nodes: Set[str], start_node: str, metrics: Dict) -> float`:
            *   Calculates a relevance score (0.0 to 1.0) for the generated context based on subgraph size, diversity of node types, and density.
        *   `_describe_local_subgraph(self, nodes: Set[str], start_node: str, metrics: Dict) -> str`:
            *   Generates a detailed textual description of the local subgraph.
            *   Includes information about the start node (name, type, module).
            *   Lists subgraph metrics (size, density, clustering).
            *   Shows node type distribution.
            *   Lists key signals, connected modules, and related registers within the subgraph (limited to a few examples).
            *   Describes direct connections of the `start_node`, including relationship types derived from edge attributes.
            *   Provides "Analysis insights" based on node name keywords (e.g., identifying control signals like 'clk', 'reset', 'enable', or data signals) and connectivity (e.g., identifying hub nodes).
*   **Data Structures:**
    *   `ContextResult`: A named tuple or class (imported from `context_pruner`) likely holding `text`, `score`, `source_type`, and `metadata`.
*   **Functionality:**
    1.  Identifies relevant portions of a larger Knowledge Graph by exploring locally around specified entry points (signals).
    2.  Uses BFS for this local exploration, constrained by depth and subgraph size.
    3.  Quantifies the characteristics of the explored subgraph using graph metrics.
    4.  Generates a human-readable summary of the subgraph, highlighting key elements like connected signals, modules, and potential roles of the starting signal based on its name and connections.
    5.  Scores the generated context to assess its potential relevance or quality.
*   **Dependencies:** `kg_traversal.KGTraversal`, `networkx` (as `nx`), `typing` (List, Dict, Set, Tuple, Optional), `context_pruner.ContextResult`, `config.FLAGS`, `numpy` (as `np`, though not explicitly used in the provided snippet, might be used by dependencies or other parts of the class), `saver.saver` (for logging via `print = saver.log_info`).
*   **How it's used:** This class would be instantiated with a `KGTraversal` object. Its `get_contexts` method would then be called with a list of signal names. The output contexts can then be used, for example, to provide relevant information to an LLM or for other analytical tasks. It's a component in a larger system that leverages a KG for tasks like design understanding or verification. It relies heavily on the configuration provided by `FLAGS` for its operational parameters.

---

## `src/context_generator_path.py`

*   **Purpose:** This script defines the `PathBasedContextGenerator` class. Its goal is to extract meaningful paths from a Knowledge Graph (KG) starting from specified nodes (signals). It then describes these paths to provide context, potentially for an LLM or other analysis. Different styles of path description can be generated.
*   **Key Classes and Methods:**
    *   `PathBasedContextGenerator`:
        *   `__init__(self, kg_traversal: KGTraversal)`:
            *   Initializes with a `KGTraversal` object (providing graph access `self.G` and `signal_to_node_map`).
            *   Calculates and stores global graph metrics (`_calculate_global_metrics`) once.
        *   `get_contexts(self, start_nodes: List[str]) -> List[ContextResult]`:
            *   Main public method. Maps input `start_nodes` (signal names) to graph node IDs.
            *   For each valid node:
                *   Calls `_find_significant_paths` to find paths originating from it.
                *   For each found path:
                    *   Calculates path importance using `_calculate_path_importance`.
                    *   Generates a path description using `_describe_enhanced_path` (which can vary based on `FLAGS`).
                    *   Creates a `ContextResult` object containing the description, score, and metadata.
            *   Returns a list of `ContextResult` objects.
        *   `_calculate_global_metrics(self) -> Dict`:
            *   Calculates global metrics for the entire graph (e.g., average degree, density, average clustering, centrality, betweenness). Used as a baseline for assessing path significance.
        *   `_find_significant_paths(self, start_node: str) -> List[Tuple[List[str], Dict]]`:
            *   Identifies potential endpoint nodes for paths starting at `start_node` using `_identify_potential_endpoints`.
            *   For each potential endpoint, finds all simple paths (up to `max_depth` from `FLAGS`) between `start_node` and the endpoint using `nx.all_simple_paths`.
            *   For each path found, calculates its metrics (`_calculate_path_metrics`) and checks if it's significant (`_is_significant_path`).
            *   Returns a list of (path, path\_metrics) tuples.
        *   `_identify_potential_endpoints(self, start_node: str) -> Set[str]`:
            *   Identifies a set of interesting target nodes for pathfinding.
            *   Considers nodes in a multi-hop neighborhood of `start_node` and some random nodes.
            *   Looks for nodes with higher-than-average degree or clustering, local hubs, specific node types (port, module, assignment), or names similar to the `start_node`.
            *   Limits the number of endpoints to manage performance.
        *   `_calculate_path_metrics(self, path: List[str]) -> Dict`:
            *   Calculates metrics for a given path (list of node IDs), such as length, edge density of the path subgraph, path clustering, and degrees of nodes in the path. It also tries to include centrality and betweenness scores for nodes in the path from global metrics.
        *   `_is_significant_path(self, metrics: Dict) -> bool`:
            *   Determines if a path is significant by comparing its metrics (average node degree, clustering, edge density) against global graph averages.
        *   `_calculate_path_importance(self, path: List[str], metrics: Dict) -> float`:
            *   Calculates an importance score for a path based on its length (favoring moderate lengths), density, clustering (relative to global), and average centrality of its nodes.
        *   `_describe_enhanced_path(self, path: List[str], metrics: Dict) -> str`:
            *   Acts as a dispatcher based on `FLAGS.dynamic_prompt_settings['path_based']['representation_style']`.
            *   Calls one of `_generate_concise_path_description`, `_generate_detailed_path_description`, `_generate_verification_focused_path_description`, or a standard default description.
        *   `_generate_concise_path_description(...)`: Generates a very short summary of the path.
        *   `_generate_detailed_path_description(...)`: Generates a verbose description including all node and edge attributes along the path.
        *   `_generate_verification_focused_path_description(...)`: Generates a description highlighting aspects relevant to hardware verification (timing, control/data dependencies, cross-module interfaces).
        *   The default (standard) path description:
            *   Provides a header with source/target, path length, and clustering.
            *   Details source and target node info (name, type, module).
            *   Lists the "Relationship chain" showing each node in the path, its type, module, and the relationship (from edge attributes) to the next node.
            *   Includes an "Analysis" section that attempts to infer path characteristics (e.g., signal flow, cross-module path, input-to-output path, signal transformation).
*   **Functionality:**
    1.  Identifies structurally interesting paths within the KG, starting from given signals.
    2.  Uses graph algorithms (`nx.all_simple_paths`) and heuristics to find and evaluate these paths.
    3.  Provides different textual representations of these paths, configurable via `FLAGS`, to cater to various downstream uses (e.g., concise for pruning, detailed for deep analysis, verification-focused for specific tasks).
    4.  Scores paths to rank their potential importance or relevance.
*   **Dependencies:** `kg_traversal.KGTraversal`, `context_pruner.ContextResult`, `config.FLAGS`, `numpy` (as `np`), `networkx` (as `nx`), `saver.saver` (for logging), `typing`, `dataclasses`.
*   **How it's used:** This generator would be used to extract linear contextual information from the KG, focusing on sequences of connected entities. The generated path descriptions can help understand relationships and flows between different parts of a design.

---

## `src/context_generator_rag.py`

*   **Purpose:** This script defines classes for Retrieval Augmented Generation (RAG). The main class, `RAGContextGenerator`, aims to retrieve relevant text chunks from both specification documents and RTL (Register Transfer Level) code based on a query. It supports configurable chunking strategies.
*   **Key Classes and Methods:**
    *   `RAGContextGenerator`:
        *   `__init__(self, spec_text: str, rtl_code: Optional[str] = None, chunk_sizes: List[int] = None, overlap_ratios: List[float] = None)`:
            *   Initializes with specification text (`spec_text`) and optionally RTL code (`rtl_code`).
            *   Takes lists of `chunk_sizes` and `overlap_ratios` for document chunking, or uses defaults.
            *   Reads RTL-specific RAG settings from `FLAGS.dynamic_prompt_settings['rag']` (e.g., `enable_rtl_rag`, `baseline_full_spec_RTL`).
            *   If `baseline_full_spec_RTL` is true, it bypasses normal chunking/retrieval and prepares to return the full spec and RTL concatenated.
            *   Otherwise, it creates multiple `DocRetriever` instances for `spec_text` with different chunk/overlap configurations.
            *   If `enable_rtl_rag` is true and `rtl_code` is provided (and not in baseline mode), it also creates `DocRetriever` instances for `rtl_code`.
        *   `get_contexts(self, query: str, k: int = None) -> List[ContextResult]`:
            *   If `baseline_full_spec_RTL` is active, returns a single `ContextResult` containing the full `spec_text` and `rtl_code`.
            *   Otherwise, iterates through all initialized `spec_retrievers` and `rtl_retrievers`.
            *   For each retriever, calls its `retrieve` method with the `query` and `k` (number of chunks to get).
            *   Scores each retrieved chunk based on its rank and chunk size (favoring chunks around 100 words for spec, 50 for RTL). RTL chunks get a slight score bonus.
            *   Creates `ContextResult` objects for each chunk, including metadata like chunk size, overlap, rank, and content type ('spec' or 'rtl').
            *   Sorts all collected contexts by score and returns the top `max_contexts` (default 10).
    *   `DocRetriever`:
        *   `__init__(self, text, chunk_size=100, overlap=20, source_type='spec')`:
            *   Stores `chunk_size`, `overlap`, and `source_type` ('spec' or 'rtl').
            *   Calls `_create_chunks` to split the input `text` into overlapping chunks.
            *   Initializes a `TfidfVectorizer` and computes TF-IDF vectors for all chunks.
        *   `_create_chunks(self, text, chunk_size, overlap)`:
            *   If `source_type` is 'rtl', calls `_create_code_aware_chunks`.
            *   Otherwise (for 'spec'), splits text by words and creates chunks of `chunk_size` words with `overlap` words.
        *   `_create_code_aware_chunks(self, code_text, chunk_size, overlap)`:
            *   Attempts to create chunks from RTL code that respect code structure (e.g., module definitions, functions, always blocks).
            *   Splits by lines and tries to keep important blocks (`module`, `function`, `task`, `always`, `initial`, `endmodule`, etc.) intact within chunks, while adhering to `chunk_size` (in words) as much as possible.
        *   `retrieve(self, query, k=3)`:
            *   Transforms the `query` into a TF-IDF vector.
            *   Computes cosine similarity between the query vector and all chunk vectors.
            *   Returns the top `k` chunks with the highest similarity.
    *   `KGNodeRetriever`: (This class seems somewhat misplaced in a "RAG" context generator primarily focused on text documents, but it's present in the file.)
        *   `__init__(self, kg)`:
            *   Takes a KG structure (dictionary with a 'nodes' key).
            *   Initializes a `SentenceTransformer` model (`paraphrase-MiniLM-L6-v2`).
            *   Calls `_create_node_embeddings` to generate embeddings for all nodes in the KG.
        *   `_create_node_embeddings(self)`:
            *   Creates a textual representation for each node (ID + attributes).
            *   Encodes these texts into embeddings using the sentence transformer model.
        *   `retrieve(self, query, k=3)`:
            *   Encodes the `query` into an embedding.
            *   Computes cosine similarity between the query embedding and all node embeddings.
            *   Returns the IDs of the top `k` most similar KG nodes.
*   **Functionality:**
    1.  Provides a mechanism to retrieve relevant snippets of text from larger documents (specifications, RTL code) based on a textual query.
    2.  Employs TF-IDF and cosine similarity for retrieval in `DocRetriever`.
    3.  Allows for flexible chunking strategies by creating multiple retrievers with different parameters.
    4.  Offers a "code-aware" chunking method for RTL to try and maintain semantic block integrity.
    5.  Includes a baseline mode to just return all text, and an option to enable/disable RTL RAG.
    6.  The `KGNodeRetriever` class uses sentence embeddings to find KG nodes similar to a query, which is a different type of retrieval (node-based vs. text-chunk-based).
*   **Dependencies:** `typing`, `context_pruner.ContextResult`, `sklearn.feature_extraction.text.TfidfVectorizer`, `sklearn.metrics.pairwise.cosine_similarity`, `sentence_transformers.SentenceTransformer`, `config.FLAGS`, `saver.saver`.
*   **How it's used:** `RAGContextGenerator` is instantiated with the full text of a specification and optionally RTL code. Its `get_contexts` method is then called with a query to get relevant text chunks. These chunks can be used to augment prompts for LLMs, providing specific textual context. The `KGNodeRetriever` could be used separately to find relevant starting points or entities within a KG based on a natural language query.

---

## `src/context_generator_rw.py`

*   **Purpose:** This script defines `GuidedRandomWalkContextGenerator`, a class that generates contextual information from a Knowledge Graph (KG) by performing guided random walks. These walks start from specified interface signals and are biased to explore paths towards other interface signals, aiming to discover relevant subgraphs or sequences of nodes.
*   **Key Classes and Methods:**
    *   `GuidedRandomWalkContextGenerator`:
        *   `__init__(self, kg_traversal: 'KGTraversal')`:
            *   Initializes with a `KGTraversal` object (for graph access `self.G` and `signal_to_node_map`).
            *   Loads parameters for the random walk from `FLAGS.dynamic_prompt_settings['guided_random_walk']` (e.g., `num_walks`, `walk_budget`, `teleport_probability`, weights for scoring components `alpha`, `beta`, `gamma`, `max_targets_per_walk`).
            *   Initializes placeholders for pre-computed data: `_signal_distance_map`, `_gateway_nodes`, `_node_importance`.
        *   `get_contexts(self, start_nodes: List[str]) -> List[ContextResult]`:
            *   Main public method. Maps input `start_nodes` (signal names) to graph node IDs.
            *   Pre-computes signal distances (`_precompute_signal_distances`), identifies gateway nodes (`_identify_gateway_nodes`), and computes node importance (`_compute_node_importance`) if not already done. This is done once for all `start_nodes`.
            *   For each valid `focus_node` (mapped from `start_nodes`):
                *   Identifies other interface signals as potential targets.
                *   Performs `self.num_walks` guided random walks using `_guided_random_walk`.
                *   Filters and ranks the resulting paths using `_filter_and_rank_paths`.
                *   For the top-ranked paths (up to a limit from `FLAGS`):
                    *   Calculates path metrics (`_calculate_path_metrics`).
                    *   Generates a textual description of the path using `_describe_path`.
                    *   Calculates a score based on discovered signals and path quality.
                    *   Creates and appends a `ContextResult` object.
            *   Returns a list of `ContextResult` objects.
        *   `_precompute_signal_distances(self, focus_nodes: List[str])`:
            *   Calculates and stores shortest path distances and the actual paths (including the next hop) between all pairs of interface signals (or at least those related to `focus_nodes`). Uses `nx.single_source_shortest_path_length` and `nx.single_source_shortest_path`. Stores results in `self._signal_distance_map`.
        *   `_identify_gateway_nodes(self)`:
            *   Identifies "gateway" nodes that frequently appear on shortest paths between different interface signals.
            *   Counts node occurrences in the precomputed shortest paths.
            *   Stores top gateway nodes for each signal pair in `self._gateway_nodes`.
        *   `_compute_node_importance(self)`:
            *   Calculates an importance score for each node in the graph based on its type (e.g., 'port', 'signal', 'register' are more important) and degree. Stores in `self._node_importance`.
        *   `_guided_random_walk(self, start_node: str, target_signals: List[str], budget: int) -> Tuple[List[str], Set[str]]`:
            *   Performs a single random walk.
            *   Selects a subset of `target_signals` for the current walk.
            *   Iteratively moves from `current_node` to a `next_node` chosen from neighbors.
            *   With `self.teleport_prob`, may jump to a `_select_gateway` node.
            *   Otherwise, chooses `next_node` based on `_compute_transition_probabilities`, which considers local node importance, direction towards targets, and discovery of new nodes.
            *   Keeps track of `discovered_signals` (targets reached).
            *   Returns the path taken and the set of discovered signals.
        *   `_compute_transition_probabilities(...)`: Calculates probabilities for moving to each candidate neighbor based on a weighted sum of local importance (`alpha`), direction towards targets (`beta`), and discovery of unvisited nodes (`gamma`).
        *   `_select_gateway(...)`: Selects a potential gateway node to teleport to, chosen from pre-identified gateways on paths towards current targets.
        *   `_filter_and_rank_paths(...)`: Filters out duplicate paths and paths that didn't discover any target signals. Ranks remaining paths based on the number of signals discovered, path length (shorter is better), and path quality metrics.
        *   `_calculate_path_metrics(self, path: List[str]) -> Dict`: Calculates metrics for a given path, including length, node type distribution, diversity, average node importance, and edge quality (based on relationship types on edges). Combines these into an overall `quality_score`.
        *   `_describe_path(...)`: Generates a human-readable description of a path, including focus node details, the sequence of nodes and their relationships (using `_get_relationship_description`), discovered signals, and a summary of path analysis and metrics.
        *   `_get_relationship_description(self, relationship: str) -> str`: Maps technical relationship identifiers from edge attributes to more human-readable phrases (e.g., "assigns_to" -> "assigns to").
        *   `_get_signal_for_node(self, node: str) -> Optional[str]`: Helper to find the interface signal name corresponding to a graph node ID using `self.signal_to_node_map`.
*   **Functionality:**
    1.  Generates context by simulating intelligent exploration (random walks) within the KG.
    2.  Guides these walks towards discovering connections between a starting signal and other interface signals.
    3.  Uses precomputed graph properties (distances, gateways, node importance) to inform walk decisions and scoring.
    4.  The walk incorporates elements of local importance, goal direction, and novelty/discovery.
    5.  Produces textual summaries of the most fruitful walks, highlighting the path taken and signals found.
*   **Dependencies:** `networkx` (as `nx`), `numpy` (as `np`), `random`, `typing`, `collections.defaultdict`, `collections.Counter`, `context_pruner.ContextResult`, `config.FLAGS`, `saver.saver`, `heapq`.
*   **How it's used:** This generator provides a dynamic way to explore the KG. Instead of fixed patterns (like BFS or specific paths), it uses a probabilistic approach to find potentially interesting connections or subgraphs that link multiple interface signals. The contexts generated can reveal complex interactions.

---

## `src/context_pruner.py`

*   **Purpose:** This script defines `LLMContextPruner`, a class that uses a Large Language Model (LLM) to evaluate and select the most relevant contexts from a list of candidates. The goal is to refine a set of generated contexts (from various sources like RAG, pathfinding, etc.) to a smaller, more focused set for tasks like hardware verification prompt generation. It's described as a "simplified tolerant" version, suggesting it tries to be inclusive.
*   **Key Classes and Methods:**
    *   `ContextResult` (dataclass):
        *   A simple data structure to hold a piece of context.
        *   Attributes: `text` (str), `score` (float, likely from the generating process), `source_type` (str, e.g., 'rag', 'path'), `metadata` (Dict). This class is also defined here, likely for convenience or to ensure consistency if it's used by multiple modules that might not all import a central definition.
    *   `LLMContextPruner`:
        *   `__init__(self, llm_agent, max_contexts_per_type=3, max_total_contexts=10)`:
            *   Initializes with an `llm_agent` (for making LLM calls).
            *   Sets `max_contexts_per_type` (how many contexts to keep from each source type like 'rag', 'path') and `max_total_contexts` (overall limit).
            *   Sets `min_contexts_per_type` (minimum to select per type if available, default 2).
        *   `prune(self, contexts: List[ContextResult], query: str, signal_name: str = None) -> List[ContextResult]`:
            *   The main public method. Takes a list of `ContextResult` objects, the original `query` (e.g., verification intent), and an optional `signal_name` to focus on.
            *   Groups input `contexts` by their `source_type`.
            *   For each `context_type`:
                *   If there are too many contexts of that type, pre-filters them based on their original `score` down to `max_eval_contexts` (e.g., 20).
                *   Creates a prompt for the LLM using `_create_tolerant_prompt`. This prompt asks the LLM to select a range of contexts (between `min_contexts_per_type` and `max_contexts_per_type`) of the current `context_type` that are relevant to the `query` and `signal_name`.
                *   Calls the LLM via `_call_llm`.
                *   Parses the LLM's response (expected to be a list of indices) using `_parse_llm_response`.
                *   If the LLM fails to select any or an error occurs, it falls back to selecting the top-scoring contexts based on their original scores (up to `min_contexts_per_type`).
                *   Adds the LLM-selected (or fallback-selected) contexts to `selected_contexts`.
            *   If the total number of `selected_contexts` across all types exceeds `max_total_contexts`, it calls `_select_balanced_subset` to further reduce the count while trying to maintain diversity from different source types.
            *   Returns the final pruned list of `ContextResult` objects.
        *   `_select_balanced_subset(self, contexts: List[ContextResult]) -> List[ContextResult]`:
            *   If too many contexts are selected overall, this method tries to pick a subset up to `max_total_contexts`.
            *   It ensures a minimum number from each type (pro-rata) and then fills remaining slots with the highest-scoring contexts from the leftovers.
        *   `_create_tolerant_prompt(...)`:
            *   Constructs the detailed prompt for the LLM.
            *   Instructs the LLM to act as an expert hardware verification engineer.
            *   Provides the `query` and `signal_name`.
            *   Asks the LLM to select between `min_selection` and `max_selection` contexts of a specific `context_type`.
            *   Emphasizes tolerance: "Select at least `min_selection` contexts even if they seem only indirectly relevant," "When in doubt, include rather than exclude."
            *   Lists the contexts (with truncation for very long ones) and some of their metadata.
            *   Specifies the output format: "Selected contexts: \[list of indices]".
            *   Gives examples of information useful for hardware verification (signal connections, timing, protocols, etc.).
        *   `_call_llm(self, prompt: str, tag: str) -> str`: A wrapper to call `utils_LLM.llm_inference`.
        *   `_parse_llm_response(self, response: str, max_index: int) -> List[int]`:
            *   Uses a regular expression (`Selected contexts:\s*\[(.*?)\]`) to extract the list of indices from the LLM's response string.
            *   Validates that indices are within the valid range.
*   **Functionality:**
    1.  Refines a potentially large and diverse set of context candidates into a smaller, more relevant set using LLM judgment.
    2.  Operates per context source type initially, then globally, to ensure a manageable number of high-quality contexts.
    3.  The "tolerant" approach in prompting aims to retain contexts that might be indirectly useful, which can be important for complex verification tasks.
    4.  Includes fallback mechanisms (using original scores) if LLM processing fails or returns no selections.
    5.  Balances the final selection to try and include diverse types of context.
*   **Dependencies:** `utils_LLM.llm_inference`, `typing.List`, `typing.Dict`, `dataclasses.dataclass`, `time`, `re`, `saver.saver`.
*   **How it's used:** After various context generators (RAG, path-based, BFS, random walk, etc.) have produced their candidate contexts, the `LLMContextPruner` is used to intelligently filter and select the best ones to pass on to a subsequent stage, likely another LLM call for generating a verification plan or SVA. It acts as an intelligent filter leveraging the LLM's understanding.

---

## `src/design_context_summarizer.py`

*   **Purpose:** This script defines the `DesignContextSummarizer` class, which is responsible for generating various textual summaries of a hardware design using an LLM. These summaries cover the design specification, RTL architecture, signal details, and design patterns. The goal is to create enhanced context, likely for augmenting prompts for SVA (SystemVerilog Assertions) generation or other verification tasks.
*   **Key Classes and Methods:**
    *   `DesignContextSummarizer`:
        *   `__init__(self, llm_agent: str = "gpt-4")`:
            *   Initializes with an `llm_agent` string (e.g., "gpt-4").
            *   Sets up caches: `summary_cache` (for signal-specific summaries), `global_summary` (for overall design summary), and `all_signals_summary`.
        *   `generate_global_summary(self, spec_text: str, rtl_text: str, valid_signals: List[str]) -> Dict[str, Any]`:
            *   Generates a comprehensive, one-time summary of the entire design if not already cached in `self.global_summary`.
            *   Calls helper methods to generate:
                *   `_generate_design_specification_summary(spec_text)`
                *   `_generate_rtl_architecture_summary(rtl_text)`
                *   `_generate_comprehensive_signals_summary(spec_text, rtl_text, valid_signals)` (this result is also stored in `self.all_signals_summary`)
                *   `_generate_design_patterns_summary(spec_text, rtl_text)`
            *   Stores these summaries in `self.global_summary` dictionary along with a generation timestamp.
            *   Returns the `self.global_summary` dictionary.
        *   `_generate_design_specification_summary(self, spec_text: str) -> str`:
            *   Creates a prompt asking the LLM to summarize the provided `spec_text` (3-5 sentences focusing on main functionality, key components, architecture).
            *   Calls `_call_llm` to get the summary.
        *   `_generate_rtl_architecture_summary(self, rtl_text: str) -> str`:
            *   Creates a prompt asking the LLM to summarize the `rtl_text` (3-5 sentences focusing on module hierarchy, interfaces, key architectural features).
            *   Calls `_call_llm`.
        *   `_generate_comprehensive_signals_summary(self, spec_text: str, rtl_text: str, signals: List[str]) -> str`:
            *   Creates a prompt asking the LLM to analyze `spec_text` and `rtl_text` to provide a summary for each signal in the `signals` list.
            *   Requests details for each signal: name, type, bit width, functionality/purpose, key interactions.
            *   Asks for the output to be a list with each signal having its own paragraph.
            *   Calls `_call_llm`.
        *   `_generate_design_patterns_summary(self, spec_text: str, rtl_text: str) -> str`:
            *   Creates a prompt asking the LLM to identify and summarize key design patterns, protocols, or verification-critical structures (e.g., handshaking, FSMs, pipelines, arbiters, CDCs) from `spec_text` and `rtl_text`.
            *   Requests a concise summary (5-10 sentences) of patterns and their verification implications.
            *   Calls `_call_llm`.
        *   `get_signal_specific_summary(self, signal_name: str, spec_text: str, rtl_text: str) -> Dict[str, str]`:
            *   Generates a detailed summary for a specific `signal_name` if not already in `self.summary_cache`.
            *   Creates a prompt asking the LLM for a detailed description of `signal_name` based on `spec_text` and `rtl_text`.
            *   Requests details on: precise function, type/width, timing, key relationships, system behavior impact, special conditions/corner cases.
            *   Asks for 3-5 sentences with verification-focused details.
            *   Calls `_call_llm`.
            *   Caches and returns the result (a dictionary with "description" and "generation_time").
        *   `add_enhanced_context(self, dynamic_context: str, target_signal_name: str) -> str`:
            *   Combines various cached summaries to create an "enhanced context" string.
            *   It prepends the `global_summary` (design overview, RTL architecture), the specific summary for `target_signal_name` (from `self.summary_cache`), the `all_signals_summary`, and the key design patterns summary to the provided `dynamic_context`.
            *   If `global_summary` isn't generated, it returns the original `dynamic_context` with a warning.
        *   `_call_llm(self, prompt: str, tag: str) -> str`: A wrapper to call `utils_LLM.llm_inference`.
*   **Functionality:**
    1.  Leverages an LLM to extract and summarize key information from design specifications and RTL code.
    2.  Generates both global design summaries and detailed summaries for specific signals.
    3.  Identifies and summarizes common hardware design patterns.
    4.  Caches generated summaries to avoid redundant LLM calls.
    5.  Provides a method to consolidate these summaries into an "enhanced context" block, presumably to provide rich, structured information to a downstream LLM task (like SVA generation).
*   **Dependencies:** `typing.Dict`, `typing.List`, `typing.Optional`, `typing.Any`, `utils_LLM.llm_inference`, `time`, `re`, `saver.saver`.
*   **How it's used:** An instance of `DesignContextSummarizer` would be created. `generate_global_summary` would be called once with the full spec and RTL. Then, `get_signal_specific_summary` might be called for individual signals of interest. Finally, `add_enhanced_context` could be used to prepend these structured summaries to other dynamically generated context before feeding it to another LLM.

---

## `src/doc_KG_processor.py`

*   **Purpose:** This script acts as a processor and factory for various context generation modules, particularly those interacting with a Knowledge Graph (KG) and textual specifications. It includes functions to initialize different context generators based on configuration flags and to map signal names to KG nodes. It also defines `MotifContextGenerator` and `CommunityContextGenerator` classes.
*   **Key Functions and Classes:**
    *   `create_context_generators(spec_text: str, kg: Optional[Dict], valid_signals: List[str], rtl_knowledge) -> Dict[str, object]`:
        *   A factory function that initializes and returns a dictionary of context generator objects based on settings in `FLAGS.dynamic_prompt_settings`.
        *   Takes `spec_text`, the `kg` (can be a dict or `nx.Graph`), a list of `valid_signals`, and `rtl_knowledge` (presumably containing RTL content).
        *   If `kg` and `valid_signals` are provided, it calls `build_signal_to_node_mapping` to map signal names to KG node IDs. Handles different KG input types (dict vs. `nx.Graph`).
        *   Initializes `RAGContextGenerator` if `FLAGS.dynamic_prompt_settings['rag']['enabled']` is true.
        *   If `kg` is available:
            *   Creates a `KGTraversal` object and attaches the `signal_to_node_map` to it.
            *   Initializes other KG-based generators if their respective `enabled` flags are true in `FLAGS`:
                *   `PathBasedContextGenerator`
                *   `MotifContextGenerator`
                *   `CommunityContextGenerator`
                *   `LocalExpansionContextGenerator` (from `context_generator_BFS.py`)
                *   `GuidedRandomWalkContextGenerator` (from `context_generator_rw.py`)
        *   Returns a dictionary where keys are generator types (e.g., 'rag', 'path') and values are the generator instances.
    *   `build_signal_to_node_mapping(kg: Dict, valid_signals: List[str]) -> Dict[str, List[str]]`:
        *   Maps textual `valid_signals` names to lists of node IDs in the `kg`.
        *   Assumes `kg` is a dictionary (converts `nx.Graph` if needed before calling this, though the type hint says `Dict`).
        *   Iterates through nodes in the KG. If a node's type is 'port', 'signal', or 'assignment', it checks if its 'name' attribute matches any of the `valid_signals`.
        *   For 'assignment' types, it tries to extract a base signal name (e.g., from "baud\_clk\_assignment").
        *   Also checks 'expression' attributes for signal name occurrences using regex word boundary matching.
        *   Prints extensive debug information about the mapping process.
        *   Returns a dictionary mapping signal names to lists of corresponding node IDs.
    *   `MotifContextGenerator`:
        *   `__init__(self, kg_traversal: KGTraversal)`: Initializes with `KGTraversal` and gets `signal_to_node_map`.
        *   `get_contexts(self, start_nodes: List[str]) -> List[ContextResult]`:
            *   Maps `start_nodes` (signal names) to KG node IDs.
            *   Calls internal methods to find different types of motifs/patterns anchored around these `valid_nodes`:
                *   `_find_cycles`: Uses `nx.simple_cycles`.
                *   `_find_hubs`: Identifies nodes with degree significantly higher than average.
                *   `_find_dense_subgraphs`: Uses `nx_comm.louvain_communities` and checks density.
            *   For each discovered pattern, calculates importance (`_analyze_pattern_importance`), generates a description (`_describe_enhanced_pattern`), and creates a `ContextResult`.
        *   `_analyze_pattern_importance(self, nodes: List[str]) -> float`: Scores pattern importance based on structural metrics, connectivity, and centrality.
        *   `_describe_enhanced_pattern(self, pattern_type: str, pattern: Dict) -> str`: Generates detailed textual descriptions for 'cycle', 'hub', and 'dense' patterns, focusing on the `start_node` and interpreting the pattern's meaning in a hardware context.
    *   `CommunityContextGenerator`:
        *   `__init__(self, kg_traversal: KGTraversal)`: Initializes with `KGTraversal` and gets `signal_to_node_map`.
        *   `get_contexts(self, start_nodes: List[str]) -> List[ContextResult]`:
            *   Maps `start_nodes` to KG node IDs.
            *   Detects communities using `_detect_communities_safely` (which tries Louvain or falls back to greedy modularity/connected components if issues with weights are detected).
            *   For each `node_id` in `valid_nodes`, if it belongs to a community:
                *   Calculates community metrics (`_calculate_community_metrics`).
                *   Generates a description using `_describe_enhanced_community`.
                *   Scores the context and creates a `ContextResult`.
            *   Includes fallback logic to create a small local community (neighborhood) if a node isn't in any detected larger community.
        *   `_detect_communities_safely(self) -> List[Set[str]]`: Tries `nx_comm.louvain_communities` with `weight=None`. If it suspects string weights or other issues, it creates an unweighted graph copy and tries `nx_comm.greedy_modularity_communities` or connected components.
        *   `_calculate_community_metrics(self, community: Set[str]) -> Dict`: Calculates density, avg. degree, clustering coeff. for the community subgraph.
        *   `_calculate_node_centrality(self, node: str, community: Set[str]) -> float`: Calculates degree centrality of a node within its community subgraph.
        *   `_describe_enhanced_community(self, community: Set[str], start_node: str) -> str`: Generates a detailed description of the community, focusing on the `start_node`, listing related modules, signals, key relationships of the `start_node`, and relevant assignments.
*   **Functionality:**
    1.  **Context Generator Orchestration:** The `create_context_generators` function acts as a central point for initializing various context generation components based on global configuration.
    2.  **Signal Mapping:** `build_signal_to_node_mapping` is crucial for connecting textual signal names (likely from RTL analysis or user input) to their representations within the KG.
    3.  **Motif Detection:** `MotifContextGenerator` identifies common structural patterns (cycles, hubs, dense areas) in the KG that involve specified signals. These patterns can indicate important functional blocks or relationships.
    4.  **Community Detection:** `CommunityContextGenerator` finds groups of densely interconnected nodes (communities) in the KG that include specified signals. These communities can represent modules or closely related functional units.
    5.  Both motif and community generators provide descriptive textual contexts for their findings.
*   **Dependencies:** `kg_traversal.KGTraversal`, `context_generator_rag.RAGContextGenerator`, `context_generator_path.PathBasedContextGenerator`, `context_generator_BFS.LocalExpansionContextGenerator`, `context_generator_rw.GuidedRandomWalkContextGenerator`, `context_pruner.ContextResult`, `config.FLAGS`, `numpy`, `sklearn` (TF-IDF, cosine similarity), `networkx` (including `nx_comm`), `sentence_transformers`, `saver.saver`, `typing`, `dataclasses`, `re`.
*   **How it's used:** `create_context_generators` is called early in a processing pipeline to set up all the necessary tools for context generation. `build_signal_to_node_mapping` is a key utility used by this factory. The `MotifContextGenerator` and `CommunityContextGenerator` (along with others initialized by the factory) are then invoked with lists of signals to produce diverse types of contextual information from the KG.

---

## `src/dynamic_prompt_builder.py`

*   **Purpose:** This script defines the `DynamicPromptBuilder` class, which is responsible for constructing prompts for an LLM by dynamically gathering and integrating various types of context. It uses multiple context generators, prunes the collected contexts, and then distributes them across one or more prompts, respecting token limits.
*   **Key Classes and Methods:**
    *   `DynamicPromptBuilder`:
        *   `__init__(self, context_generators: Dict[str, object], pruning_config: Dict, llm_agent, context_summarizer=None)`:
            *   Initializes with a dictionary of `context_generators` (e.g., RAG, path-based, motif), `pruning_config` settings (from `FLAGS`), an `llm_agent` for pruning, and an optional `context_summarizer`.
            *   Initializes an `LLMContextPruner` instance if an `llm_agent` is provided.
        *   `build_prompt(self, query: str, base_prompt: str, signal_name: Optional[str] = None, enable_context_enhancement: bool = False) -> List[str]`:
            *   The main public method.
            *   Determines `max_prompts_per_signal` and `max_tokens_per_prompt` from `FLAGS`.
            *   If `enable_context_enhancement` is true and a `context_summarizer` is available, it estimates the token overhead of adding summarized context.
            *   Calculates `max_context_tokens` available for dynamically fetched context, reserving space for the LLM's response and enhancement overhead.
            *   Collects `start_nodes` for KG-based generators, using `kg_node_retriever` if available, or `signal_name`.
            *   Iterates through `self.context_generators`:
                *   Calls `get_contexts` on each enabled generator (RAG, path, motif, community, local_expansion, guided_random_walk).
                *   Aggregates all generated contexts into `all_contexts`.
            *   **Pruning:** If `self.pruning_config['enabled']` is true:
                *   Uses `self.llm_pruner.prune()` (which internally uses an LLM) to select the most relevant contexts from `all_contexts` based on the `query` and `signal_name`.
                *   (The `_prune_contexts_similarity` method is a legacy fallback using sentence similarity, but the main flow uses the LLM pruner).
            *   **Context Distribution:**
                *   Groups the `selected_contexts` by their `source_type`.
                *   Calculates how many contexts of each type should go into each of the `max_prompts_per_signal` prompts to ensure even distribution.
            *   **Prompt Assembly:**
                *   Constructs `max_prompts_per_signal` prompts.
                *   Each prompt starts with `base_prompt_text` (which includes the `base_prompt` and headers for context).
                *   Iterates through each prompt slot and context type, adding formatted contexts (`format_context`) while respecting `max_context_tokens`.
                *   `format_context` includes the context text and relevant metadata.
                *   If a context doesn't fit in the current prompt, it tries to place it in another prompt. If it fits nowhere, it's skipped.
                *   Adds headers for each context type (e.g., "RAG Context:", "PATH Context:") if contexts of that type are present in the prompt.
            *   Returns a list of fully constructed prompt strings (`final_prompts`).
            *   If `enable_context_enhancement` is true, it uses `self.context_summarizer.add_enhanced_context()` to prepend summaries to each final prompt.
        *   `_prune_contexts_similarity(...)`: A legacy/fallback method for pruning contexts based on sentence embedding similarity to the query. Not the primary pruning method if an LLM agent is available for the `LLMContextPruner`.
*   **Functionality:**
    1.  Orchestrates the generation of context from multiple heterogeneous sources (RAG, various KG traversal methods).
    2.  Intelligently prunes the collected contexts using an LLM (primarily) to select the most relevant information for a given query and signal.
    3.  Distributes the selected contexts across a configured number of prompts, aiming for an even spread of context types and adherence to token limits.
    4.  Formats the contexts with their source type and metadata for clarity within the prompt.
    5.  Optionally enhances prompts by prepending pre-generated design summaries via a `ContextSummarizer`.
*   **Dependencies:** `context_pruner.LLMContextPruner`, `context_pruner.ContextResult`, `utils_LLM.count_prompt_tokens`, `config.FLAGS`, `sentence_transformers.SentenceTransformer`, `sklearn.metrics.pairwise.cosine_similarity`, `numpy`, `networkx`, `saver.saver`, `typing`, `dataclasses`.
*   **How it's used:** This class is a core component for preparing complex prompts. It's instantiated with various context generators and an LLM agent. The `build_prompt` method is then called with a base query and prompt structure. The output is a list of ready-to-use prompts, each enriched with diverse and relevant contextual information, tailored for an LLM to perform a task like verification plan generation.

---

## `src/gen_KG_graphRAG.py`

*   **Purpose:** This script provides functionality to build a Knowledge Graph (KG) using the GraphRAG tool. It automates the process of setting up the GraphRAG environment, processing input design documents (PDFs or JSONL), and running the GraphRAG indexing pipeline.
*   **Key Functions:**
    *   `build_KG()`:
        *   The main function that orchestrates the KG building process.
        *   Retrieves `input_file_path` and other configurations from `FLAGS`.
        *   `get_base_dir()`: Determines the base directory from the input file path(s).
        *   `create_directory_structure()`: Creates a `graph_rag_<design_name>` directory with an `input` subdirectory.
        *   `clean_input_folder()`: Removes any existing files in the `input` directory.
        *   `process_files()`: Processes the input file(s). This calls `process_single_file`.
            *   `process_single_file()`:
                *   If JSONL: `get_jsonl_stats()` and `parse_jsonl_to_text()`.
                *   If PDF: `get_pdf_stats()` and `parse_pdf_to_text()`.
                *   `parse_pdf_to_text()`: Extracts text from PDF pages using `PyPDF2` and saves it to a .txt file in the `graph_rag_<design_name>/input` directory.
                *   `parse_jsonl_to_text()`: Extracts text content from JSONL, potentially cleaning tables and LaTeX-like syntax, and saves to a .txt file.
                *   `get_pdf_stats()`/`get_jsonl_stats()`: Collect statistics like page count, file size, word count, and token count (using `tiktoken` for PDFs).
        *   `initialize_graphrag()`: Runs `python -m graphrag.index --init --root <graph_rag_dir>` to set up the GraphRAG project structure. Handles cases where the project is already initialized.
        *   Copies a custom `.env` file (from `FLAGS.env_source_path`) and `settings.yaml` (from `FLAGS.settings_source_path`) into the `graph_rag_dir`.
        *   `copy_entity_extraction_prompt()`: Copies a custom entity extraction prompt (from `FLAGS.entity_extraction_prompt_source_path`) to `graph_rag_dir/prompts/entity_extraction.txt`.
        *   `run_graphrag_index()`: Executes the main GraphRAG indexing command: `python -m graphrag.index --root <graph_rag_dir>`. It streams the output of this command.
        *   `detect_graphrag_log_folder()`: (Although defined, this function seems to be called *after* `run_graphrag_index` in `build_KG`, but its internal `run_graphrag_index` call suggests it might be intended to find the folder *created by* the indexing run. The current structure in `build_KG` calls `run_graphrag_index` first, then `detect_graphrag_log_folder` which itself calls `run_graphrag_index` again – this might be redundant or a slight logical oversight.) The goal is to identify the timestamped output directory created by GraphRAG.
        *   Uses `utils.OurTimer` for timing various steps.
    *   Supporting utility functions:
        *   `get_base_dir()`: Determines the common parent directory for input files.
        *   `get_pdf_stats()`, `get_jsonl_stats()`: Calculate statistics for input files.
        *   `parse_pdf_to_text()`, `parse_jsonl_to_text()`: Convert input documents to plain text. `parse_jsonl_to_text` includes specific logic for table cleaning.
        *   `clean_input_folder()`: Clears the target input directory.
        *   `clean_table()`: Helper for `parse_jsonl_to_text` to reformat table content.
        *   `create_directory_structure()`: Sets up the necessary folder structure for GraphRAG.
        *   `initialize_graphrag()`: Initializes a GraphRAG project.
        *   `copy_entity_extraction_prompt()`: Copies a specific prompt file.
        *   `run_graphrag_index()`: Runs the GraphRAG indexing process and streams its output.
        *   `detect_graphrag_log_folder()`: Tries to identify the output folder created by a GraphRAG run.
*   **Functionality:**
    1.  Automates the setup and execution of the GraphRAG indexing pipeline.
    2.  Handles PDF and JSONL input files, converting them to text suitable for GraphRAG.
    3.  Configures the GraphRAG run by providing custom `.env`, `settings.yaml`, and entity extraction prompt files.
    4.  Executes the GraphRAG indexing command and monitors its output.
    5.  Provides basic statistics and timing for the KG generation process.
*   **Dependencies:** `os`, `subprocess`, `shutil`, `json`, `re`, `pathlib.Path`, `PyPDF2`, `logging`, `datetime`, `utils.OurTimer`, `utils.get_ts`, `saver.saver`, `config.FLAGS`, `tiktoken`.
*   **How it's used:** The `build_KG()` function is called when `FLAGS.task` is set to 'build\_KG'. It takes design documents as input and produces a Knowledge Graph using the external GraphRAG tool. The resulting KG (likely GraphML files and other artifacts in the `graph_rag_<design_name>/output` directory) would then be used by other parts of the system (e.g., by `KGTraversal` and the various KG-based context generators).

---

## `src/gen_plan.py`

*   **Purpose:** This script is the main orchestrator for a process that involves generating test plans and SystemVerilog Assertions (SVAs) for hardware designs. It reads design specifications (PDFs), optionally uses a Knowledge Graph (KG) and RTL information, interacts with an LLM to generate natural language (NL) plans and SVAs, and then can run these SVAs through JasperGold for formal verification. It also includes functionality to parse and analyze results from previous runs.
*   **Key Functions:**
    *   `gen_plan()`: The main entry point.
        *   Handles two `FLAGS.subtask` modes:
            *   `'actual_gen'`: The full generation and verification flow.
            *   `'parse_result'`: Loads and analyzes results from a directory.
        *   **In `'actual_gen'` mode:**
            1.  `read_pdf()`: Reads PDF specification file(s).
            2.  If `FLAGS.use_KG` is true:
                *   `load_and_process_kg()`: Loads a KG (GraphML) into a NetworkX graph and a JSON representation.
                *   If `FLAGS.refine_with_rtl`: Calls `refine_kg_from_rtl()` (from `rtl_parsing.py`) to augment the KG with information extracted from RTL files.
            3.  `get_llm()`: Initializes an LLM agent.
            4.  `write_svas_to_file([])` (called with an empty list just to get valid signals) or uses `FLAGS.valid_signals`: Extracts/defines a list of valid signal names from the design's top-level SVA file.
            5.  If `FLAGS.enable_context_enhancement`: Initializes `DesignContextSummarizer`, generates global summaries, and pre-generates summaries for signals to be processed.
            6.  `generate_nl_plans()`: Generates natural language test plans. This function internally chooses between `generate_dynamic_nl_plans` or `generate_static_nl_plans` based on `FLAGS.prompt_builder`.
            7.  If `FLAGS.generate_SVAs` is true:
                *   `generate_svas()`: Generates SVAs from the NL plans. Also chooses dynamic or static generation based on `FLAGS.prompt_builder`.
                *   `write_svas_to_file()`: Writes each SVA to a separate `.sva` file, embedding it within the original module interface.
                *   `generate_tcl_scripts()`: Creates TCL scripts for JasperGold for each SVA file.
                *   `run_jaspergold()`: Executes JasperGold for each TCL script and collects reports.
                *   `analyze_coverage_of_proven_svas()`: (Appears to be defined in `utils_gen_plan.py` but its direct output isn't heavily used in `analyze_results` beyond the fact that it's called).
            8.  `analyze_results()`: Prints and saves a summary of the entire process, including PDF stats, plan/SVA counts, JasperGold outcomes, and coverage metrics.
        *   Uses `utils.OurTimer` to track execution time of different steps.
    *   `read_pdf(file_path: Union[str, List[str]]) -> Tuple[str, dict]`: Reads text from one or more PDF files using `PdfReader` and returns concatenated text and statistics (page count, token count via `count_tokens_in_file`, file size).
    *   `load_and_process_kg(kg_path: str) -> Tuple[nx.Graph, Dict]`: Loads a GraphML file into an `nx.Graph` and also converts it to a JSON-like dictionary using `convert_nx_to_json`. Prints KG stats.
    *   `convert_nx_to_json(G: nx.Graph) -> Dict`: Converts an `nx.Graph` to a simpler dictionary representation (nodes with attributes, edges with attributes).
    *   `generate_nl_plans(...)`: A dispatcher that calls either `generate_dynamic_nl_plans` or `generate_static_nl_plans`.
        *   `generate_dynamic_nl_plans(...)`:
            *   Uses `create_context_generators` (from `doc_KG_processor.py`) to get various context providers.
            *   Uses `DynamicPromptBuilder` to construct prompts with context (potentially enhanced by `DesignContextSummarizer`).
            *   Calls `llm_inference` for each signal and context combination to get NL plans.
            *   Deduplicates plans.
        *   `generate_static_nl_plans(...)`: Calls `llm_inference` with a statically constructed prompt (`construct_static_nl_prompt`) to get NL plans. Parses the output with `parse_nl_plans`.
    *   `generate_svas(...)`: A dispatcher for `generate_dynamic_svas` or `generate_static_svas`.
        *   `generate_dynamic_svas(...)`: Similar to dynamic NL plan generation, uses `DynamicPromptBuilder` for context. Constructs prompts including NL plans and SVA examples (`get_sva_icl_examples`), then calls `llm_inference`. Extracts SVAs using `extract_svas_from_block` (from `sva_extraction.py`). Handles plan distribution across multiple contexts.
        *   `generate_static_svas(...)`: Calls `llm_inference` with a statically constructed prompt (`construct_static_sva_prompt`) and extracts SVAs.
    *   `get_sva_icl_examples()`: Returns a string with few-shot examples of NL plan to SVA conversion.
    *   `construct_static_nl_prompt(...)` & `construct_static_sva_prompt(...)`: Functions to build large, static prompts that include the spec text, (optional) KG JSON, valid signal lists, and examples. These prompts heavily emphasize using only valid signals.
    *   `parse_nl_plans(result: str) -> Dict[str, List[str]]`: Parses LLM output (expected format: "SIGNAL: plan text") into a dictionary.
    *   `write_svas_to_file(svas: List[str]) -> Tuple[List[str], Set[str]]`:
        *   Reads the module interface from an original SVA file (`property_goldmine.sva` in `FLAGS.design_dir`).
        *   Extracts `valid_signals` from this interface using `extract_signal_names`.
        *   For each generated SVA, creates a new `.sva` file containing the original module interface and the SVA formatted as an asserted property.
    *   `generate_tcl_scripts(sva_file_paths: List[str]) -> List[str]`: Creates JasperGold TCL scripts by modifying an original TCL script template to point to each new SVA file.
    *   `modify_tcl_content(original_content: str, new_sva_path: str) -> str`: Helper to substitute the SVA file path in TCL content.
    *   `run_jaspergold(tcl_file_paths: List[str]) -> List[str]`: Runs JasperGold for each TCL script in a separate project directory and saves the output reports. Placeholder for JasperGold command path.
    *   `analyze_results(...)`: Compiles and prints various statistics: PDF input size, NL plan counts, SVA counts, JasperGold proof outcomes (proven, CEX, inconclusive, error), and coverage metrics (extracted by `calculate_coverage_metric`). Uses `generate_detailed_sva_report` to create a CSV and tabulated output of SVA statuses.
    *   `calculate_coverage_metric(jasper_out_str)`: Parses JasperGold output text to extract various coverage percentages (stimuli/COI for statement, branch, functional, toggle, expression). Also determines a basic "functionality" metric based on proof results.
    *   `generate_detailed_sva_report(...)`: Creates a Pandas DataFrame and CSV file summarizing each SVA's ID, proof status, and error message (if any). Uses `extract_proof_status` and `extract_short_error_message`.
    *   `extract_error_message(...)` & `extract_short_error_message(...)`: Helpers to pull error details from JasperGold reports.
    *   `log_llm_interaction(...)`: Logs LLM prompts and responses.
    *   `extract_signal_names(module_interface: str) -> Set[str]`: Extracts signal names from a Verilog module interface string using regex.
*   **Functionality:** This script implements an end-to-end pipeline for:
    1.  Processing design specifications.
    2.  Leveraging KGs and RTL information for context.
    3.  Using LLMs for generating NL test intentions and then SVAs.
    4.  Automating the formal verification of these SVAs using JasperGold.
    5.  Collecting, analyzing, and reporting the results, including verification status and coverage metrics.
    It supports both a "static" prompt building approach (one large prompt) and a "dynamic" approach (multiple smaller prompts with context from various generators, managed by `DynamicPromptBuilder`).
*   **Dependencies:** `sva_extraction.extract_svas_from_block`, `doc_KG_processor.create_context_generators`, `dynamic_prompt_builder.DynamicPromptBuilder`, `load_result` (various functions), `rtl_parsing.refine_kg_from_rtl`, `utils_gen_plan` (various functions), `design_context_summarizer.DesignContextSummarizer`, `os`, `math`, `subprocess`, `config.FLAGS`, `saver.saver`, `utils.OurTimer`, `utils_LLM` (get\_llm, llm\_inference), `networkx`, `typing`, `PyPDF2`, `json`, `random`, `re`, `pandas`, `tabulate`, `pathlib.Path`, `tqdm`.
*   **How it's used:** This is likely the main executable script for the 'gen\_plan' task. It's run with specific configurations set in `config.FLAGS` to process a design and evaluate the quality of LLM-generated SVAs.

---

## `src/kg_traversal.py`

*   **Purpose:** This script defines the `KGTraversal` class, which provides basic functionalities to represent and traverse a Knowledge Graph (KG). The KG is assumed to be provided in a specific JSON-like dictionary format (nodes and edges) and is converted into a `networkx.Graph` object internally.
*   **Key Classes and Methods:**
    *   `KGTraversal`:
        *   `__init__(self, kg)`:
            *   Takes `kg` (expected to be a dictionary with 'nodes' and 'edges' keys) as input.
            *   Calls `_build_graph()` to construct an internal `networkx.Graph`.
            *   (Note: In `doc_KG_processor.py`, an instance variable `self.signal_to_node_map` is attached to `KGTraversal` instances after initialization, making it available to other generators that use `KGTraversal`.)
        *   `_build_graph(self)`:
            *   Initializes `self.graph = nx.Graph()`.
            *   Adds nodes to `self.graph` from `self.kg['nodes']`, including their attributes.
            *   Adds edges to `self.graph` from `self.kg['edges']`, including their attributes.
        *   `traverse(self, start_node, max_depth=2)`:
            *   Performs a graph traversal (specifically a Depth First Search as implemented in `_dfs`) starting from `start_node` up to `max_depth`.
            *   Keeps track of `visited_nodes` and `visited_edges` to avoid cycles and redundant processing in this specific traversal instance.
            *   Returns `result_nodes` (list of visited node IDs in DFS order) and `result_edges` (list of visited edges as (source, target, attributes) tuples).
        *   `_dfs(self, node, max_depth, current_depth, visited_nodes, visited_edges, result_nodes, result_edges)`:
            *   Recursive helper function for `traverse`.
            *   Adds current `node` to `result_nodes`.
            *   For each neighbor, if the edge hasn't been visited, adds it to `result_edges` and recursively calls `_dfs` on the neighbor.
        *   `get_node_info(self, node_id)`:
            *   Returns the attributes of a node with `node_id` if it exists in the graph.
        *   `get_edge_info(self, source, target)`:
            *   Returns the attributes of the edge between `source` and `target` if it exists.
*   **Functionality:**
    1.  Provides a simple wrapper around `networkx` to represent a KG that is initially defined in a dictionary format.
    2.  Offers a basic DFS traversal mechanism.
    3.  Allows querying for node and edge attributes.
*   **Dependencies:** `config.FLAGS` (though not directly used in the snippet, often KG-related files might reference it), `numpy` (as `np`, not directly used but common in graph contexts), `networkx` (as `nx`), `saver.saver` (for logging), `typing`, `dataclasses`.
*   **How it's used:** An instance of `KGTraversal` is created by passing it the KG data (e.g., loaded from a file and processed into the expected dictionary format). This instance is then passed to various context generator classes (like `PathBasedContextGenerator`, `MotifContextGenerator`, etc., often via `doc_KG_processor.create_context_generators`). These generators use the `self.graph` attribute (the `networkx.Graph` object) of the `KGTraversal` instance to perform their specific graph analysis and context extraction tasks. The `signal_to_node_map` attribute, added externally, is also heavily used by these generators.

---

## `src/main.py`

*   **Purpose:** This script serves as the main entry point for the application. It uses a `FLAGS.task` variable (from `config.py`) to determine which primary operation to perform: generating a plan (`gen_plan`), building a Knowledge Graph (`build_KG`), or using/querying a Knowledge Graph (`use_KG`).
*   **Key Functions:**
    *   `main()`:
        *   Reads `FLAGS.task`.
        *   If `FLAGS.task == 'gen_plan'`, it calls `gen_plan()` (from `gen_plan.py`).
        *   If `FLAGS.task == 'build_KG'`, it calls `build_KG()` (from `gen_KG_graphRAG.py`).
        *   If `FLAGS.task == 'use_KG'`, it calls `use_KG()` (from `use_KG.py`).
        *   Raises `NotImplementedError` if the task is unknown.
    *   The `if __name__ == '__main__':` block:
        *   Initializes an `OurTimer` to measure total execution time.
        *   Calls `main()` within a `try...except` block to catch and log any exceptions.
        *   Uses `saver.log_info()` and `saver.save_exception_msg()` to record errors.
        *   Prints the total execution time and the log directory path using `report_save_dir()`.
        *   Calls `saver.close()` to finalize logging.
*   **Functionality:**
    1.  Acts as a simple command-line dispatcher based on a configuration flag.
    2.  Provides top-level error handling and logging for the selected task.
    3.  Measures and reports the total execution time of the chosen task.
*   **Dependencies:** `gen_plan.gen_plan`, `gen_KG_graphRAG.build_KG`, `use_KG.use_KG`, `saver.saver`, `config.FLAGS`, `utils.OurTimer`, `utils.get_root_path`, `utils.report_save_dir`, `traceback`, `sys`.
*   **How it's used:** This script is executed to run one of the main functionalities of the project. The specific behavior is controlled by setting the `task` variable in `src/config.py` before running `python src/main.py`.

---

## `src/rtl_analyzer.py`

*   **Purpose:** This script defines the `RTLAnalyzer` class, designed to parse and analyze Verilog/SystemVerilog RTL (Register Transfer Level) code. It aims to extract structural and behavioral information from the design, such as module definitions, ports, signals, instances, FSMs (Finite State Machines), and data flow. This information can then be used to enhance other processes, like KG construction or SVA generation.
*   **Key Classes and Methods:**
    *   `RTLAnalyzer`:
        *   `__init__(self, design_dir: str, verbose: bool = False)`:
            *   Initializes with the `design_dir` (path to RTL files) and a `verbose` flag.
            *   Sets up various internal dictionaries and sets to store analysis results: `file_info`, `module_info`, `control_flow`, `fsm_info`, `signal_type_info`, `primary_signals`, `data_flow_graph` (an `nx.DiGraph`), `verification_suggestions`.
        *   `analyze_design()`:
            *   The main public method to start the analysis.
            *   `process_files_in_order()`: Gets a list of Verilog files, attempting to sort them by dependency (includes first).
            *   Combines content of all Verilog files into a single string (`combined_content`) and also writes it to `_combined_rtl.v`. This is done to potentially help parsers that handle includes better or to get a global view.
            *   Attempts to process the `_combined_rtl.v` file first using `_process_file()`.
            *   If the combined approach yields no module info, it falls back to processing individual files.
            *   Calls various helper methods to extract specific types of information:
                *   `_extract_port_info_direct()`: Fallback to get port info via regex if AST parsing fails.
                *   `_find_module_instances_direct()`: Fallback for instance detection.
                *   `_build_cross_module_dataflow()`: Connects data flow across module instances.
                *   `_identify_fsms()`: Tries to detect FSMs through AST analysis and pattern matching.
                *   `_analyze_primary_signal_relationships()`: Looks for paths between I/O signals in the data flow graph.
            *   `_print_summary()`: Prints a basic summary of findings.
            *   `_enhanced_signal_analysis()`: Triggers more detailed signal and protocol pattern analysis.
            *   Calls further extraction methods per module: `_extract_control_flow`, `_extract_assignments`, `_extract_signal_attributes`.
            *   `_print_expanded_summary()`: Prints a more detailed summary.
            *   Cleans up the temporary combined RTL file.
        *   `_process_file(self, file_path: str)`:
            *   Core processing logic for a single Verilog file.
            *   Reads file content, handles `sockit` design with a simplified model as a special case due to Pyverilog parsing issues.
            *   `_preprocess_includes()`: A simple preprocessor to inline `include` directives.
            *   Uses `pyverilog.vparser.parser.parse()` to generate an AST.
            *   `_extract_module_info(ast_output)`: Extracts module names, ports, signals, instances, always blocks, and assign statements from the Pyverilog AST.
            *   If AST parsing fails or yields no modules, falls back to `_extract_module_info_from_content()` (regex-based).
            *   `_extract_dataflow_info()`: Uses `pyverilog.dataflow.dataflow_analyzer` to build a data flow graph for signals within a module.
        *   `_extract_module_info_from_content(self, content)`: Regex-based extraction of module names, ports, and basic always blocks.
        *   `_parse_senslist(self, senslist)`: Parses sensitivity list strings.
        *   `_extract_module_info_from_simplified(self, file_path)`: Regex-based extraction for the special `sockit` case.
        *   `_parse_width(self, width_node)`: Parses AST width nodes to strings.
        *   `_add_to_dataflow_graph(self, signal_name: str, term, dfg)`: Adds signal and its driving terms to `self.data_flow_graph`.
        *   `_build_cross_module_dataflow(self)`: Iterates through module instances and their port connections to add edges to `self.data_flow_graph` representing data flow between signals of the parent and instantiated modules.
        *   `_find_module_instances_direct(self)` & `_extract_port_info_direct(self)`: Regex-based fallbacks.
        *   `_identify_fsms(self)`: Detects FSMs by looking for clock-sensitive always blocks in AST, case statements, state register naming conventions, and state parameter definitions through regex on file content. Calls `_direct_fsm_pattern_match` as a last resort.
        *   `_direct_fsm_pattern_match(self)`: Uses a list of regex patterns to find common FSM-related code structures.
        *   `_analyze_primary_signal_relationships(self)`: Checks for paths in the `data_flow_graph` between primary I/O signals.
        *   `_print_summary(self)` & `_print_expanded_summary(self)`: Print analysis results to console.
        *   `_enhanced_signal_analysis(self)`: Calls `_analyze_protocol_patterns` or `_direct_rtl_analysis`.
        *   `_analyze_protocol_patterns(self)`: Identifies common signal types (clock, reset, data, control, handshaking) based on port naming conventions. Calls `_generate_verification_suggestions`.
        *   `_generate_verification_suggestions(self, module_name, patterns)`: Creates generic verification suggestions based on identified protocol patterns (e.g., reset behavior, CDC, data stability, handshaking).
        *   `_direct_rtl_analysis(self)`: A fallback analysis method that uses regex to find modules, ports, always blocks, case statements, and assignments directly from file content when AST-based parsing is insufficient.
        *   `get_analysis_results(self)`: Returns a dictionary containing all collected analysis information.
        *   `_extract_control_flow(self, module_name, file_path)`: Regex-based extraction of if/case/loop statements.
        *   `_extract_assignments(self, module_name, file_path)`: Regex-based extraction of continuous and procedural assignments.
        *   `_extract_signal_attributes(self, module_name, file_path)`: Regex-based extraction of signal types (wire, reg, logic, parameters), widths, and initial values.
    *   `process_files_in_order(design_dir)`: (Outside the class)
        *   Attempts to determine a processing order for Verilog files by building a dependency graph based on `` `include`` directives and module instantiations.
        *   Uses `nx.topological_sort`. Falls back to an unsorted list if cyclic dependencies are detected.
    *   `main()` (in `if __name__ == "__main__":`):
        *   Command-line argument parsing (`argparse`) for `design_dir` and `verbose`.
        *   Calls `process_files_in_order`.
        *   Instantiates `RTLAnalyzer` and calls `analyze_design()`.
*   **Functionality:**
    1.  Parses a hierarchy of Verilog/SystemVerilog files.
    2.  Attempts to build an Abstract Syntax Tree (AST) using Pyverilog and extracts structural information (modules, ports, instances, signals, always blocks, assignments).
    3.  Uses Pyverilog's dataflow analyzer to understand signal dependencies within modules.
    4.  Constructs a cross-module data flow graph.
    5.  Employs regex-based fallbacks and direct content analysis when AST parsing is incomplete or fails, making it resilient to some non-standard Verilog or parsing limitations.
    6.  Identifies potential Finite State Machines (FSMs) using a combination of AST features and pattern matching.
    7.  Analyzes protocol patterns based on signal naming conventions and generates generic verification suggestions.
    8.  Provides detailed summaries of the analyzed design.
    9.  The output (`get_analysis_results()`) is a structured dictionary that can be used by other tools (e.g., to refine a KG or provide context for LLMs).
*   **Dependencies:** `utils_LLM.count_prompt_tokens`, `os`, `sys`, `re`, `argparse`, `typing`, `networkx` (as `nx`), `pyverilog.vparser.parser`, `pyverilog.vparser.ast`, `pyverilog.dataflow.dataflow_analyzer`, `pyverilog.dataflow.optimizer`, `pyverilog.dataflow.walker`.
*   **How it's used:** This script/class is used to gain an understanding of the RTL design's structure and behavior directly from the source code. The information extracted (`rtl_knowledge` in `gen_plan.py`) can be used to:
    *   Refine a Knowledge Graph (e.g., add RTL-specific nodes and edges).
    *   Provide direct RTL context to LLMs when generating plans or SVAs.
    *   Help identify critical areas for verification.
    When run as a standalone script, it prints its analysis to the console.

---

## `src/rtl_kg.py`

*   **Purpose:** This script is focused on processing RTL (Register Transfer Level) design information, specifically using the `RTLAnalyzer` class (from `rtl_analyzer.py`), and then transforming this information into a structured Knowledge Graph (KG) using `networkx`. It also includes utilities for exporting this KG.
*   **Key Functions:**
    *   `extract_rtl_knowledge(design_dir, output_dir=None, verbose=False)`:
        *   Instantiates `RTLAnalyzer` with the `design_dir`.
        *   Calls `analyzer.analyze_design()` to perform the RTL analysis.
        *   Structures the results from `analyzer` into a dictionary `rtl_knowledge`. This dictionary includes:
            *   `design_info`: Basic counts (files, modules, primary signals).
            *   `modules`: Detailed information about each module from `analyzer.module_info`.
            *   `files`: Information about processed files.
            *   `fsm_info`: Detected FSMs.
            *   `protocol_patterns`: Identified protocol patterns.
            *   `verification_points`: Suggestions for verification based on analysis (generated by `extract_verification_points`).
            *   `signal_types`: Detailed type information for signals.
            *   `primary_signals`: List of I/O ports.
            *   `combined_content`: The concatenated content of all RTL files.
        *   If `output_dir` is specified, it saves the `analyzer.data_flow_graph` as GraphML and the `rtl_knowledge` dictionary as JSON (using `make_json_serializable` to handle non-serializable types).
        *   Returns the `rtl_knowledge` dictionary.
    *   `extract_verification_points(analyzer)`:
        *   Extracts or generates verification points.
        *   If `analyzer.verification_suggestions` (from `RTLAnalyzer`) exists, it uses those.
        *   Otherwise, it generates basic verification points based on `analyzer.protocol_patterns` (e.g., reset behavior, data stability, handshaking).
    *   `build_knowledge_graph(rtl_knowledge)`:
        *   Takes the `rtl_knowledge` dictionary as input.
        *   Constructs an `nx.MultiDiGraph` (a directed graph that can have multiple edges between two nodes).
        *   Uses a helper `get_node_id` to create unique integer IDs for KG nodes, mapping them from original string identifiers (e.g., "module:mod\_name", "port:mod\_name.port\_name").
        *   Adds nodes to the KG for:
            *   Modules (`type="module"`)
            *   Ports (`type="port"`), linking them to their modules.
            *   FSMs (`type="fsm"`), linking them to their modules.
            *   Protocol Patterns (`type="protocol_pattern"`), linking to modules and signals.
            *   Verification Points (`type="verification_point"`), linking to modules and signals.
            *   Control Flow structures (if, case, loop) (`type="control_flow"`), linking to modules and relevant signals.
            *   Assignments (`type="assignment"`), linking to modules and involved signals (LHS and RHS).
            *   Internal Signals (`type="signal"`), if not already ports, linking to modules.
        *   Adds edges representing relationships like "input\_to", "outputs", "instantiates", "drives", "part\_of", "found\_in", "includes", "targets", "involves", "references", "switches\_on", "assigns\_to", "used\_in". Each edge is given a unique ID (e.g., "e0", "e1").
        *   Stores `node_id_map` (original string ID to integer ID) and `reverse_id_map` as graph attributes.
        *   Returns the constructed `nx.MultiDiGraph`.
    *   `export_graph_to_graphml(kg, output_path, simplify=False)`:
        *   Exports the `kg` to a GraphML file.
        *   Handles potential serialization issues by converting complex attributes (dicts, lists) to JSON strings using `make_json_serializable`.
        *   Has a `simplify` option to create a version with fewer attributes, possibly for better compatibility with some visualization tools.
        *   Includes a fallback to save a minimal graph if the full export fails.
    *   `save_knowledge_graph(kg, output_path, output_dir)`: (Seems somewhat redundant with `export_graph_to_graphml` but might have slightly different handling or was an earlier version). It tries to save as GraphML and falls back to JSON.
    *   `make_json_serializable(obj)`: Recursively converts an object (dicts, lists, sets, etc.) into a JSON-serializable representation by stringifying non-standard types.
    *   `save_ultra_simplified_gephi_kg(kg, output_path)`: Creates a very simplified version of the KG specifically for Gephi visualization, ensuring basic attributes like 'id', 'label', 'type'.
    *   `write_graphml_with_unique_edge_ids(G, path)`: A utility to ensure all edges in the GraphML output have unique 'id' attributes, which can be important for tools like Gephi.
    *   `main()` (in `if __name__ == "__main__":`):
        *   Argument parsing for `design_dir`, `output_dir`, `verbose`.
        *   Calls `extract_rtl_knowledge`.
        *   Calls `build_knowledge_graph`.
        *   Calls `export_graph_to_graphml` and `save_knowledge_graph`.
        *   Attempts to generate a PNG visualization of the KG using `matplotlib` and `nx.spring_layout` (this can be slow or memory-intensive for large graphs).
        *   Calls `save_ultra_simplified_gephi_kg`.
*   **Functionality:**
    1.  Uses `RTLAnalyzer` to parse RTL code and extract detailed information.
    2.  Transforms this extracted RTL information into a formal Knowledge Graph (a `networkx.MultiDiGraph`). Nodes represent modules, ports, signals, FSMs, assignments, etc., and edges represent their relationships.
    3.  Provides utilities to export the generated KG in GraphML format, with considerations for compatibility with visualization tools like Gephi.
    4.  Can also save the extracted RTL knowledge as a JSON file.
*   **Dependencies:** `saver.saver`, `os`, `sys`, `json`, `networkx` (as `nx`), `matplotlib.pyplot` (optional, for visualization), `rtl_analyzer.RTLAnalyzer`, `argparse`.
*   **How it's used:** This script is likely run as a standalone tool or as part of a larger pipeline (e.g., called by `build_KG` if the task involves creating an RTL-based KG). The `extract_rtl_knowledge` function is also used by `rtl_parsing.py` to get RTL information for refining a spec-based KG. The output KG (GraphML or JSON) can then be consumed by other analysis tools or context generators.

---

## `src/rtl_parsing.py`

*   **Purpose:** This script focuses on integrating knowledge derived from RTL (Register Transfer Level) analysis with an existing specification-based Knowledge Graph (KG). The main goal is to create a richer, more comprehensive KG that combines insights from both the design specification and the actual hardware implementation.
*   **Key Functions:**
    *   `refine_kg_from_rtl(spec_kg: nx.Graph) -> Tuple[nx.Graph, dict]`:
        *   The main public function. Takes an existing `spec_kg` (presumably built from design documents).
        *   Calls `link_spec_and_rtl_graphs` to perform the core linking logic.
        *   Calls `analyze_graph_connectivity` to report on the structure of the combined graph.
        *   Returns the `combined_kg` and the `rtl_knowledge` dictionary obtained from `extract_rtl_knowledge`.
    *   `link_spec_and_rtl_graphs(spec_kg: nx.Graph, design_dir: str) -> Tuple[nx.Graph, dict]`:
        *   `extract_rtl_knowledge(design_dir, ...)`: Calls the function from `rtl_kg.py` to analyze RTL files in `FLAGS.design_dir` and get structured `rtl_knowledge`.
        *   `build_knowledge_graph(rtl_knowledge)`: Calls the function from `rtl_kg.py` to build a new KG (`rtl_kg`) purely from the extracted RTL information.
        *   Creates `combined_kg` by copying `spec_kg`.
        *   Adds nodes and edges from `rtl_kg` into `combined_kg`, prefixing RTL node IDs with "rtl\_" to avoid clashes and adding a 'source': 'rtl' attribute.
        *   `link_modules_to_spec(combined_kg, rtl_node_mapping)`: Creates links between RTL module nodes and relevant specification nodes in the `combined_kg`. It adds a "design\_root" node and connects spec nodes and RTL module nodes to it. It also tries to link spec text nodes to RTL module nodes if the module name appears in the spec text.
        *   `link_signals_to_spec(combined_kg, rtl_node_mapping)`: Creates links between RTL signal/port nodes and spec text nodes if the signal name appears in the spec text.
        *   `ensure_graph_connectivity(combined_kg)`: Checks if the graph is connected and, if not, connects components to a "knowledge\_root" node.
        *   Returns the `combined_kg` and the `rtl_knowledge`.
    *   `link_modules_to_spec(...)`: (Detailed above) Creates "describes" relationships between spec nodes and RTL module nodes.
    *   `link_signals_to_spec(...)`: (Detailed above) Creates "references" relationships between spec nodes and RTL signal nodes.
    *   `ensure_graph_connectivity(kg: nx.Graph)`: (Detailed above) Ensures the graph is connected by adding a root node and linking disconnected components to it.
    *   `analyze_graph_connectivity(kg: nx.Graph)`: Prints information about the graph's connectivity, including the number of connected components, bridges between RTL and spec sections, and high-degree (hub) nodes.
*   **Functionality:**
    1.  Orchestrates the analysis of RTL code using `rtl_kg.extract_rtl_knowledge`.
    2.  Constructs a separate KG from this RTL analysis using `rtl_kg.build_knowledge_graph`.
    3.  Merges this RTL KG with an existing specification-based KG.
    4.  Establishes explicit links between the specification parts and RTL parts of the combined KG, primarily by:
        *   Creating a common "design\_root" to bridge spec and RTL module hierarchies.
        *   Matching module names and signal names found in RTL with text in specification nodes.
    5.  Analyzes and reports on the connectivity of the resulting combined graph.
*   **Dependencies:** `utils_gen_plan.count_tokens_in_file` (though not directly used in the provided snippet of this file, it's imported), `os`, `re`, `sys`, `networkx` (as `nx`), `config.FLAGS`, `saver.saver`, `rtl_kg.extract_rtl_knowledge`, `rtl_kg.build_knowledge_graph`.
*   **How it's used:** This script is called by `gen_plan.py` when `FLAGS.refine_with_rtl` is true. It takes a KG built from specifications and enriches it by parsing the corresponding RTL code, creating a new KG from RTL, and then intelligently merging and linking the two. The resulting `combined_kg` provides a more holistic view of the design, connecting high-level requirements to low-level implementation details. This richer KG can then be used by context generators for more effective prompt augmentation.

---

## `src/saver.py`

*   **Purpose:** This script defines a `Saver` class responsible for comprehensive logging, saving of results, and managing output directories for experiments or runs. It centralizes how information (text logs, plots, pickled objects, model info, configuration) is saved to disk. It also includes a `MyTimer` class.
*   **Key Classes and Methods:**
    *   `MyTimer`:
        *   `__init__(self)`: Records the start time.
        *   `elapsed_time(self)`: Returns the elapsed time in integer minutes.
    *   `Saver`:
        *   `__init__(self)`:
            *   First, it imports and calls `parse_command_line_args` (from `command_line_args.py`) which updates `config.FLAGS`. This means the `Saver`'s behavior can be influenced by command-line arguments at instantiation time.
            *   Constructs `self.logdir` path using `FLAGS.task`, timestamp, hostname, and user. This directory will store all outputs for a run.
            *   Creates the `logdir` and subdirectories `plotdir` (for plots) and `objdir` (for pickled objects).
            *   Opens `model_info.txt` for writing.
            *   Calls `_log_model_info()` to save initial model/config details.
            *   Calls `_save_conf_code()` to save a copy of `config.py` and the `FLAGS` object.
            *   Initializes `self.timer = MyTimer()`.
            *   Initializes `self.stats` (a `defaultdict(list)`) for accumulating statistics.
        *   `get_log_dir()`, `get_plot_dir()`, `get_obj_dir()`: Return paths to respective directories.
        *   `log_list_of_lists_to_csv(...)`, `log_dict_of_dicts_to_csv(...)`: Save data to CSV files.
        *   `save_emb_accumulate_emb(...)`, `save_emb_save_to_disk(...)`, `save_emb_dict(...)`: Methods for saving embedding data (likely for machine learning models), potentially accumulating them before saving.
        *   `log_dict_to_json(...)`: Saves a dictionary to a JSON file.
        *   `log_model_architecture(self, model)`: Writes model architecture (e.g., a PyTorch model string) and estimated size to `model_info.txt`.
        *   `log_info(self, s, silent=False, build_str=None)`:
            *   Primary logging method. Prints `s` to console (unless `silent`).
            *   Writes `s` to `log.txt` in `logdir`.
            *   Can pretty-print lists/dicts as JSON.
            *   Can optionally append to a `build_str`.
        *   `log_info_once(...)`, `log_info_at_most(...)`: Variants of `log_info` to prevent duplicate messages or limit message frequency.
        *   `info(...)`, `error(...)`, `warning(...)`, `debug(...)`: Timed logging methods that write to `log.txt`, `error.txt`, or `debug.txt` with an elapsed time prefix.
        *   `save_dict(self, d, p, subfolder='')`: Saves a dictionary as a pickle file.
        *   `_save_conf_code(self)`: Saves the content of `config.py` (extracted via `extract_config_code` from `utils.py`) and the current `FLAGS` object.
        *   `save_graph_as_gexf(self, g, fn)`: Saves a `networkx` graph as a GEXF file.
        *   `save_overall_time(...)`, `save_exception_msg(...)`: Save total time and exception messages.
        *   `_log_model_info(self)`: Writes detailed model and configuration information (from `FLAGS`) to `model_info.txt`.
        *   `_save_to_result_file(...)`: Appends various objects/strings to `results.txt`.
        *   `save_stats(...)`, `print_stats()`: Accumulate named statistics and print summaries (e.g., mean, std) at the end.
        *   `close(self)`: Closes open file handlers and prints final stats.
    *   `NoOpContextManager`: A simple context manager that does nothing, used internally.
    *   Global `saver` instance: `saver = Saver()` is instantiated at the end, making a global `saver` object available for import and use throughout the project.
*   **Functionality:**
    1.  Provides a centralized and standardized way to log information and save various types of data (text, objects, plots, configuration) during a program run.
    2.  Automatically creates a unique, timestamped log directory for each run.
    3.  Handles command-line argument parsing at initialization to configure its behavior and the global `FLAGS`.
    4.  Includes utilities for timing and basic statistics reporting.
    5.  Offers different levels and styles of logging (info, error, debug, once, limited frequency).
*   **Dependencies:** `config.FLAGS`, `command_line_args.parse_command_line_args`, various functions from `utils.py` (related to time, paths, saving, plotting, system info), `json`, `collections.OrderedDict`, `collections.defaultdict`, `pprint`, `os.path.join`, `os.path.dirname`, `os.path.basename`, `torch` (conditionally, for tensor checks), `networkx` (for GEXF saving), `numpy`, `time`, `csv`.
*   **How it's used:** The global `saver` object is imported in many other scripts (e.g., `from saver import saver`). Its methods like `saver.log_info(...)`, `saver.save_dict(...)`, etc., are then used for all output logging and data saving needs, ensuring consistency and organization of output files. The `print = saver.log_info` pattern is common, redirecting standard print statements to the logging mechanism.

---

## `src/sva_extraction.py`

*   **Purpose:** This script provides functions to extract SystemVerilog Assertions (SVAs) from text blocks, typically the output of an LLM. It employs multiple regular expression-based strategies to identify and parse SVA statements.
*   **Key Functions:**
    *   `extract_svas_strategy1(result: str) -> List[str]`: Finds blocks wrapped in "SVA: \`\`\` ... \`\`\`" and then extracts SVA patterns (`@(posedge...);`) from within these blocks.
    *   `extract_svas_strategy2(result: str) -> List[str]`: Directly matches the core SVA pattern (`@(posedge...);`) anywhere in the input `result` string.
    *   `extract_svas_strategy3(result: str) -> List[str]`: Looks for blocks like "SVA for Plan X: ... \`\`\` ... \`\`\`" and extracts SVAs from the code block.
    *   `extract_svas_strategy4(result: str) -> List[str]`: Splits the input by "SVA for Plan X:" headers and then searches for SVA code blocks within each resulting section.
    *   `extract_svas_strategy5(result: str) -> List[str]`: Looks for "```systemverilog ... ```" blocks and then extracts `property ... endproperty; assert property(...);` structures.
    *   `extract_svas_strategy6(result: str) -> List[str]`: Finds commented-out assertions (e.g., `// assert property(...)`) within general code blocks.
    *   `extract_svas_strategy7(result: str) -> List[str]`: Finds `assert property(...);` lines within general code blocks, not necessarily commented.
    *   `extract_svas_strategy8(result: str) -> List[str]`: Matches a more structured `property P; @(posedge clk) ...; endproperty assert property(P);` pattern.
    *   `extract_svas_strategy9(result: str) -> List[str]`: Matches `assert property (@(posedge clk) ...;)` pattern.
    *   `clean_sva(sva: str) -> str`: A helper function to remove multi-line (`/* ... */`) and single-line (`// ...`) comments from an SVA string and normalize whitespace.
    *   `extract_svas_from_block(block: str) -> List[str]`:
        *   The main public function.
        *   Iterates through a dictionary of the strategies defined above.
        *   Calls each strategy function on the input `block`.
        *   Collects all unique SVAs extracted by any strategy into a `set` to avoid duplicates.
        *   Cleans each extracted SVA using `clean_sva`.
        *   Prints a summary of how many SVAs each strategy found.
        *   Returns a list of unique, cleaned SVAs.
*   **Functionality:**
    1.  Provides a robust mechanism for extracting SVAs from potentially messy or variably formatted text (like LLM outputs).
    2.  Uses a multi-strategy approach, increasing the chances of correctly identifying SVAs even if the formatting isn't perfect or consistent.
    3.  Cleans up extracted SVAs by removing comments and standardizing whitespace.
    4.  Reports on the effectiveness of each extraction strategy.
*   **Dependencies:** `re`, `typing.List`, `typing.Dict`, `typing.Optional`, `typing.Set`, `saver.saver` (for `print = saver.log_info`).
*   **How it's used:** This script is primarily used in `gen_plan.py` after an LLM generates text that is expected to contain SVAs. The `extract_svas_from_block` function is called to parse this text and retrieve a clean list of SVA strings, which can then be written to files for formal verification.

---

## `src/use_KG.py`

*   **Purpose:** This script provides a command-line interface or function to query a pre-built Knowledge Graph (KG) using the GraphRAG tool. It's designed for direct interaction or testing of the KG.
*   **Key Functions:**
    *   `use_KG()`:
        *   The main function for this task.
        *   Retrieves `KG_root` (path to KG artifacts), `graphrag_method` (e.g., 'local'), and the `query` string from `FLAGS`.
        *   Constructs a command to run GraphRAG query: `python -m graphrag.query --data <KG_root> --method <method> "<query>"`. It also sets `PYTHONPATH` to include `FLAGS.graphrag_local_dir`.
        *   Executes this command using `subprocess.Popen` and streams its output.
        *   Captures the full output and assumes the last line of the output is the answer to the query.
        *   Uses `utils.OurTimer` for timing.
*   **Functionality:**
    1.  Provides a way to execute queries against a GraphRAG KG from the command line or programmatically.
    2.  Handles the construction and execution of the GraphRAG query command.
    3.  Prints the query output and attempts to extract a final answer.
*   **Dependencies:** `os`, `subprocess`, `shutil`, `json`, `re`, `pathlib.Path`, `PyPDF2`, `logging`, `utils.OurTimer`, `saver.saver`, `config.FLAGS`.
*   **How it's used:** This script is run when `FLAGS.task` is set to `'use_KG'`. It's a utility for interacting with and getting responses from the generated Knowledge Graph via GraphRAG's querying interface.

---

## `src/utils.py`

*   **Purpose:** This script is a general-purpose utility module containing a wide variety of helper functions and classes used across the project. These range from path manipulation and file operations to plotting, data structure manipulation, and system information retrieval.
*   **Key Components:**
    *   **Path Helpers:** `get_root_path()`, `get_log_path()`, `get_file_path()`, `get_save_path()`, `get_src_path()`, `create_dir_if_not_exists()`, `proc_filepath()`, `append_ext_to_filepath()`.
    *   **Data Handling & Saving/Loading:**
        *   `save(obj, filepath, ...)` & `load(filepath, ...)`: General save/load functions using Klepto by default.
        *   `save_klepto(...)`, `load_klepto(...)`: Specific Klepto saving/loading.
        *   `save_pickle(obj, filepath, ...)` & `load_pickle(filepath, ...)`: Pickle-based saving/loading.
    *   **List/Sequence Helpers:** `argsort(seq)`, `sorted_nicely(l, reverse=False)`.
    *   **Execution & System:**
        *   `exec_cmd(cmd, timeout=None, exec_print=True)`: Executes a shell command with optional timeout.
        *   `get_ts()`, `set_ts(ts)`, `get_current_ts(zone='US/Pacific')`: Timestamp generation.
        *   `timeout` class: A context manager for function timeouts using signals.
        *   `get_user()`, `get_host()`: Get username and hostname.
        *   `get_best_gpu(...)`, `get_gpu_info()`, `print_gpu_free_mem_info(...)`: GPU selection and information utilities (uses `nvidia-smi`).
        *   `format_file_size(...)`: Formats bytes into human-readable sizes.
        *   `report_save_dir(save_dir)`: Calculates total size and file count in a directory.
    *   **Graph Helpers (NetworkX):** `assert_valid_nid()`, `assert_0_based_nids()`, `node_has_type_attrib()`, `print_g()`, `create_edge_index()`.
    *   **Plotting (Matplotlib/Seaborn):** `plot_dist()`, `_analyze_dist()`, `plot_scatter_line()`, `plot_points()`, `multi_plot_dimension()`, `plot_scatter_with_subplot()`, `plot_scatter_with_subplot_trend()`, `plot_points_with_subplot()`.
    *   **PyTorch Related (Simple MLP):**
        *   `MLP` class: A basic Multi-Layer Perceptron model.
        *   `MLP_multi_objective` class: An MLP with multiple output heads for multi-task learning.
        *   `create_act(act, ...)`: Creates activation function layers.
    *   **Miscellaneous:**
        *   `C` class: A simple counter.
        *   `OurTimer` class: For timing code blocks and logging durations.
        *   `format_seconds(seconds)`: Formats seconds into a human-readable string (years, months, days, etc.).
        *   `random_w_replacement(input_list, k=1)`: Random sampling with replacement.
        *   `get_sparse_mat(...)`: Creates a SciPy sparse matrix.
        *   `prompt(str, options=None)`, `prompt_get_cpu()`, `prompt_get_computer_name()`: User input prompting.
        *   `parse_as_int(s)`: Safely parses a string to an int.
        *   `get_model_info_as_str(FLAGS)`, `extract_config_code()`: For saving configuration details.
        *   `TopKModelMaintainer` class: Helper to save the top K best models during training based on validation loss.
        *   `coo_to_sparse(...)`: Converts a SciPy COO matrix to a PyTorch sparse tensor.
        *   `estimate_model_size(model, label, saver)`: Estimates and logs the size of a PyTorch model.
        *   `create_loss_dict_get_target_list(...)`, `update_loss_dict(...)`: Helpers for managing loss dictionaries in training.
        *   `format_loss_dict(...)`: Formats a loss dictionary for printing.
*   **Functionality:** This module provides a toolkit of commonly needed functions, preventing code duplication and promoting consistency in areas like file I/O, logging, path management, plotting, and basic machine learning model components.
*   **Dependencies:** `pytz`, `datetime`, `pathlib.Path`, `sys`, `scipy`, `scipy.sparse`, `numpy`, `signal`, `pickle`, `random`, `requests` (though not used in visible code), `re`, `time`, `threading.Timer`, `subprocess`, `klepto`, `collections.OrderedDict`, `socket.gethostname`, `os` (makedirs, system, environ, remove, path functions), `networkx`, `seaborn`, `matplotlib.pyplot`, `torch`, `torch.nn`, `scipy.stats.mstats`, `copy.deepcopy`, `config.FLAGS` (implicitly, as some functions might use it).
*   **How it's used:** Functions and classes from `utils.py` are imported and used extensively throughout the project by most other modules. For example, `OurTimer` is used for performance tracking, path functions for locating files, and saving/loading functions for data persistence.

---

## `src/utils_LLM.py`

*   **Purpose:** This file is intended as a placeholder for general utility functions related to Large Language Model (LLM) interactions.
*   **Key Functions/Classes (Expected):**
    *   `get_llm(model_name, **kwargs)`: (Referenced in `gen_plan.py`) Expected to initialize and return an LLM client or agent based on the `model_name` and other arguments.
    *   `llm_inference(llm_agent, prompt, tag)`: (Referenced in multiple files) Expected to take an initialized `llm_agent`, a `prompt` string, and a `tag` (for logging/tracking), send the prompt to the LLM, and return the LLM's textual response.
    *   `count_prompt_tokens(prompt_text)`: (Referenced in `dynamic_prompt_builder.py`, often implemented using `tiktoken`) Expected to calculate the number of tokens a given `prompt_text` would consume for a specific LLM.
*   **Current Content:** The provided file is currently empty except for copyright comments and a placeholder comment: "# Add your own LLM client".
*   **Functionality (Conceptual):**
    1.  Abstract the specifics of LLM API interactions.
    2.  Provide a standardized interface for sending prompts and receiving responses from LLMs.
    3.  Offer helper functions for common LLM-related tasks, such as token counting.
*   **Dependencies (Conceptual):** Likely to include libraries such as `openai`, `tiktoken`, or other LLM client libraries, and would depend on configurations from `config.FLAGS`.
*   **How it's used (Intended):** Modules like `gen_plan.py`, `dynamic_prompt_builder.py`, and `design_context_summarizer.py` import and call functions (e.g., `llm_inference`, `get_llm`) that are expected to be defined in this file to interact with LLMs.

---

## `src/utils_LLM_client.py`

*   **Purpose:** This file is intended as a placeholder for the specific client implementation for interacting with a Large Language Model (LLM) API or service.
*   **Current Content:** The provided file is currently empty except for copyright comments and a placeholder comment: "# Add your own LLM client".
*   **Functionality (Conceptual):**
    1.  Contain the low-level code for making requests to an LLM (e.g., OpenAI API, a local Hugging Face model endpoint, etc.).
    2.  Handle authentication, request formatting, response parsing, and error handling related to LLM communication.
    3.  The functions defined in `src/utils_LLM.py` (like `llm_inference`) would likely call functions or use classes defined in this client file.
*   **Dependencies (Conceptual):** Would depend heavily on the chosen LLM provider or library (e.g., `openai`, `requests`, `transformers`).
*   **How it's used (Intended):** This module would encapsulate the direct communication logic with the LLM service, providing a cleaner separation of concerns. Higher-level LLM utility functions in `utils_LLM.py` would use this client.

---

## `src/utils_gen_plan.py`

*   **Purpose:** This script contains utility functions specifically tailored for the `gen_plan.py` workflow, particularly those related to SVA generation, JasperGold execution, and results analysis for formal verification.
*   **Key Functions:**
    *   `analyze_coverage_of_proven_svas(svas: List[str], jasper_reports: List[str]) -> str`:
        *   Collects SVAs that were proven (based on `extract_proof_status` from Jasper reports).
        *   Creates a combined SVA file (`combined_proven_svas.sva`) containing all proven SVAs, embedded in the design's module interface.
        *   Generates a TCL script (`coverage_analysis.tcl`) to run JasperGold for coverage analysis on these combined proven SVAs. This TCL script includes commands to initialize coverage, analyze design files, elaborate, define clock/reset, run proofs (with a time limit), and then report various coverage metrics (stimuli/COI for functional, statement, toggle, expression, branch).
        *   Executes JasperGold with this TCL script.
        *   Saves the coverage report and returns the stdout of the JasperGold run.
        *   Includes logic to find the original design TCL file to extract necessary commands like `elaborate -top`, clock/reset definitions, and design file names.
    *   `find_and_read_original_tcl(design_dir: str) -> Tuple[str, str]`: Searches `design_dir` for a `.tcl` file (likely the main project TCL) and reads its content.
    *   `extract_top_module(tcl_content: str) -> str`: Parses TCL content to find the `elaborate -top <module_name>` line and extract the top module name.
    *   `extract_design_file(tcl_content: str) -> str`: Parses TCL content (handling line continuations) to find the Verilog design file specified in `analyze -v2k ${RTL_PATH}/<design_file.v>`.
    *   `find_original_tcl_file(logdir: str) -> str`: (Note: This seems to be a duplicate name of a function that should search `design_dir`. The one in `gen_plan.py` searches `design_dir`, this one in `utils_gen_plan.py` searches `logdir/tcl_scripts`. This might be an oversight or for a different purpose, perhaps finding generated TCLs). The one that searches `design_dir` for `FPV_*.tcl` is also present in this file.
    *   `extract_clock_and_reset(tcl_content: str) -> Tuple[str, str, str]`: Parses TCL content for `stopat -env`, `clock`, and `reset` commands.
    *   `extract_proof_status(report_content: str) -> str`: Parses JasperGold report text to determine the proof status of assertions (proven, cex, inconclusive, error). It specifically looks at the "assertions" part of the summary.
    *   `count_tokens_in_file(file_path)`: Reads a file and counts tokens using `tiktoken.get_encoding("cl100k_base").encode()`.
*   **Functionality:**
    1.  Provides specialized helpers for the formal verification flow using JasperGold.
    2.  Automates the setup and execution of coverage analysis runs for proven SVAs.
    3.  Includes robust parsing of JasperGold reports and TCL scripts to extract necessary information (proof status, top module, design files, clock/reset commands).
    4.  Offers a utility for token counting, essential for managing LLM prompt sizes.
*   **Dependencies:** `utils.get_ts`, `config.FLAGS`, `saver.saver`, `typing.Tuple`, `typing.List`, `os`, `subprocess`, `re`, `tiktoken`, `shutil`.
*   **How it's used:** These functions are primarily called from `gen_plan.py` to manage the interaction with JasperGold, prepare input files for it, and interpret its output, especially for coverage analysis.

---

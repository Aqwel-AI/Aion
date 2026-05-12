# Changelog

All notable changes to the Aqwel-Aion project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-12

### Added — 10 New Modules
- **`aion.agents`** — Autonomous agent framework: `ReActAgent` (observe/think/act loop), `PlanningAgent` (task decomposition), `MultiAgent` (role-based orchestration); conversation memory: `SlidingWindowMemory`, `SummaryMemory`, `TokenBudgetMemory`.
- **`aion.cache`** — Caching with TTL: `MemoryCache` (in-memory, thread-safe), `DiskCache` (SQLite-backed), `LLMCache` (prompt-keyed response cache with hit/miss stats), `@cached` decorator.
- **`aion.data`** — Data processing: `load_csv`/`load_json`/`load_jsonl` loaders + savers; `train_test_split`, `train_val_test_split`, `kfold_split` with stratification; text augmentation (`random_delete`, `random_swap`, `random_insert`, `synonym_replace`, `augment_text`); schema validation (`Schema`, `Field`, `validate_record`, `validate_dataset`).
- **`aion.tokenizer`** — Trainable tokenizers: `BPETokenizer` (byte-pair encoding), `WordPieceTokenizer` (BERT-style), `Vocabulary` (bidirectional token-to-id mapping, special tokens, save/load).
- **`aion.pipeline`** — Step-based processing pipelines: `Pipeline`, `Step`, `FunctionStep`, `MapStep`, `FilterStep`, `BatchStep`; per-step timing, retry, fallback, dry-run, JSON serialization.
- **`aion.store`** — Persistent storage: `KeyValueStore` (SQLite with namespaces), `PersistentVectorStore` (cosine similarity search), `ChatHistoryStore` (conversation threads with full-text search).
- **`aion.tracker`** — Experiment tracking: `Tracker`/`Run` for logging params, metrics, tags, artifacts; `compare_runs`/`best_run` for experiment comparison. All data stored as local JSON.
- **`aion.llm_eval`** — LLM output evaluation: `semantic_similarity`/`batch_similarity`, `faithfulness_score`/`check_groundedness`, `toxicity_check`/`contains_pii`, `estimate_cost`/`CostTracker`.
- **`aion.structures`** — Advanced data structures: `Trie`, `BloomFilter`, `LRUCache`, `MinHeap`/`MaxHeap`/`PriorityQueue`, `UnionFind`.
- **`aion.serve`** — REST API serving: `AionServer`/`create_app` with FastAPI `/chat`, `/rag`, `/health` endpoints; custom route registration; CORS.

### Added — New extras
- **`[serve]`** optional dependency group: `fastapi`, `uvicorn`.

### Fixed
- Missing `matrix_transpose`, `matrix_multiply`, `z_score_normalization`, `min_max_scaling` in `aion.algorithms.arrays`.
- `a_star` and `pagerank` import name mismatches in `aion.algorithms.__init__`.

## [0.1.9] - 2026-03-29

### Added
- **`aion.tools`** — OpenAI-style tool schemas, `ToolRegistry`, `run_tool_loop`, HTTP retry helper, token bucket, optional `tiktoken` estimates (`pip install aqwel-aion[tools]`).
- **`aion.rag`** — Text chunking, `MemoryVectorStore`, optional `FaissVectorStore`, `SimpleRAGIndex` pipeline (`[rag]` extra).
- **Providers** — `complete_turn` / `AssistantTurn` / `NormalizedToolCall` for OpenAI and OpenAI-compatible APIs; `complete_turn` stubs on Anthropic/Gemini with clear `NotImplementedError`.
- **`aion.config`**, **`aion.env`** — TOML/YAML config merge with env overrides, `.env` parsing.
- **`aion.benchmarks`** — Timing helpers. Offline tool-loop testing uses **`aion.tools.FakeToolProvider`** (former checkpoint sidecars use stdlib JSON in `save_checkpoint_sidecar_meta`).
- **Graph algorithms** — `dijkstra`, `connected_components`, `shortest_path_unweighted` in `aion.algorithms.graphs`.
- **Visualization** — `plot_3d_scatter`, `plot_3d_surface`, `save_figures_pdf`, `figures_to_html_img_tags`.
- **Former** — `save_checkpoint_sidecar_meta` for JSON metadata next to `.npz` checkpoints.
- **Optional extras** in `pyproject.toml` / `setup.py`: `[tools]`, `[rag]`, `[config]`; `[full]` extended with tiktoken, tomli (Python 3.8–3.10), PyYAML.

### Removed (package trim)
- Top-level **`aion.datasets`** and **`aion.dataframe`** — use **`aion.io`**, the stdlib, or **pandas** directly; transformer text pipelines remain under **`aion.former.datasets`**.

## [0.1.8] - 2025-02-22

### Changed
- Project metadata: authors, maintainers, Documentation URL (https://aqwelai.xyz/#/docs), Homepage (https://aqwelai.xyz)
- pyproject.toml: license, classifiers (Python 3.13, Science/Research, Apache-2.0), optional dependency versions

## [0.1.7] - 2025-01-20

### REVOLUTIONARY UPDATE - Complete AI Research Library

#### Added
- **Complete Mathematics Library** - 71+ mathematical functions for AI/ML research
  - Linear algebra operations (matrices, eigenvalues, SVD, determinants)
  - Advanced statistics (correlation, regression, probability distributions)
  - Machine learning utilities (activation functions, loss functions, distance metrics)
  - Signal processing (FFT, convolution, filtering)
  - Data science pipeline tools (normalization, scaling, preprocessing)

- **AI Research Capabilities**
  - Text embeddings with sentence transformers integration
  - Vector similarity operations and semantic search
  - 11+ specialized AI prompt templates for research
  - Advanced code complexity analysis and quality assessment
  - Comprehensive model evaluation metrics

- **Professional Documentation System**
  - Automated PDF generation for research papers
  - Complete API documentation with 175+ functions
  - User guides with examples and tutorials
  - Professional formatting with academic standards

- **Enhanced Development Tools**
  - Advanced file management with professional operations
  - Enhanced code parser supporting 30+ programming languages
  - Real-time monitoring with intelligent change detection
  - Git integration and version control utilities

#### Changed
- **Branding Update**: LinkAI → Aqwel AI
- **Package Name**: linkai-aion → aqwel-aion
- **Focus**: General utilities → AI research library
- **Target Audience**: Developers → AI researchers and developers
- **URLs**: linkaiapps.com → aqwelai.com

#### Enhanced
- **Code Analysis Module** - Advanced complexity metrics and code smell detection
- **Evaluation Module** - Complete ML evaluation suite with classification and regression metrics
- **Embedding Module** - Professional text vectorization with fallback systems
- **Prompt Module** - Comprehensive prompt engineering templates and utilities

#### Technical Improvements
- Added comprehensive error handling across all modules
- Implemented proper type hints throughout the codebase
- Enhanced documentation with detailed examples and use cases
- Added optional dependency management for different use cases
- Improved performance and memory efficiency

### Quality Metrics
- **96/100 Quality Score** - Production-ready code
- **175+ Functions** across 13 core modules
- **100% Documentation Coverage** with automated generation
- **Complete Test Coverage** for critical functions

---

## [0.1.6] - 2024-12-15

### Added
- Enhanced file management with improved upload and organization
- Better multi-language code parser with syntax error support
- Real-time file change monitoring for auto-refresh
- Code snippet extraction for reuse and documentation
- Extended CLI support with new commands
- Comprehensive language detection for 30+ programming languages
- Enhanced security scanning with pattern detection
- Advanced text intelligence and analysis capabilities

### Changed
- Improved performance across all modules
- Better error handling and user feedback
- Enhanced documentation and examples

---

## [0.1.5] - 2024-11-20

### Added
- Initial file management utilities
- Basic code parsing functionality
- Text processing and analysis tools
- CLI interface foundation

### Changed
- Core architecture improvements
- Better module organization

---

## [0.1.0] - 2024-10-01

### Added
- Initial release of LinkAI-Aion
- Basic utility functions
- Text processing capabilities
- File management tools
- Code parsing foundation

---

## Upcoming Features

The Aqwel-Aion library continues to evolve based on user feedback and research community needs. We're committed to maintaining the highest standards of code quality, documentation, and user experience.

### Contributing

We welcome contributions from the AI research community! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to get involved.

### Support

For questions, issues, or feature requests:
- Visit: https://aqwelai.xyz/
- Issues: https://github.com/aqwelai/aion/issues
- Company Gmail: aqwelai.company@gmail.com

---

**Author:** Aksel Aghajanyan · **Developed by:** Aqwel AI Team — pioneering the future of AI research tools.
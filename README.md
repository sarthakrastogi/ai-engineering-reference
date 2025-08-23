<h1 align="center">
    AI Engineering Reference
</h1>
<p align="center">
    <p align="center">
    <a target="_blank" href="https://github.com/sarthakrastogi/ai-engineering-reference">
        <img src="https://img.shields.io/github/stars/sarthakrastogi/ai-engineering-reference?style=social" alt="GitHub Stars">
    </a>
    </p>
    <p align="center">A collection of learning material and tools for building production AI applications
    <br>
    </p>
<h4 align="center">
    <a href="https://github.com/sarthakrastogi/ai-engineering-reference" target="_blank">
        <img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg" alt="Contributions Welcome">
    </a>
    <a href="https://github.com/sarthakrastogi/ai-engineering-reference/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
    </a>
</h4>

---

## Table of Contents
- [Evals](#evals) - Tools for evaluating and testing LLM applications
- [Prompt Engineering](#prompt-engineering) - Frameworks for optimizing prompts automatically  
- [AI Safety](#ai-safety) - Advanced techniques for protecting RAG systems
- [Monitoring and Experiments](#monitoring-and-experiments) - Platforms for tracking experiments and model performance
- [RAG Improvement](#rag-improvement) - Advanced techniques for enhancing RAG systems
- [Query Management](#query-management) - Tools for routing and caching LLM applications
- [AI Agent Building](#ai-agent-building) - Frameworks for creating autonomous agents
- [AI Agent Tools](#ai-agent-tools) - Specialized tools for agent interactions
- [Computer Use](#computer-use) - Tools for AI agents to interact with computers and web interfaces
- [Vector Databases](#vector-databases) - Databases for storing vector embeddings
- [LLM Tuning](#llm-tuning) - Libraries for optimizing and quantizing models
- [LLM Inference](#llm-inference) - Runtime engines for deploying models efficiently
- [Memory](#memory) - Systems for persistent AI memory and context

---

# AI Engineering Reference

## Evals

### Learning material
| Name  | Description | Link |
|-------|-------------|------|
| Arize AI Eval Course | Learn how to systematically assess and improve AI agent performance with structured evaluations, error analysis, and disciplined evaluation-driven development processes for agentic workflows | [course](https://www.deeplearning.ai/short-courses/evaluating-ai-agents/) |
| Field Guide | Comprehensive guide to LLM evaluation methodologies, best practices, and practical implementation strategies for production AI systems | [blog](https://hamel.dev/blog/posts/field-guide/) |
| Evidently AI guide | In-depth guide on using LLMs as judges for evaluation, covering implementation patterns, bias considerations, and best practices for automated assessment | [blog](https://www.evidentlyai.com/llm-guide/llm-as-a-judge) |
| Databricks blog | Best practices for automated RAG evaluation and techniques for enhancing LLM-as-a-judge systems with grading notes and structured feedback mechanisms | [blog](https://www.databricks.com/blog/LLM-auto-eval-best-practices-RAG) [blog](https://www.databricks.com/blog/enhancing-llm-as-a-judge-with-grading-notes) |

### Tools
Tools for evaluating and testing LLM applications, from basic metrics to comprehensive evaluation frameworks.

| Name | Description | When to Use | Link |
|------|-------------|-------------|------|
| Ragas | Framework for evaluating Retrieval Augmented Generation (RAG) systems with automated metrics | When building RAG pipelines and need to evaluate retrieval and generation quality | [docs](https://docs.ragas.io) |
| DeepEval | Simple-to-use, open-source LLM evaluation framework similar to Pytest but specialized for LLM outputs | When unit testing LLM outputs and need comprehensive evaluation metrics like G-Eval, hallucination, RAGAS | [GitHub repo](https://github.com/confident-ai/deepeval) |
| AgentEvals | Readymade evaluators for agent trajectories focusing on intermediate steps agents take | When evaluating agentic applications and need to understand agent decision-making processes | [GitHub repo](https://github.com/langchain-ai/agentevals) |
| OpenAI Evals | Framework for evaluating LLMs and LLM systems with open-source registry of benchmarks | When evaluating OpenAI models or need standardized benchmarks for model comparison | [GitHub repo](https://github.com/openai/evals) |
| Athina Evals | Python SDK with 50+ preset evaluations for LLM-generated responses | When needing quick access to diverse evaluation metrics without custom implementation | [GitHub repo](https://github.com/athina-ai/athina-evals) |
| TruLens | Open-source ground truth evaluation for LLMs | When needing ground truth evaluation and comprehensive LLM observability | [docs](https://trulens.org/) |
| Galileo | Enterprise-scale evaluation platform for testing, evaluating, and benchmarking prompts, models, agents, and RAG pipelines with powerful insights and safety guardrails | When building GenAI systems at enterprise scale and need comprehensive evaluation intelligence with advanced analytics | [docs](https://v2docs.galileo.ai/sdk-api/python/reference/base) |
| Weights and Biases | AI developer platform for training, fine-tuning, and managing models from experimentation to production | When managing ML experiments, tracking metrics, and need comprehensive model lifecycle management | [docs](https://wandb.ai/site/)

---

## Prompt Engineering

### Learning material
| Name  | Description | Link |
|-------|-------------|------|
| DSPy Introduction by DataCamp | Comprehensive introduction to DSPy framework covering declarative programming approach to LLMs and automated prompt optimization techniques | [tutorial](https://www.datacamp.com/blog/dspy-introduction) |
| IBM DSPy Tutorial | Practical guide to prompt engineering with DSPy, covering structured programming for language models and optimization strategies | [tutorial](https://www.ibm.com/think/tutorials/prompt-engineering-with-dspy) |
| DigitalOcean DSPy Guide | Step-by-step tutorial on using DSPy for building modular AI systems with automated prompt optimization and reasoning capabilities | [tutorial](https://www.digitalocean.com/community/tutorials/prompting-with-dspy) |
| LLM Optimization Frameworks Analysis | In-depth review of test-time iterative refinement versus compile-time declarative programming for enhancing AI system performance | [blog](https://medium.com/@adnanmasood/beyond-prompt-engineering-how-llm-optimization-frameworks-like-textgrad-and-dspy-are-building-the-6790d3bf0b34) |

### Tools
Frameworks and tools for optimizing prompts automatically and building structured prompt systems.

| Name | Description | When to Use | Link |
|------|-------------|-------------|------|
| DSPy | Declarative framework for building modular AI software that compiles AI programs into effective prompts | When building structured AI systems and want to optimize prompts algorithmically rather than manually | [docs](https://dspy-docs.vercel.app/) |
| TextGrad | Automatic differentiation via text using LLMs to backpropagate textual gradients | When optimizing text-based variables like prompts, solutions, or any textual content through gradient-like feedback | [GitHub repo](https://github.com/zou-group/textgrad) |
| AdalFlow | PyTorch-like library to build and auto-optimize LLM workflows with auto-differentiative framework | When building LLM applications and need unified optimization for both zero-shot and few-shot prompts | [GitHub repo](https://github.com/SylphAI-Inc/AdalFlow) |
| Quality Prompts | Implementation of 58 prompting techniques from University of Maryland research | When needing research-backed prompting techniques and structured prompt optimization methods | [GitHub repo](https://github.com/sarthakrastogi/quality-prompts) |

---

## AI Safety

### Learning material
| Name  | Description | Link |
|-------|-------------|------|
| Red Teaming LLM Applications Course | Learn to enhance security of LLM applications with red teaming techniques using Giskard's open source library to identify vulnerabilities | [course](https://www.deeplearning.ai/short-courses/red-teaming-llm-applications/) |
| AI Red Teaming and Security Masterclass | Comprehensive course on AI security and red teaming by Learn Prompting, covering advanced techniques for testing and securing AI systems | [course](https://maven.com/learn-prompting-company/ai-red-teaming-and-ai-safety-masterclass) |
| OffSec LLM Red Teaming Training | Professional training path covering LLM security awareness, vulnerability analysis, and practical exploitation techniques for security professionals | [course](https://www.offsec.com/learning/paths/llm-red-teaming/) |
| Hands-On AI Red Teaming Course | Practical course for offensive cybersecurity researchers focusing on identifying and exploiting vulnerabilities in large language models | [course](https://www.udemy.com/course/hands-on-ai-llm-red-teaming/) |
| Microsoft AI Red Team Guide | Industry-leading guidance and best practices from Microsoft's AI Red Team for safeguarding organizational AI systems | [guide](https://learn.microsoft.com/en-us/security/ai-red-team/) |
| LLM Red Teaming Complete Guide | Step-by-step guide to detecting vulnerabilities like bias, PII leakage, and misinformation through adversarial prompting techniques | [guide](https://www.confident-ai.com/blog/red-teaming-llms-a-step-by-step-guide) |

### Tools
Advanced tools for protecting AI systems.

| Name | Description | When to Use | Link |
|------|-------------|-------------|------|
| Rival AI | Real-time detection of malicious user queries, to protect your AI agents in production | When you need comprehensive AI safety tools for production environments: 1. Real-time attack detection using custom lightweight models and 2. Automated red teaming and benchmarking | [GitHub repo](https://github.com/sarthakrastogi/rival) |

---

## Monitoring and Experiments

### Learning material
| Name  | Description | Link |
|-------|-------------|------|
| Langsmith Course | Introduction to LangSmith for monitoring, tracing, and debugging LLM applications with hands-on experience in production observability and evaluation workflows | [course](https://academy.langchain.com/courses/intro-to-langsmith) |

### Tools
Platforms for tracking experiments, monitoring model performance, and managing ML lifecycles.

| Name | Description | When to Use | Link |
|------|-------------|-------------|------|
| LangSmith | Monitor and trace LLM applications with debugging and evaluation tools | When building LangChain applications and need comprehensive tracing and monitoring | [docs](https://docs.smith.langchain.com/) |
| Weights & Biases | AI developer platform for training, fine-tuning, and managing models from experimentation to production | When managing ML experiments, tracking metrics, and need comprehensive model lifecycle management | [GitHub repo](https://github.com/wandb/wandb) |
| MLflow | Open-source platform for ML lifecycle management including experiment tracking, model packaging, and serving | When needing end-to-end ML lifecycle management with experiment tracking and model deployment | [GitHub repo](https://github.com/mlflow/mlflow) |
| Deepchecks | Holistic solution for AI & ML validation needs, testing data and models from research to production | When needing continuous validation of ML models and data quality monitoring in production | [docs](https://docs.deepchecks.com/stable) |
| HoneyHive | Unified LLMOps platform providing AI evaluation, testing, and observability tools with OpenTelemetry-native monitoring for collaborative LLM application development | When building LLM applications and need comprehensive evaluation, monitoring, and debugging tools with team collaboration features for production AI systems | [docs](https://www.honeyhive.ai/)

---

## RAG Improvement

### Learning material
| Name  | Description | Link |
|-------|-------------|------|
| Graph RAG vs Vector RAG Tutorial | Comprehensive tutorial with code examples comparing graph-based and vector-based RAG approaches for different use cases | [tutorial](https://ragaboutit.com/graph-rag-vs-vector-rag-a-comprehensive-tutorial-with-code-examples/) |
| Graph RAG Implementation Guide | Step-by-step guide on implementing Graph RAG using knowledge graphs and vector databases for enhanced information retrieval | [tutorial](https://towardsdatascience.com/how-to-implement-graph-rag-using-knowledge-graphs-and-vector-databases-60bb69a22759/) |
| Pinecone RAG Learning Hub | Comprehensive resource exploring limitations of foundation models and how RAG addresses them for chat, search, and agentic workflows | [guide](https://www.pinecone.io/learn/retrieval-augmented-generation/) |
| Vectors and Graphs Integration | Advanced guide on combining vectors and graphs for better understanding of complex, interconnected information in AI systems | [guide](https://www.pinecone.io/learn/vectors-and-graphs-better-together/) |

### Tools
Advanced techniques and tools for enhancing Retrieval Augmented Generation systems.

| Name | Description | When to Use | Link |
|------|-------------|-------------|------|
| GraphRAG | Structured, hierarchical approach to RAG using knowledge graphs instead of semantic search | When dealing with complex information that requires connecting disparate pieces of data or holistic understanding | [docs](https://microsoft.github.io/graphrag/) |
| Rerankers | Lightweight, unified API for various reranking and cross-encoder models with support for multiple architectures | When improving retrieval results in RAG systems and need flexible reranking model options | [GitHub repo](https://github.com/AnswerDotAI/rerankers) |

---

## Query Management

### Tools
Tools for routing requests, caching responses, and managing LLM application infrastructure.

| Name | Description | When to Use | Link |
|------|-------------|-------------|------|
| RouteLLM | Framework for cost-effective LLM routing by intelligently selecting between different models | When managing costs by routing simple queries to cheaper models and complex ones to premium models | [GitHub repo](https://github.com/lm-sys/RouteLLM) |
| GPTCache | Semantic cache for LLM applications to reduce costs and improve response times | When building LLM applications with repeated or similar queries that can benefit from intelligent caching | [docs](https://gptcache.readthedocs.io/) |

---

## AI Agent Building

### Learning material
| Name  | Description | Link |
|-------|-------------|------|
| LangGraph Course | Learn to build stateful, multi-actor AI applications using LangGraph's graph-based architecture with hands-on experience in complex AI workflows and state management | [course](https://academy.langchain.com/courses/intro-to-langgraph) |

### Tools
Frameworks and platforms for creating autonomous AI agents and multi-agent systems.

| Name | Description | When to Use | Link |
|------|-------------|-------------|------|
| LangGraph | Build stateful, multi-actor applications with LLMs using graph-based architecture | When building complex AI workflows with state management, human-in-the-loop, and multi-step reasoning | [GitHub repo](https://github.com/langchain-ai/langgraph) |
| AG2 | Multi-agent conversation framework enabling multiple AI agents to collaborate | When building systems where multiple specialized agents need to work together on complex tasks | [docs](https://ag2.ai/) |
| CrewAI | Framework for orchestrating role-playing, autonomous AI agents for collaborative task execution | When creating teams of AI agents with specific roles and responsibilities for business processes | [docs](https://www.crewai.com/) |
| AutoGen | Multi-agent conversation framework from Microsoft for automated task solving | When needing automated conversations between multiple agents for problem-solving and decision-making | [docs](https://microsoft.github.io/autogen/stable/) |
| SmolAgents | Minimal, hackable agents framework from Hugging Face with tool calling capabilities | When building lightweight agents with simple tool integration and minimal dependencies | [GitHub repo](https://github.com/huggingface/smolagents) |
| PydanticAI | Type-safe agent framework with structured outputs and validation | When building agents that require strict type safety and structured data handling | [docs](https://ai.pydantic.dev) |

---

## AI Agent Tools

### Tools
Specialized tools and services that AI agents can use to interact with external systems and data sources.

| Name | Description | When to Use | Link |
|------|-------------|-------------|------|
| Sonar | Perplexity's search API for real-time information retrieval and web search | When agents need access to current web information and real-time search capabilities | [docs](https://sonar.perplexity.ai/) |
| Tavily | AI-powered search API optimized for LLMs with structured, relevant results | When building RAG systems or agents that need high-quality, AI-optimized search results | [docs](https://tavily.com/) |

---

## Computer Use

### Tools

| Name | Description | When to Use | Link |
|------|-------------|-------------|------|
| Browser Use | AI agent tool for automated web browsing and interaction with websites | When agents need to navigate websites, fill forms, or extract information from web pages | [docs](https://browser-use.com/) |
| Rtrvr | Web navigation and automation tool for AI agents to interact with web interfaces | When building agents that need to perform complex web navigation and data extraction tasks | [docs](https://www.rtrvr.ai/docs/web-navigation) |

---

## Vector Databases

### Learning material
| Name  | Description | Link |
|-------|-------------|------|
| Building Applications with Vector Databases | DeepLearning.AI course on creating applications using Pinecone, including hybrid search and multimodal applications | [course](https://www.deeplearning.ai/short-courses/building-applications-vector-databases/) |
| Vector Databases for RAG Course | Comprehensive Coursera course covering ChromaDB fundamentals, similarity search, and recommendation systems development | [course](https://www.coursera.org/learn/vector-databases-for-rag-an-introduction) |
| RAG with Embeddings & Vector Databases | Scrimba course exploring advanced AI engineering concepts focusing on embeddings and vector database management | [course](https://www.coursera.org/learn/learn-embeddings-and-vector-databases) |
| Best Vector Database Courses Guide | Curated collection of top vector database courses covering semantic search, RAG applications, and AI memory systems | [guide](https://www.classcentral.com/report/best-vector-database-courses/) |
| Enhancing RAG with Pinecone | Practical tutorial on implementing high-performance semantic search using Pinecone vector database for RAG applications | [tutorial](https://adasci.org/how-to-enhance-rag-models-with-pinecone-vector-database/) |

### Tools
Specialized databases for storing and querying high-dimensional vector embeddings.

| Name | Description | When to Use | Link |
|------|-------------|-------------|------|
| ChromaDB | Simple vector store for embeddings with easy-to-use API | When building small to medium RAG applications and need straightforward vector storage | [docs](https://docs.trychroma.com/) |
| Pinecone | Scalable, hosted vector search service | When needing production-scale vector search with managed infrastructure and high performance | [docs](https://docs.pinecone.io/) |
| Weaviate | GraphQL-based vector database with schema management | When needing structured vector data with GraphQL queries and advanced filtering capabilities | [docs](https://weaviate.io/developers/weaviate) |
| Qdrant | High-performance vector search engine with advanced filtering | When requiring high-performance vector search with complex filtering and payload support | [docs](https://qdrant.tech/documentation/) |

---

## LLM Tuning

### Learning material
| Name  | Description | Link |
|-------|-------------|------|
| Hugging Face Quantization Guide | Official documentation covering 8-bit and 4-bit quantization techniques using bitsandbytes for memory-efficient LLM deployment | [docs](https://huggingface.co/docs/bitsandbytes/main/en/index) |
| Hugging Face Optimum Tutorials | Collection of guides for optimizing Transformer models for different hardware accelerators and deployment scenarios | [docs](https://huggingface.co/docs/optimum/main/en/index) |

### Tools
Libraries and tools for optimizing and quantizing large language models for efficient deployment.

| Name | Description | When to Use | Link |
|------|-------------|-------------|------|
| Bitsandbytes | Accessible large language models via k-bit quantization for PyTorch with 8-bit and 4-bit support | When needing to reduce memory usage of LLMs through quantization without significant performance loss | [docs](https://huggingface.co/docs/bitsandbytes/main/en/index) |
| Optimum | Tools to optimize Transformers models for accelerated training and inference on targeted hardware | When optimizing Hugging Face models for specific hardware like GPUs, TPUs, or specialized accelerators | [GitHub repo](https://github.com/huggingface/optimum) |
| Unsloth | Fast and memory-efficient fine-tuning of LLMs with up to 5x speedups | When fine-tuning open-source LLMs and need faster training with lower memory requirements | [docs](https://docs.unsloth.ai/) |

---

## LLM Inference

### Learning material
| Name  | Description | Link |
|-------|-------------|------|
| Ollama Tutorial | Step-by-step guide to running large language models locally with installation, configuration, and model management | [docs](https://ollama.com/docs) |
| Llama.cpp Guide | Technical documentation for optimized C/C++ inference engine with hardware-specific optimization techniques | [docs](https://github.com/ggml-org/llama.cpp/blob/master/README.md) |

### Tools
Runtime engines and servers for deploying and serving large language models efficiently.

| Name | Description | When to Use | Link |
|------|-------------|-------------|------|
| LiteLLM | Unified API to call 100+ LLM APIs using OpenAI format with load balancing and fallbacks | When building applications that need to work with multiple LLM providers through a consistent interface | [GitHub repo](https://github.com/BerriAI/litellm) |
| Ollama | Run large language models locally with simple installation and management | When needing to run LLMs on local machines for development, privacy, or offline scenarios | [docs](https://ollama.com/) |
| Llama.cpp | Inference engine for LLaMA models in pure C/C++ with optimized performance | When deploying LLaMA models with minimal dependencies and maximum performance on various hardware | [GitHub repo](https://github.com/ggml-org/llama.cpp) |
| vLLM | High-throughput and memory-efficient inference and serving engine for LLMs | When serving LLMs in production with high concurrent requests and need optimal throughput | [docs](https://docs.vllm.ai/en/latest/) |

---

## Memory
Systems for giving AI applications persistent memory and context across interactions.

| Name | Description | When to Use | Link |
|------|-------------|-------------|------|
| Cognee | Modular tool for organizing data and improving AI agent accuracy with structured memory algorithms | When building AI agents that need to connect data points and maintain context across interactions | [docs](https://docs.cognee.ai) |
| Mem0 | Universal, self-improving memory layer for LLM applications that enables personalized AI experiences | When building LLM applications that require persistent memory and personalization across user sessions | [docs](https://mem0.ai) |


---
---
---


## Contributing
Contributions are welcome! If you have any ideas, improvements, or new tools to add, please create a new GitHub Issue or submit a pull request.

## Support
- [DM the creator on LinkedIn 💭](https://www.linkedin.com/in/sarthakrastogi/)
- [GitHub Issues 🐛](https://github.com/sarthakrastogi/ai-engineering-reference/issues)

## Star History
You can **star ⭐️ this repo** to stay updated on the latest AI Engineering tools.

[![Star History Chart](https://api.star-history.com/svg?repos=sarthakrastogi/ai-engineering-reference&type=Date)](https://star-history.com/#sarthakrastogi/ai-engineering-reference&Date)

---
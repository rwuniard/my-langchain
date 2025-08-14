# LangChain Learning Projects

This repository contains a progressive series of LangChain learning projects, each building upon previous concepts to demonstrate different aspects of LLM integration and application development.

## Projects Overview

### 1. My First LangChain (`my-first-langchain/`)
**Introduction to LangChain basics**
- Basic LLM integration with OpenAI and Google Gemini
- Simple question-answering setup
- Environment configuration and API key management

### 2. Using Chains (`2.usingChain/`)
**Chain composition and complex workflows**
- LangChain Expression Language (LCEL) usage
- Chain composition and pipeline creation
- Prompt templates and data flow
- PassThrough and generated key demonstrations

### 3. Memory Management (`3.memory_management/`)
**Conversation memory and state persistence**
- Buffer memory, summary memory, and file persistence
- Session management and conversation flow
- Memory optimization strategies
- Built-in vs custom memory implementations

### 4. Context with Embedding (`4.context_with_embedding/`)
**Retrieval-Augmented Generation (RAG) implementation**
- Vector embeddings with OpenAI and Google models
- ChromaDB for persistent vector storage
- Similarity search and context retrieval
- Interactive Q&A with smart duplicate filtering
- Notebook-based similarity score analysis (`scores.ipynb`)

## Getting Started

Each project is self-contained with its own dependencies and setup instructions. Navigate to any project directory and follow the README for specific setup steps.

### Common Requirements
- Python 3.8+
- UV package manager (recommended) or pip
- API keys for OpenAI and/or Google Gemini

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd langchain

# Navigate to any project
cd 4.context_with_embedding

# Install dependencies
uv install

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the project
uv run main.py
```

## Project Dependencies

All projects use modern Python dependency management:
- **UV**: Fast, reliable Python package management
- **pyproject.toml**: Standardized project configuration
- **uv.lock**: Deterministic dependency resolution

## Learning Path

1. **Start with `my-first-langchain/`** - Learn basic LLM integration
2. **Progress to `2.usingChain/`** - Understand chain composition
3. **Explore `3.memory_management/`** - Master state and persistence
4. **Advance to `4.context_with_embedding/`** - Implement full RAG pipeline

Each project builds conceptual knowledge while demonstrating practical implementations of increasingly sophisticated LangChain patterns.

## Key Features Demonstrated

- **Multiple LLM Providers**: OpenAI and Google Gemini integration
- **Modern Tooling**: UV, pyproject.toml, type hints
- **Production Patterns**: Error handling, configuration management
- **Interactive Examples**: Command-line interfaces and Jupyter notebooks
- **Documentation**: Comprehensive READMEs with examples and explanations

## Contributing

This is a learning repository. Each project demonstrates specific LangChain concepts with practical, runnable examples suitable for education and experimentation.
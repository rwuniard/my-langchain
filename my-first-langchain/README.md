# my-first-langchain

A Python project for learning LangChain fundamentals.

## Setup

### Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Initial Project Setup

1. **Install uv** (if not already installed):
   ```bash
   # On macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Or via pip
   pip install uv
   ```

2. **Initialize the project**:
   ```bash
   uv init my-first-langchain
   cd my-first-langchain
   ```

3. **Set up OpenAI API key**:
   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your-api-key-here
   ```
   Or set as environment variable:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```
   Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)

4. **Run the project**:
   ```bash
   uv run main.py
   ```
   This will automatically create a virtual environment and run the application.

## Project Structure

```
my-first-langchain/
├── README.md          # Project documentation
├── main.py           # Main application entry point
├── pyproject.toml    # Project configuration and dependencies
└── .venv/            # Virtual environment (auto-created)
```

## Next Steps

- Add LangChain dependencies to `pyproject.toml`
- Explore LangChain tutorials and examples
- Build your first AI application
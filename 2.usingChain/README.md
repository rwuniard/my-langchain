# 2-usingchain

LangChain project focused on exploring and implementing chains.

## Setup the Environment

1. **Initialize the project**:
   ```bash
   uv init
   ```

2. **Add dependencies**:
   ```bash
   uv add python-dotenv
   uv add langchain langchain-openai langchain-core
   ```

3. **Set up OpenAI API key**:
   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your-api-key-here
   ```

4. **Run the project**:
   
   **Legacy approach** (using deprecated LLMChain):
   ```bash
   uv run main.py
   ```
   
   **Modern approach** (using LCEL):
   ```bash
   uv run main_lcel.py
   ```

## Features

This project demonstrates:
- Basic LangChain chain implementation (legacy and modern approaches)
- Prompt templates with variables
- Integration with OpenAI models
- Environment variable management
- LCEL (LangChain Expression Language) syntax

## Files

- `main.py` - Legacy implementation using `LLMChain` (deprecated)
- `main_lcel.py` - Modern implementation using LCEL syntax (`prompt | llm`)

## Notes

- `LLMChain` is deprecated in LangChain 0.3+ and will be removed in 1.0
- The modern approach uses LCEL syntax: `prompt | llm`
- LCEL provides better performance and composability 
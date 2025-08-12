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
   ```bash
   uv run main.py
   ```

## Features

This project demonstrates:
- Basic LangChain chain implementation
- Prompt templates with variables
- Integration with OpenAI models
- Environment variable management

## Notes

- The current implementation uses `LLMChain` which is deprecated in LangChain 0.3+
- For modern LangChain, consider using the LCEL (LangChain Expression Language) syntax: `prompt | llm` 
# 3-memory-management

LangChain project focused on exploring and implementing memory management patterns.

## Setup the Environment

1. **Initialize the project**:
   ```bash
   uv init
   ```

2. **Add dependencies**:
   ```bash
   uv add langchain langchain-openai python-dotenv
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
- Memory management patterns in LangChain
- Conversation history handling
- State persistence across interactions
- Memory optimization techniques

## Files

- `main.py` - Main implementation file (to be created)

## Notes

- Environment variables are loaded from `.env` file
- OpenAI API key is required for functionality
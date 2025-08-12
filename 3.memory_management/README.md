# 3-memory-management

LangChain project focused on exploring and implementing memory management patterns.

## Setup the Environment

1. **Initialize the project**:
   ```bash
   uv init
   ```

2. **Add dependencies**:
   ```bash
   uv add langchain langchain-openai langchain-community python-dotenv
   ```

3. **Set up OpenAI API key**:
   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your-api-key-here
   ```

4. **Run the project**:
   
   **Interactive conversation** (with memory):
   ```bash
   uv run main.py
   ```
   
   **Memory test** (automated demonstration):
   ```bash
   uv run test_memory.py
   ```

## Features

This project demonstrates:
- **Modern LCEL Memory Management**: Using `RunnableWithMessageHistory` with `ChatMessageHistory`
- **Conversation Context**: AI remembers previous messages within a session
- **Session-based Storage**: Multiple conversation sessions with unique IDs
- **Interactive Chat**: Real-time conversation with persistent memory
- **Memory Testing**: Automated verification of memory functionality

## Files

- `main.py` - Interactive conversation application with memory using modern LCEL approach
- `test_memory.py` - Automated test demonstrating memory functionality across messages

## Memory Implementation

This project uses the **modern LCEL approach** for memory management:

### Key Components

- **`ChatMessageHistory`**: Stores conversation messages in memory
- **`RunnableWithMessageHistory`**: Wraps the chain to automatically handle message history
- **`MessagesPlaceholder`**: Injects conversation history into the prompt template
- **Session Management**: Each conversation gets a unique session ID for isolated memory

### Memory Flow

1. **User Input** → Chain receives message with session ID
2. **History Retrieval** → `get_session_history()` loads previous messages
3. **Prompt Construction** → History injected via `MessagesPlaceholder`
4. **AI Response** → Generated based on current input + conversation context
5. **Memory Update** → Both user message and AI response saved to session history

### Example Usage

```python
# First message
chain_with_memory.invoke(
    {"content": "My name is Alice"},
    config={"configurable": {"session_id": "user123"}}
)

# Second message - AI remembers Alice's name
chain_with_memory.invoke(
    {"content": "What is my name?"},
    config={"configurable": {"session_id": "user123"}}
)
```

## Notes

- Environment variables are loaded from `.env` file
- OpenAI API key is required for functionality
- Memory is stored in-memory (use database for production)
- Each session maintains isolated conversation history
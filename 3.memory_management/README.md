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
   
   **Interactive conversation** (in-memory):
   ```bash
   uv run main.py
   ```
   
   **Interactive conversation** (with file persistence):
   ```bash
   uv run main_with_file_persistence.py
   ```
   
   **Memory test** (automated demonstration):
   ```bash
   uv run test_memory.py
   ```
   
   **Educational demos** (understanding how memory works):
   ```bash
   # Show session flow and history management
   uv run demo_session_flow.py
   
   # Demonstrate why MessagesPlaceholder is essential
   uv run demo_placeholder_detailed.py
   
   # Show the hidden connection between components
   uv run demo_hidden_connection.py
   
   # Explain config parameter and session isolation
   uv run demo_config_explained.py
   
   # File persistence demos
   uv run demo_file_persistence_comparison.py
   uv run demo_file_autosave_internals.py
   uv run demo_when_saving_happens_in_chain.py
   ```

## Features

This project demonstrates:
- **Modern LCEL Memory Management**: Using `RunnableWithMessageHistory` with `ChatMessageHistory`
- **Conversation Context**: AI remembers previous messages within a session
- **Session-based Storage**: Multiple conversation sessions with unique IDs
- **Interactive Chat**: Real-time conversation with persistent memory
- **Memory Testing**: Automated verification of memory functionality
- **File Persistence**: Automatic conversation saving with `FileChatMessageHistory`
- **Auto-save Mechanism**: Understanding how conversations persist automatically

## Files

### Core Implementation
- `main.py` - Interactive conversation application with in-memory storage
- `main_with_file_persistence.py` - Interactive conversation with automatic file persistence
- `test_memory.py` - Automated test demonstrating memory functionality across messages

### Educational Demonstrations

**Memory System Internals**:
- `demo_session_flow.py` - Shows how `get_session_history()` manages session isolation
- `demo_placeholder_detailed.py` - Demonstrates why `MessagesPlaceholder` is essential
- `demo_hidden_connection.py` - Reveals the hidden connection between components
- `demo_config_explained.py` - Explains the config parameter and session management

**File Persistence Deep Dive**:
- `demo_file_persistence_comparison.py` - Old vs new approach to file persistence
- `demo_file_autosave_internals.py` - How `FileChatMessageHistory` auto-saves
- `demo_when_saving_happens_in_chain.py` - When saves occur during chain execution

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

## Key Concepts Explained

### Config Parameter
The `config` parameter tells `RunnableWithMessageHistory` which session to use:

```python
config={"configurable": {"session_id": "user_123"}}
```

- **`"configurable"`**: Required top-level key (LangChain convention)
- **`"session_id"`**: Identifies which conversation history to use
- **Session isolation**: Each session ID maintains separate conversation history

### MessagesPlaceholder Connection
The memory system works through parameter matching:

```python
# Prompt template expects "history" key
MessagesPlaceholder(variable_name="history")

# RunnableWithMessageHistory provides "history" key  
history_messages_key="history"
```

This string matching (`"history"`) is what connects the memory system to the prompt template.

### Session Management Flow
1. **Extract session ID** from config parameter
2. **Get history** using `get_session_history(session_id)`
3. **Inject messages** into prompt via `MessagesPlaceholder`
4. **Execute chain** with full conversation context
5. **Save response** automatically to session history

## Production Considerations

### Session IDs
In production, use meaningful session identifiers:
```python
# User-based sessions
config={"configurable": {"session_id": f"user_{user_id}"}}

# Channel-based sessions (Discord/Slack)
config={"configurable": {"session_id": f"channel_{channel_id}"}}

# Room-based sessions (chat rooms)
config={"configurable": {"session_id": f"room_{room_id}_{date}"}}
```

### Persistent Storage
Replace in-memory storage with databases:
```python
# Redis example
from langchain_community.chat_message_histories import RedisChatMessageHistory

def get_session_history(session_id: str):
    return RedisChatMessageHistory(
        session_id=session_id,
        url="redis://localhost:6379"
    )
```

## File Persistence

### Modern File Persistence with FileChatMessageHistory

Replace the `get_session_history` function to automatically save conversations to files:

```python
from langchain_community.chat_message_histories import FileChatMessageHistory

def get_session_history(session_id: str) -> FileChatMessageHistory:
    file_path = f"conversations/conversation_{session_id}.json"
    return FileChatMessageHistory(file_path)  # That's it!
```

### How Auto-saving Works

`FileChatMessageHistory` automatically saves every message immediately:

1. **On Message Add**: Every `add_message()` call triggers an immediate file save
2. **Crash Safe**: Messages saved before any failure can occur
3. **Auto Load**: Existing conversations loaded automatically on startup
4. **No Manual Operations**: No save/load methods needed

### Save Timing in Chains

Each `chain_with_memory.invoke()` triggers **two automatic saves**:

```
User calls invoke()
   ↓
📝 SAVE #1: User message saved to file
   ↓  
🤖 LLM processes request
   ↓
📝 SAVE #2: AI response saved to file
   ↓
Response returned
```

### File Organization

```
conversations/
├── conversation_alice.json     # Alice's chat history
├── conversation_bob.json       # Bob's chat history
├── conversation_room_123.json  # Room-based chat
└── conversation_default.json   # Default session
```

## Notes

- Environment variables are loaded from `.env` file
- OpenAI API key is required for functionality
- In-memory storage vs file persistence - choose based on needs
- Each session maintains isolated conversation history
- `conversations/` folder excluded from git (contains private data)
- Demo files provide deep understanding of memory system internals
- File persistence works automatically - no manual save/load needed
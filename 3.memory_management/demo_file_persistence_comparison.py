"""
Comparison: Old vs New way of file persistence
"""
from langchain_community.chat_message_histories import FileChatMessageHistory, ChatMessageHistory
import os

def demo_old_vs_new_approach():
    print("=== Old vs New File Persistence Approach ===\n")
    
    print("🔺 OLD WAY (Deprecated):")
    print("```python")
    print("from langchain.memory import ConversationBufferMemory")
    print("from langchain_community.chat_message_histories import FileChatMessageHistory")
    print("")
    print("# Create file-based message history")
    print("message_history = FileChatMessageHistory('conversation.json')")
    print("")
    print("# Add to ConversationBufferMemory")
    print("memory = ConversationBufferMemory(")
    print("    chat_memory=message_history,")
    print("    memory_key='chat_history',")
    print("    return_messages=True")
    print(")")
    print("")
    print("# Use with LLMChain (deprecated)")
    print("conversation = ConversationChain(llm=llm, memory=memory)")
    print("```")
    
    print("\n" + "="*60)
    
    print("\n✅ NEW WAY (Modern LCEL):")
    print("```python")
    print("from langchain_community.chat_message_histories import FileChatMessageHistory")
    print("from langchain_core.runnables import RunnableWithMessageHistory")
    print("")
    print("def get_session_history(session_id: str) -> FileChatMessageHistory:")
    print("    file_path = f'conversations/conversation_{session_id}.json'")
    print("    return FileChatMessageHistory(file_path)  # That's it!")
    print("")
    print("# Use with RunnableWithMessageHistory")
    print("chain_with_memory = RunnableWithMessageHistory(")
    print("    chain,")
    print("    get_session_history,  # Function returns FileChatMessageHistory")
    print("    input_messages_key='content',")
    print("    history_messages_key='history'")
    print(")")
    print("```")

def demo_file_persistence_behavior():
    print("\n\n=== How File Persistence Works ===\n")
    
    # Clean up any existing demo files
    demo_file = "demo_conversation.json"
    if os.path.exists(demo_file):
        os.remove(demo_file)
    
    print("📝 Creating a new conversation...")
    
    # Create FileChatMessageHistory - this creates the file
    history = FileChatMessageHistory(demo_file)
    
    print(f"📁 File exists: {os.path.exists(demo_file)}")
    print(f"📊 Messages in history: {len(history.messages)}")
    
    # Add some messages
    print("\n💬 Adding messages...")
    history.add_user_message("Hello! My name is Alice.")
    history.add_ai_message("Hi Alice! Nice to meet you.")
    history.add_user_message("I love programming in Python.")
    history.add_ai_message("That's great! Python is an excellent language.")
    
    print(f"📊 Messages in history: {len(history.messages)}")
    print(f"📁 File exists: {os.path.exists(demo_file)}")
    
    # Show file contents
    if os.path.exists(demo_file):
        with open(demo_file, 'r') as f:
            content = f.read()
        print(f"\n📄 File contents preview:")
        print(f"   File size: {len(content)} characters")
        print(f"   Contains: {content[:100]}..." if len(content) > 100 else content)
    
    print("\n🔄 Loading conversation from file...")
    
    # Create a NEW FileChatMessageHistory pointing to the same file
    # This should automatically load the existing messages
    history2 = FileChatMessageHistory(demo_file)
    
    print(f"📊 Messages loaded: {len(history2.messages)}")
    print("🎯 Loaded messages:")
    for i, msg in enumerate(history2.messages):
        print(f"   {i+1}. {msg.type}: {msg.content}")
    
    # Clean up
    if os.path.exists(demo_file):
        os.remove(demo_file)
        print(f"\n🧹 Cleaned up demo file: {demo_file}")

def show_key_benefits():
    print("\n\n=== Key Benefits of Modern Approach ===\n")
    
    print("✅ **Simpler Setup**:")
    print("   - No need to manage ConversationBufferMemory")
    print("   - Direct return of FileChatMessageHistory from function")
    print("   - Automatic loading/saving handled by FileChatMessageHistory")
    
    print("\n✅ **Session-Based Storage**:")
    print("   - Each session ID gets its own file")
    print("   - Easy to organize conversations by user/channel/room")
    print("   - Natural multi-tenant support")
    
    print("\n✅ **Automatic Persistence**:")
    print("   - Messages saved immediately when added")
    print("   - No manual save/load operations needed")
    print("   - Conversation resumes automatically on restart")
    
    print("\n✅ **Flexible File Organization**:")
    print("   conversations/")
    print("   ├── conversation_alice.json")
    print("   ├── conversation_bob.json")
    print("   ├── conversation_channel_123.json")
    print("   └── conversation_room_456.json")

if __name__ == "__main__":
    demo_old_vs_new_approach()
    demo_file_persistence_behavior()
    show_key_benefits()
"""
Detailed explanation of the config parameter in RunnableWithMessageHistory
"""
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

# Store for session histories
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """This function receives the session_id from the config"""
    print(f"🔍 get_session_history called with session_id: '{session_id}'")
    if session_id not in store:
        print(f"✨ Creating new history for session: {session_id}")
        store[session_id] = ChatMessageHistory()
    else:
        print(f"📚 Using existing history for session: {session_id} ({len(store[session_id].messages)} messages)")
    return store[session_id]

def demo_config_parameter():
    print("=== Understanding the config parameter ===\n")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{content}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    chain_with_memory = RunnableWithMessageHistory(
        chain,
        get_session_history,  # ← This function will receive the session_id
        input_messages_key="content",
        history_messages_key="history"
    )
    
    print("🎯 The config structure:")
    print('   config={"configurable": {"session_id": "some_value"}}')
    print("   │")
    print("   ├── 'configurable' is a required key")
    print("   └── 'session_id' is passed to get_session_history()")
    
    print("\n" + "="*60)
    
    # Demo 1: Different sessions
    print("\n📱 Demo 1: Multiple independent sessions")
    
    print("\n   Session 'alice' - First message:")
    result1 = chain_with_memory.invoke(
        {"content": "My name is Alice and I love cats"},
        config={"configurable": {"session_id": "alice"}}
    )
    print(f"   Response: {result1}")
    
    print("\n   Session 'bob' - First message:")
    result2 = chain_with_memory.invoke(
        {"content": "My name is Bob and I love dogs"},
        config={"configurable": {"session_id": "bob"}}
    )
    print(f"   Response: {result2}")
    
    print("\n   Session 'alice' - Second message (remembers cats):")
    result3 = chain_with_memory.invoke(
        {"content": "What do I love?"},
        config={"configurable": {"session_id": "alice"}}
    )
    print(f"   Response: {result3}")
    
    print("\n   Session 'bob' - Second message (remembers dogs):")
    result4 = chain_with_memory.invoke(
        {"content": "What do I love?"},
        config={"configurable": {"session_id": "bob"}}
    )
    print(f"   Response: {result4}")

def demo_config_flow():
    print("\n\n=== How config flows through the system ===\n")
    
    print("🔄 The complete flow:")
    print("   1. You call: chain_with_memory.invoke({...}, config={...})")
    print("   2. RunnableWithMessageHistory extracts: config['configurable']['session_id']")
    print("   3. Calls: get_session_history(session_id)")
    print("   4. Gets: ChatMessageHistory object for that session")
    print("   5. Extracts: history.messages")
    print("   6. Builds: {'content': input, 'history': messages}")
    print("   7. Calls: prompt.format_messages(**data)")
    print("   8. Sends to LLM with full conversation context")
    print("   9. Saves both user input and AI response to the session history")

def demo_what_if_no_config():
    print("\n\n=== What happens without config? ===\n")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{content}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    chain_with_memory = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="content",
        history_messages_key="history"
    )
    
    try:
        # This will fail - no config provided
        result = chain_with_memory.invoke({"content": "Hello"})
        print(f"Unexpected success: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Config is REQUIRED for RunnableWithMessageHistory!")

def demo_session_isolation():
    print("\n\n=== Session isolation demonstration ===\n")
    
    # Show current store state
    print("📊 Current store contents:")
    for session_id, history in store.items():
        print(f"   Session '{session_id}': {len(history.messages)} messages")
        for i, msg in enumerate(history.messages):
            print(f"     {i+1}. {msg.type}: {msg.content[:50]}{'...' if len(msg.content) > 50 else ''}")

if __name__ == "__main__":
    demo_config_parameter()
    demo_config_flow()
    demo_what_if_no_config()
    demo_session_isolation()
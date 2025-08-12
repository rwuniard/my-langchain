"""
Detailed demonstration showing exactly what MessagesPlaceholder does
"""
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

def demo_prompt_construction():
    print("=== How MessagesPlaceholder Works ===\n")
    
    # Create a simple history with some messages
    history = ChatMessageHistory()
    history.add_user_message("My name is Alice")
    history.add_ai_message("Nice to meet you, Alice!")
    history.add_user_message("I like programming")
    history.add_ai_message("That's great! What languages do you enjoy?")
    
    print("📚 Sample conversation history:")
    for i, msg in enumerate(history.messages):
        print(f"   {i+1}. {msg.type}: {msg.content}")
    
    print("\n" + "="*50)
    
    # Prompt WITHOUT MessagesPlaceholder
    print("\n🚫 Prompt WITHOUT MessagesPlaceholder:")
    prompt_without = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{content}")
    ])
    
    # This is what gets sent to the LLM (NO HISTORY!)
    formatted_without = prompt_without.format_messages(content="What's my name?")
    print("   Messages sent to LLM:")
    for i, msg in enumerate(formatted_without):
        print(f"     {i+1}. {msg.type}: {msg.content}")
    
    print("\n" + "="*50)
    
    # Prompt WITH MessagesPlaceholder  
    print("\n✅ Prompt WITH MessagesPlaceholder:")
    prompt_with = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history_chat"),
        ("human", "{content}")
    ])
    
    # This is what gets sent to the LLM (WITH HISTORY!)
    formatted_with = prompt_with.format_messages(
        content="What's my name?",
        history_chat=history.messages  # The placeholder gets replaced with actual messages
    )
    print("   Messages sent to LLM:")
    for i, msg in enumerate(formatted_with):
        print(f"     {i+1}. {msg.type}: {msg.content}")

def demo_memory_mapping():
    print("\n\n=== How RunnableWithMessageHistory Maps History ===\n")
    
    # Store
    store = {}
    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    
    # Setup
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),  # This variable name matters!
        ("human", "{content}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    chain_with_memory = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="content",
        history_messages_key="history"  # This MUST match MessagesPlaceholder variable_name!
    )
    
    print("🔗 The mapping process:")
    print("   1. RunnableWithMessageHistory gets session history")
    print("   2. Takes messages from ChatMessageHistory")
    print("   3. Maps them to 'history' key (history_messages_key)")
    print("   4. MessagesPlaceholder(variable_name='history') receives them")
    print("   5. Injects messages into the prompt at the placeholder location")
    
    # Show what happens step by step
    print("\n📝 Step-by-step execution:")
    
    # First message
    print("\n   First message: 'My name is Bob'")
    result1 = chain_with_memory.invoke(
        {"content": "My name is Bob"},
        config={"configurable": {"session_id": "demo"}}
    )
    print(f"   Response: {result1}")
    print(f"   Messages in history: {len(store['demo'].messages)}")
    
    # Second message - now history matters
    print("\n   Second message: 'What's my name?'")
    print("   History will be injected before this message!")
    result2 = chain_with_memory.invoke(
        {"content": "What's my name?"},
        config={"configurable": {"session_id": "demo"}}
    )
    print(f"   Response: {result2}")
    print(f"   Messages in history: {len(store['demo'].messages)}")

if __name__ == "__main__":
    demo_prompt_construction()
    demo_memory_mapping()
"""
Demonstration showing the hidden connection in main.py
"""
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

# Store
store = {}
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def show_hidden_connection():
    print("=== The Hidden Connection in main.py ===\n")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # This is EXACTLY what's in main.py
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),  # ← Expects "history" key
        ("human", "{content}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    chain_with_memory = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="content",
        history_messages_key="history"  # ← Will provide "history" key
    )
    
    print("🔗 The connection:")
    print('   MessagesPlaceholder(variable_name="history")')
    print('   ↕️ MATCHES ↕️')
    print('   history_messages_key="history"')
    
    print("\n📋 What RunnableWithMessageHistory does internally:")
    print("   1. Gets session history from get_session_history()")
    print("   2. Extracts .messages from ChatMessageHistory")
    print('   3. Creates: {"history": messages, "content": user_input}')
    print("   4. Passes this dict to the prompt")
    print("   5. MessagesPlaceholder receives messages via 'history' key")
    
    # Let's simulate what happens step by step
    print("\n🎭 Manual simulation of what happens:")
    
    # Step 1: Add some history manually to see the effect
    session_history = get_session_history("demo")
    session_history.add_user_message("My name is Charlie")
    session_history.add_ai_message("Nice to meet you, Charlie!")
    
    # Step 2: Show what gets passed to the prompt
    print("\n   What gets passed to prompt.format_messages():")
    data_passed_to_prompt = {
        "content": "What's my name?",
        "history": session_history.messages  # This is what RunnableWithMessageHistory does!
    }
    
    for key, value in data_passed_to_prompt.items():
        if key == "history":
            print(f'     {key}: [{len(value)} messages]')
            for i, msg in enumerate(value):
                print(f'       {i+1}. {msg.type}: {msg.content}')
        else:
            print(f'     {key}: {value}')
    
    # Step 3: Show the final formatted prompt
    print("\n   Final prompt sent to LLM:")
    formatted_prompt = prompt.format_messages(**data_passed_to_prompt)
    for i, msg in enumerate(formatted_prompt):
        print(f'     {i+1}. {msg.type}: {msg.content}')

def show_what_if_names_dont_match():
    print("\n\n=== What if the names DON'T match? ===\n")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Mismatched names - this will break!
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),  # ← Expects "chat_history"
        ("human", "{content}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        chain_with_memory = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="content",
            history_messages_key="history"  # ← Provides "history" (MISMATCH!)
        )
        
        result = chain_with_memory.invoke(
            {"content": "Test message"},
            config={"configurable": {"session_id": "test"}}
        )
        print(f"Unexpected success: {result}")
        
    except Exception as e:
        print(f"❌ Error (as expected): {e}")
        print("   The names must match for the connection to work!")

if __name__ == "__main__":
    show_hidden_connection()
    show_what_if_names_dont_match()
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
import os

load_dotenv()

# Store for session histories - in production, this would be a database
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Get or create a chat message history for the given session ID"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def test_memory():
    print("🧪 Testing memory management...")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create a prompt template that includes a placeholder for chat history
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{content}")
    ])

    # Create a chain
    chain = prompt | llm | StrOutputParser()

    # Create a chain that will use the memory
    chain_with_memory = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="content",
        history_messages_key="history"
    )

    # Test conversation with memory
    session_config = {"configurable": {"session_id": "test_session"}}
    
    print("💬 First message:")
    response1 = chain_with_memory.invoke(
        {"content": "My name is Alice. Remember this!"},
        config=session_config
    )
    print(f"AI: {response1}")
    
    print("\n💬 Second message (testing memory):")
    response2 = chain_with_memory.invoke(
        {"content": "What is my name?"},
        config=session_config
    )
    print(f"AI: {response2}")
    
    print("\n✅ Memory test completed!")
    print(f"📊 Messages in memory: {len(store['test_session'].messages)}")

if __name__ == "__main__":
    test_memory()
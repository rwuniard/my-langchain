"""
Show exactly when FileChatMessageHistory saves during RunnableWithMessageHistory execution
"""
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import FileChatMessageHistory
from dotenv import load_dotenv
import os

load_dotenv()

class VerboseFileChatMessageHistory(FileChatMessageHistory):
    """FileChatMessageHistory with verbose logging to show when saves happen"""
    
    def add_message(self, message):
        print(f"📝 SAVING: Adding {message.type} message to file...")
        super().add_message(message)
        print(f"💾 SAVED: File updated with {len(self.messages)} total messages")

def get_session_history(session_id: str) -> VerboseFileChatMessageHistory:
    """Get session history with verbose logging"""
    file_path = f"demo_chain_{session_id}.json"
    print(f"🔍 Getting session history for: {session_id}")
    print(f"📁 File path: {file_path}")
    
    history = VerboseFileChatMessageHistory(file_path)
    print(f"📊 Loaded {len(history.messages)} existing messages")
    return history

def demo_save_timing():
    print("=== When Does Saving Happen in RunnableWithMessageHistory? ===\n")
    
    # Clean up any existing demo files
    demo_file = "demo_chain_test.json"
    if os.path.exists(demo_file):
        os.remove(demo_file)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Keep responses brief."),
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
    
    print("🚀 Invoking chain with first message...")
    print("👀 Watch when the saves happen:\n")
    
    result = chain_with_memory.invoke(
        {"content": "My name is Bob"},
        config={"configurable": {"session_id": "test"}}
    )
    
    print(f"\n🤖 AI Response: {result}")
    
    print("\n" + "="*60)
    
    print("\n🔄 Invoking chain with second message (should remember Bob)...")
    print("👀 Watch the saves again:\n")
    
    result2 = chain_with_memory.invoke(
        {"content": "What's my name?"},
        config={"configurable": {"session_id": "test"}}
    )
    
    print(f"\n🤖 AI Response: {result2}")
    
    # Show final file contents
    print("\n📄 Final file contents:")
    if os.path.exists(demo_file):
        history = FileChatMessageHistory(demo_file)
        print(f"   Total messages saved: {len(history.messages)}")
        for i, msg in enumerate(history.messages):
            print(f"     {i+1}. {msg.type}: {msg.content}")
    
    # Clean up
    if os.path.exists(demo_file):
        os.remove(demo_file)

def explain_save_sequence():
    print("\n\n=== The Complete Save Sequence ===\n")
    
    print("🔄 When you call chain_with_memory.invoke():")
    print("   1. RunnableWithMessageHistory gets session history")
    print("   2. Existing messages loaded from file (if any)")
    print("   3. User message added to prompt")
    print("   4. 📝 FIRST SAVE: User message saved to file")
    print("   5. LLM generates response")
    print("   6. 📝 SECOND SAVE: AI response saved to file")
    print("   7. Response returned to you")
    
    print("\n💡 Key insight:")
    print("   Every invoke() triggers TWO saves:")
    print("   - Save #1: Your input message")
    print("   - Save #2: AI's response message")
    
    print("\n🛡️ Benefits:")
    print("   ✅ No data loss if program crashes between user input and AI response")
    print("   ✅ Conversation always up-to-date on disk")
    print("   ✅ Multiple program instances can share conversation files")
    print("   ✅ No manual save/load operations required")

if __name__ == "__main__":
    demo_save_timing()
    explain_save_sequence()
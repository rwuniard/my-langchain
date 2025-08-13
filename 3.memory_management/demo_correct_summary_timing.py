"""
Demonstration of CORRECT summarization timing in LCEL
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

def verbose_get_session_history(session_id: str) -> ChatMessageHistory:
    """Session history with verbose logging to show when summarization happens"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
        print(f"✨ Created new history for session: {session_id}")
    
    history = store[session_id]
    print(f"🔍 Called get_session_history - current messages: {len(history.messages)}")
    
    # Check if we need to summarize BEFORE adding new messages
    if len(history.messages) > 10:  # This is the key check
        print(f"📝 TRIGGERING SUMMARY: {len(history.messages)} messages > 10 limit")
        
        # Create summarization chain
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize this conversation in 2-3 sentences:"),
            ("human", "{conversation}")
        ])
        summary_chain = summary_prompt | llm | StrOutputParser()
        
        # Get messages to summarize (all but last 4)
        messages_to_summarize = history.messages[:-4]
        recent_messages = history.messages[-4:]
        
        # Convert to text for summarization
        conversation_text = "\n".join([
            f"{msg.type}: {msg.content}" for msg in messages_to_summarize
        ])
        
        # Create summary
        summary = summary_chain.invoke({"conversation": conversation_text})
        print(f"📄 Created summary: {summary[:80]}...")
        
        # Rebuild history with summary + recent messages
        new_history = ChatMessageHistory()
        new_history.add_ai_message(f"[SUMMARY] {summary}")
        
        for msg in recent_messages:
            new_history.add_message(msg)
        
        # Replace the history
        store[session_id] = new_history
        history = new_history
        
        print(f"📊 Reduced from {len(messages_to_summarize) + len(recent_messages)} to {len(history.messages)} messages")
    
    return history

def demo_correct_timing():
    """Show the CORRECT timing of when summarization happens"""
    print("=== CORRECT Summarization Timing Demo ===\n")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Be brief."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{content}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    chain_with_summary = RunnableWithMessageHistory(
        chain,
        verbose_get_session_history,
        input_messages_key="content",
        history_messages_key="history"
    )
    
    # Messages to simulate conversation
    messages = [
        "Hi, I'm Alice",
        "I'm 25 years old",
        "I work as a software engineer",
        "I live in Seattle",
        "I like Python programming",
        "I work at a tech startup",
        "We build web applications",
        "I'm interested in AI",
        "Can you help me learn ML?",
        "What should I study first?",
        "How long does it take to learn?",  # This will be message #11
    ]
    
    session_id = "timing_demo"
    
    print("🎯 Watch carefully - summarization happens BEFORE adding new messages")
    print("📊 Each invoke() adds 2 messages: user input + AI response")
    print("🔢 Summarization triggers when EXISTING messages > 10\n")
    
    for i, message in enumerate(messages):
        print(f"🚀 INVOKE #{i+1}: '{message}'")
        
        # This is where summarization happens (inside the invoke)
        result = chain_with_summary.invoke(
            {"content": message},
            config={"configurable": {"session_id": session_id}}
        )
        
        print(f"✅ Response: {result[:60]}...")
        print(f"📊 Total messages after invoke: {len(store[session_id].messages)}\n")
        
        # Show when summarization ACTUALLY happens
        if i == 5:  # After 6 user messages = 12 total messages
            print("💡 EXPLANATION: Summarization happened because:")
            print("   • Before this invoke: 10 messages existed")
            print("   • Check: 10 > 10? No, so no summary yet")
            print("   • After this invoke: 12 messages total\n")
        elif i == 6:  # After 7 user messages would be too late
            print("💡 EXPLANATION: Summarization should have happened because:")
            print("   • Before this invoke: 12 messages existed") 
            print("   • Check: 12 > 10? Yes, so summary triggered!")
            print("   • Messages reduced, then new messages added\n")
            break

def show_the_bug():
    print("\n\n=== The Bug in Original Code ===\n")
    
    print("❌ **Wrong assumption in original code:**")
    print("   if i == 10:  # After 11 messages")
    print("       print('Summary should have been created here!')")
    print("")
    print("🔍 **Why this is wrong:**")
    print("   1. Each invoke() adds 2 messages (user + AI)")
    print("   2. After 6 invokes = 12 messages total")
    print("   3. Summarization happens DURING invoke, not after")
    print("   4. The check is: len(messages) > 10 BEFORE adding new ones")
    print("")
    print("✅ **Correct timing:**")
    print("   • After invoke #5: 10 messages (5 user + 5 AI)")
    print("   • Invoke #6 starts: checks 10 > 10? No")
    print("   • Invoke #6 completes: 12 messages total")
    print("   • Invoke #7 starts: checks 12 > 10? YES → summarize!")

if __name__ == "__main__":
    demo_correct_timing()
    show_the_bug()
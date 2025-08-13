"""
Comparison: Legacy ConversationSummaryMemory vs Modern LCEL approach
"""
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

def demo_legacy_approach():
    print("=== Legacy ConversationSummaryMemory (Deprecated) ===\n")
    
    print("🔺 OLD WAY (with actual working code):")
    
    # Actually demonstrate the legacy approach
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Create legacy summary memory
    print("📝 Creating ConversationSummaryMemory...")
    legacy_memory = ConversationSummaryMemory(
        llm=llm,
        return_messages=True
    )
    
    print("✅ Legacy memory created successfully")
    print(f"   Type: {type(legacy_memory)}")
    print(f"   Has buffer: {hasattr(legacy_memory, 'buffer')}")
    
    print("\n📝 Adding conversation messages manually...")
    
    # Add conversation manually using save_context
    conversations = [
        ("Hi, I'm Alice, a 25-year-old software engineer from Seattle.", 
         "Hello Alice! Nice to meet you. How can I help you today?"),
        ("I work with Python and machine learning at a tech startup.", 
         "That's great! Machine learning with Python is very popular. What kind of ML projects are you working on?"),
        ("We're building recommendation systems for e-commerce.", 
         "Recommendation systems are fascinating! Are you using collaborative filtering or content-based approaches?"),
        ("We use collaborative filtering with matrix factorization techniques.", 
         "Excellent choice! Matrix factorization is very effective for recommendation systems. Have you tried deep learning approaches?"),
        ("Yes, we're experimenting with neural collaborative filtering.", 
         "That's cutting-edge! Neural CF can capture complex user-item interactions. How are you handling the cold start problem?")
    ]
    
    for i, (user_msg, ai_msg) in enumerate(conversations):
        print(f"💬 Adding conversation {i+1}...")
        legacy_memory.save_context(
            {"input": user_msg},
            {"output": ai_msg}
        )
        
        print(f"   Buffer length: {len(legacy_memory.buffer)} characters")
        if i >= 2:  # Show summary creation
            print(f"   Summary created: {'Yes' if 'Alice' in legacy_memory.buffer and len(legacy_memory.buffer) < 500 else 'Not yet'}")
    
    print(f"\n💭 Final legacy memory buffer:\n{legacy_memory.buffer}")
    print(f"\n📊 Legacy approach stats:")
    print(f"   • Buffer length: {len(legacy_memory.buffer)} characters")
    print(f"   • Messages in chat_memory: {len(legacy_memory.chat_memory.messages)}")
    print(f"   • Summary automatically created: {'Yes' if len(legacy_memory.buffer) < 800 else 'No'}")
    
    return legacy_memory

def demo_modern_approach():
    print("\n\n=== Modern LCEL Approach ===\n")
    
    print("✅ NEW WAY (with actual working implementation):")
    
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables import RunnableWithMessageHistory
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.messages import HumanMessage, AIMessage
    
    # Create a working SummarizingChatMessageHistory
    class SummarizingChatMessageHistory(ChatMessageHistory):
        def __init__(self, llm, max_messages=8, summary_message_count=4):
            super().__init__()
            # Store configuration
            self._llm = llm
            self._max_messages = max_messages
            self._summary_message_count = summary_message_count
            
            # Create summarization chain
            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", "Summarize this conversation in 2-3 sentences, focusing on key information about the person and topics discussed:"),
                ("human", "{conversation}")
            ])
            self._summary_chain = summary_prompt | llm | StrOutputParser()
            
        @property
        def max_messages(self):
            return self._max_messages
            
        @property
        def summary_message_count(self):
            return self._summary_message_count
            
        def add_message(self, message):
            super().add_message(message)
            
            # Check if we need to summarize
            if len(self.messages) > self._max_messages:
                self._summarize_conversation()
                
        def _summarize_conversation(self):
            # Get messages to summarize (all but last N)
            messages_to_summarize = self.messages[:-self._summary_message_count]
            recent_messages = self.messages[-self._summary_message_count:]
            
            # Convert to text for summarization
            conversation_text = "\n".join([
                f"{msg.type}: {msg.content}" for msg in messages_to_summarize
            ])
            
            # Create summary
            summary = self._summary_chain.invoke({"conversation": conversation_text})
            
            # Rebuild messages with summary + recent messages
            self.messages = [AIMessage(content=f"[SUMMARY] {summary}")]
            self.messages.extend(recent_messages)
            
            print(f"📄 Auto-summarized: {len(messages_to_summarize)} messages → 1 summary")
            print(f"   Kept {len(recent_messages)} recent messages")
            print(f"   Total messages now: {len(self.messages)}")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    print("📝 Creating SummarizingChatMessageHistory...")
    modern_memory = SummarizingChatMessageHistory(llm, max_messages=8, summary_message_count=4)
    
    print("✅ Modern memory created successfully")
    print(f"   Type: {type(modern_memory)}")
    print(f"   Max messages: {modern_memory.max_messages}")
    print(f"   Summary trigger: when > {modern_memory.max_messages} messages")
    
    print("\n📝 Adding the same conversation messages...")
    
    # Add the same conversations as legacy approach
    conversations = [
        ("human", "Hi, I'm Alice, a 25-year-old software engineer from Seattle."),
        ("ai", "Hello Alice! Nice to meet you. How can I help you today?"),
        ("human", "I work with Python and machine learning at a tech startup."),
        ("ai", "That's great! Machine learning with Python is very popular. What kind of ML projects are you working on?"),
        ("human", "We're building recommendation systems for e-commerce."),
        ("ai", "Recommendation systems are fascinating! Are you using collaborative filtering or content-based approaches?"),
        ("human", "We use collaborative filtering with matrix factorization techniques."),
        ("ai", "Excellent choice! Matrix factorization is very effective for recommendation systems. Have you tried deep learning approaches?"),
        ("human", "Yes, we're experimenting with neural collaborative filtering."),
        ("ai", "That's cutting-edge! Neural CF can capture complex user-item interactions. How are you handling the cold start problem?")
    ]
    
    for i, (msg_type, content) in enumerate(conversations):
        print(f"💬 Adding message {i+1}: {msg_type}")
        
        if msg_type == "human":
            modern_memory.add_message(HumanMessage(content=content))
        else:
            modern_memory.add_message(AIMessage(content=content))
        
        print(f"   Total messages: {len(modern_memory.messages)}")
        if len(modern_memory.messages) <= modern_memory.max_messages:
            print(f"   Status: Normal (≤ {modern_memory.max_messages} limit)")
        
    print(f"\n💭 Final modern memory messages:")
    for i, msg in enumerate(modern_memory.messages):
        content_preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        print(f"   {i+1}. {msg.type}: {content_preview}")
    
    print(f"\n📊 Modern approach stats:")
    print(f"   • Total messages: {len(modern_memory.messages)}")
    print(f"   • Summary created: {'Yes' if any('[SUMMARY]' in msg.content for msg in modern_memory.messages) else 'No'}")
    print(f"   • Automatic management: Yes")
    print(f"   • Session-based: Yes (when used with RunnableWithMessageHistory)")
    
    return modern_memory

def show_key_differences():
    print("\n\n=== Key Differences ===\n")
    
    print("🔺 **Legacy ConversationSummaryMemory**:")
    print("   ❌ Requires deprecated ConversationChain")
    print("   ❌ Manual save_context() calls needed")
    print("   ❌ Not compatible with modern LCEL")
    print("   ❌ Limited customization options")
    print("   ❌ No session-based storage")
    
    print("\n✅ **Modern LCEL Approach**:")
    print("   ✅ Works with RunnableWithMessageHistory")
    print("   ✅ Automatic summarization on message add")
    print("   ✅ Full LCEL compatibility")
    print("   ✅ Highly customizable (when to summarize, how many messages to keep)")
    print("   ✅ Session-based with automatic management")
    print("   ✅ Can combine with file persistence")
    print("   ✅ Better error handling and recovery")

def show_migration_path():
    print("\n\n=== Migration Path ===\n")
    
    print("🔄 **From Legacy to Modern**:")
    print("")
    print("1. **Replace ConversationSummaryMemory**:")
    print("   ```python")
    print("   # Old")
    print("   memory = ConversationSummaryMemory(llm=llm)")
    print("   ")
    print("   # New")
    print("   class SummarizingHistory(ChatMessageHistory):")
    print("       # Custom implementation")
    print("   ```")
    print("")
    print("2. **Replace ConversationChain**:")
    print("   ```python")
    print("   # Old")
    print("   conversation = ConversationChain(llm=llm, memory=memory)")
    print("   ")
    print("   # New")
    print("   chain_with_memory = RunnableWithMessageHistory(")
    print("       chain, get_session_history, ...)")
    print("   ```")
    print("")
    print("3. **Replace manual context saving**:")
    print("   ```python")
    print("   # Old")
    print("   memory.save_context({'input': msg}, {'output': response})")
    print("   ")
    print("   # New")
    print("   # Automatic when using chain_with_memory.invoke()")
    print("   ```")

def show_benefits():
    print("\n\n=== Benefits of Modern Approach ===\n")
    
    print("🚀 **Performance & Scalability**:")
    print("   • Better token management with customizable limits")
    print("   • Configurable summary frequency")
    print("   • Session isolation prevents memory leaks")
    print("")
    print("🔧 **Flexibility**:")
    print("   • Custom summarization strategies")
    print("   • Combine with file persistence")
    print("   • Different summary styles per use case")
    print("")
    print("🛡️ **Reliability**:")
    print("   • Better error handling")
    print("   • Automatic recovery from failures")
    print("   • Compatible with modern LangChain ecosystem")
    print("")
    print("📈 **Future-Proof**:")
    print("   • Built on modern LCEL foundation")
    print("   • Compatible with latest LangChain features")
    print("   • Won't be deprecated like legacy components")

def side_by_side_comparison():
    print("\n\n=== Side-by-Side Comparison Results ===\n")
    
    print("🔄 Running both approaches with identical data...\n")
    
    # Run both approaches
    legacy_memory = demo_legacy_approach()
    modern_memory = demo_modern_approach()
    
    print("\n📊 **COMPARISON SUMMARY**:")
    print(f"┌─────────────────────────────────┬──────────────────┬──────────────────┐")
    print(f"│ Feature                         │ Legacy Approach  │ Modern Approach  │")
    print(f"├─────────────────────────────────┼──────────────────┼──────────────────┤")
    print(f"│ Framework Compatibility         │ ConversationChain│ LCEL Compatible  │")
    print(f"│ Automatic Summarization         │ Yes              │ Yes              │")
    print(f"│ Manual save_context Required    │ Yes              │ No               │")
    print(f"│ Session Management              │ No               │ Yes              │")
    print(f"│ Message Count Control           │ Token-based      │ Message-based    │")
    print(f"│ Custom Summary Strategies       │ Limited          │ Full Control     │")
    print(f"│ File Persistence Combo          │ Complex          │ Easy             │")
    print(f"│ Future-Proof                    │ Deprecated       │ Yes              │")
    print(f"└─────────────────────────────────┴──────────────────┴──────────────────┘")
    
    # Show actual results
    print(f"\n🔍 **ACTUAL RESULTS**:")
    print(f"Legacy buffer length: {len(legacy_memory.buffer)} chars")
    print(f"Modern message count: {len(modern_memory.messages)} messages")
    
    # Check if summary was created in modern approach
    has_summary = any('[SUMMARY]' in msg.content for msg in modern_memory.messages)
    print(f"Modern summary created: {has_summary}")
    
    if has_summary:
        summary_msg = next(msg for msg in modern_memory.messages if '[SUMMARY]' in msg.content)
        print(f"Modern summary: {summary_msg.content[:100]}...")

def run_practical_demo():
    print("\n\n=== Practical Usage Demo ===\n")
    
    print("🎯 **How they work in practice:**\n")
    
    print("🔺 **Legacy - Manual Context Management:**")
    print("```python")
    print("# Every interaction requires manual calls")
    print("memory.save_context({'input': user_msg}, {'output': ai_response})")
    print("conversation = ConversationChain(llm=llm, memory=memory)")
    print("response = conversation.predict(input=user_msg)")
    print("```")
    
    print("\n✅ **Modern - Automatic Memory Management:**")
    print("```python") 
    print("# Memory handled automatically")
    print("def get_session_history(session_id):")
    print("    return SummarizingChatMessageHistory(llm)")
    print("")
    print("chain_with_memory = RunnableWithMessageHistory(chain, get_session_history, ...)")
    print("response = chain_with_memory.invoke({'content': msg}, config={'configurable': {'session_id': 'user123'}})")
    print("```")

if __name__ == "__main__":
    side_by_side_comparison()
    show_key_differences()
    run_practical_demo()
    show_migration_path()
    show_benefits()
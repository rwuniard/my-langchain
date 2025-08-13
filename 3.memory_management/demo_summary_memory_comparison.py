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
    
    print("🔺 OLD WAY:")
    print("```python")
    print("from langchain.memory import ConversationSummaryMemory")
    print("from langchain.chains import ConversationChain")
    print("")
    print("# Create summary memory")
    print("memory = ConversationSummaryMemory(")
    print("    llm=llm,")
    print("    return_messages=True")
    print(")")
    print("")
    print("# Use with deprecated ConversationChain")
    print("conversation = ConversationChain(")
    print("    llm=llm,")
    print("    memory=memory")
    print(")")
    print("```")
    
    # Actually demonstrate the legacy approach
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Create legacy summary memory
    legacy_memory = ConversationSummaryMemory(
        llm=llm,
        return_messages=True
    )
    
    print("\n📝 Adding some conversation to legacy memory...")
    
    # Add conversation manually
    legacy_memory.save_context(
        {"input": "Hi, I'm Alice, a 25-year-old software engineer from Seattle."},
        {"output": "Hello Alice! Nice to meet you. How can I help you today?"}
    )
    
    legacy_memory.save_context(
        {"input": "I work with Python and machine learning at a tech startup."},
        {"output": "That's great! Machine learning with Python is very popular. What kind of ML projects are you working on?"}
    )
    
    legacy_memory.save_context(
        {"input": "We're building recommendation systems for e-commerce."},
        {"output": "Recommendation systems are fascinating! Are you using collaborative filtering or content-based approaches?"}
    )
    
    print(f"💭 Legacy memory buffer:\n{legacy_memory.buffer}")

def demo_modern_approach():
    print("\n\n=== Modern LCEL Approach ===\n")
    
    print("✅ NEW WAY:")
    print("```python")
    print("class SummarizingChatMessageHistory(ChatMessageHistory):")
    print("    def __init__(self, llm, max_messages=10):")
    print("        super().__init__()")
    print("        self.llm = llm")
    print("        self.max_messages = max_messages")
    print("        # Create summarization chain")
    print("        self.summary_chain = summary_prompt | llm")
    print("")
    print("    def add_message(self, message):")
    print("        super().add_message(message)")
    print("        if len(self.messages) > self.max_messages:")
    print("            self._summarize_conversation()")
    print("")
    print("def get_session_history(session_id):")
    print("    return SummarizingChatMessageHistory(llm)")
    print("")
    print("chain_with_memory = RunnableWithMessageHistory(")
    print("    chain, get_session_history, ...)")
    print("```")

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

if __name__ == "__main__":
    demo_legacy_approach()
    demo_modern_approach()
    show_key_differences()
    show_migration_path()
    show_benefits()
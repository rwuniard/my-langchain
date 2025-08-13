"""
Modern LCEL approach to conversation summary memory
"""
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationSummaryBufferMemory
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

def create_summary_chain():
    """Create a chain that uses conversation summary for long conversations"""
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, verbose=True)
    
    # Create prompt template for the main conversation
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. You have access to a summary of the previous conversation and recent messages."),
        ("system", "Previous conversation summary: {summary}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{content}")
    ])
    
    # Create the basic chain
    chain = prompt | llm | StrOutputParser()
    
    return chain, llm

def demo_summary_memory():
    print("=== Modern LCEL Conversation Summary Memory ===\n")
    print("💡 This demo shows how to implement conversation summarization")
    print("   to handle long conversations efficiently.\n")
    
    chain, llm = create_summary_chain()
    
    # Create summary memory for this demo
    summary_memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=100,  # Summarize when conversation exceeds 100 tokens
        return_messages=True
    )
    
    print("🎯 Token limit set to 100 - conversation will be summarized when exceeded")
    print("📝 Let's simulate a long conversation...\n")
    
    # Simulate conversation messages
    conversations = [
        ("human", "Hi! My name is Alice and I'm 25 years old."),
        ("ai", "Hello Alice! Nice to meet you. How can I help you today?"),
        ("human", "I work as a software engineer at a tech company in San Francisco."),
        ("ai", "That's great! Software engineering is an exciting field. What technologies do you work with?"),
        ("human", "I mainly work with Python and JavaScript. I'm building web applications."),
        ("ai", "Excellent! Python and JavaScript are very popular for web development. Are you working on any interesting projects?"),
        ("human", "Yes, I'm building a machine learning platform for data scientists."),
        ("ai", "That sounds fascinating! Machine learning platforms are very valuable. What specific features are you implementing?"),
    ]
    
    # Add conversations to memory
    for msg_type, content in conversations:
        if msg_type == "human":
            summary_memory.chat_memory.add_user_message(content)
        else:
            summary_memory.chat_memory.add_ai_message(content)
    
    print("📊 After adding messages:")
    print(f"   Total messages in memory: {len(summary_memory.chat_memory.messages)}")
    print(f"   Current buffer: {summary_memory.moving_summary_buffer}")
    
    # Check if summary was created
    if hasattr(summary_memory, 'moving_summary_buffer') and summary_memory.moving_summary_buffer:
        print(f"📄 Summary created: {summary_memory.moving_summary_buffer}")
    
    # Get the formatted memory content
    memory_content = summary_memory.buffer
    print(f"\n💭 Memory content:\n{memory_content}")
    
    return summary_memory

def create_manual_summary_approach():
    """Show how to manually implement summary in modern LCEL"""
    print("\n\n=== Manual Summary Implementation in LCEL ===\n")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Create a summarization chain
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the following conversation in 2-3 sentences, focusing on key information about the person and topics discussed."),
        ("human", "Conversation to summarize:\n{conversation}")
    ])
    
    summary_chain = summary_prompt | llm | StrOutputParser()
    
    def get_session_history_with_summary(session_id: str) -> ChatMessageHistory:
        """Enhanced session history that maintains summaries"""
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        
        history = store[session_id]
        
        # If conversation is too long, summarize older messages
        if len(history.messages) > 10:  # Summarize when more than 10 messages
            print("📝 Conversation getting long, creating summary...")
            
            # Get messages to summarize (all but last 4)
            messages_to_summarize = history.messages[:-4]
            
            # Convert messages to text for summarization
            conversation_text = "\n".join([
                f"{msg.type}: {msg.content}" for msg in messages_to_summarize
            ])
            
            # Create summary
            summary = summary_chain.invoke({"conversation": conversation_text})
            print(f"📄 Created summary: {summary}")
            
            # Keep only recent messages + summary
            recent_messages = history.messages[-4:]
            
            # Create new history with summary + recent messages
            new_history = ChatMessageHistory()
            new_history.add_ai_message(f"[SUMMARY] {summary}")
            
            for msg in recent_messages:
                new_history.add_message(msg)
            
            # Replace the history
            store[session_id] = new_history
            history = new_history
            
            print(f"📊 Reduced from {len(messages_to_summarize) + len(recent_messages)} to {len(history.messages)} messages")
        
        return history
    
    # Create chain with summary-aware history
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use any summary information and recent conversation history to provide contextual responses."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{content}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    chain_with_summary = RunnableWithMessageHistory(
        chain,
        get_session_history_with_summary,
        input_messages_key="content",
        history_messages_key="history"
    )
    
    return chain_with_summary

def test_manual_summary():
    """Test the manual summary implementation"""
    print("🧪 Testing manual summary implementation...\n")
    
    chain_with_summary = create_manual_summary_approach()
    
    # Simulate a long conversation
    messages = [
        "Hi, I'm Bob, a 30-year-old doctor from Seattle.",
        "I specialize in cardiology at Seattle General Hospital.",
        "I've been practicing for 5 years now.",
        "I'm interested in learning about AI applications in medicine.",
        "Specifically, I want to know about diagnostic AI tools.",
        "Can you tell me about machine learning in medical imaging?",
        "I'm also curious about natural language processing for medical records.",
        "How accurate are AI systems compared to human doctors?",
        "What are the ethical considerations of using AI in healthcare?",
        "Do you think AI will replace doctors in the future?",
        "What should I study to better understand medical AI?",
        "Can you recommend some resources for learning about this?",
    ]
    
    session_id = "bob_doctor"
    
    for i, message in enumerate(messages):
        print(f"💬 Message {i+1}: {message[:50]}...")
        
        result = chain_with_summary.invoke(
            {"content": message},
            config={"configurable": {"session_id": session_id}}
        )
        
        print(f"🤖 Response: {result[:80]}...\n")
        
        # Show when summarization happens
        if i == 10:  # After 11 messages (triggering summary)
            print("🔄 Summary should have been created here!\n")

if __name__ == "__main__":
    # Demo the legacy approach adaptation
    summary_memory = demo_summary_memory()
    
    # Demo the manual LCEL approach
    test_manual_summary()
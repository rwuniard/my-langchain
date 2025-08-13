"""
Production-ready LCEL conversation with automatic summarization
"""
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
import os

load_dotenv()

class SummarizingChatMessageHistory(ChatMessageHistory):
    """Enhanced ChatMessageHistory that automatically summarizes long conversations"""
    
    def __init__(self, llm, max_messages=10, summary_message_count=6):
        super().__init__()
        self.llm = llm
        self.max_messages = max_messages
        self.summary_message_count = summary_message_count
        self.summary_prefix = "[CONVERSATION SUMMARY]"
        
        # Create summarization chain
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize the following conversation concisely. Focus on key information about the person, important topics discussed, and any relevant context. Keep it brief but informative."),
            ("human", "{conversation}")
        ])
        self.summary_chain = summary_prompt | llm | StrOutputParser()
    
    def add_message(self, message):
        """Add message and summarize if conversation gets too long"""
        super().add_message(message)
        
        # Check if we need to summarize
        if len(self.messages) > self.max_messages:
            self._summarize_conversation()
    
    def _summarize_conversation(self):
        """Summarize older messages and keep recent ones"""
        print(f"📝 Conversation has {len(self.messages)} messages, summarizing...")
        
        # Find existing summary
        summary_index = None
        for i, msg in enumerate(self.messages):
            if msg.content.startswith(self.summary_prefix):
                summary_index = i
                break
        
        # Determine which messages to summarize
        if summary_index is not None:
            # There's already a summary, update it with messages after the summary
            # but keep the last few messages as-is
            messages_to_summarize = self.messages[summary_index+1:-self.summary_message_count]
            recent_messages = self.messages[-self.summary_message_count:]
            old_summary = self.messages[summary_index].content[len(self.summary_prefix):].strip()
        else:
            # No existing summary, summarize all but recent messages
            messages_to_summarize = self.messages[:-self.summary_message_count]
            recent_messages = self.messages[-self.summary_message_count:]
            old_summary = ""
        
        if not messages_to_summarize:
            return  # Nothing to summarize
        
        # Convert messages to text
        conversation_text = ""
        if old_summary:
            conversation_text += f"Previous summary: {old_summary}\n\nRecent conversation:\n"
        
        conversation_text += "\n".join([
            f"{msg.type}: {msg.content}" for msg in messages_to_summarize
        ])
        
        # Create new summary
        try:
            new_summary = self.summary_chain.invoke({"conversation": conversation_text})
            print(f"📄 Created summary: {new_summary[:100]}...")
            
            # Rebuild messages list with new summary + recent messages
            self.messages = []
            self.add_ai_message(f"{self.summary_prefix} {new_summary}")
            
            for msg in recent_messages:
                super().add_message(msg)  # Use super() to avoid recursion
            
            print(f"📊 Conversation summarized: now {len(self.messages)} messages")
            
        except Exception as e:
            print(f"❌ Failed to create summary: {e}")

# Store for session histories
store = {}

def get_session_history(session_id: str) -> SummarizingChatMessageHistory:
    """Get or create a summarizing chat message history for the given session ID"""
    if session_id not in store:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        store[session_id] = SummarizingChatMessageHistory(
            llm=llm,
            max_messages=12,  # Start summarizing after 12 messages
            summary_message_count=4  # Keep last 4 messages as-is
        )
        print(f"✨ Created new summarizing history for session: {session_id}")
    else:
        print(f"📚 Using existing history for session: {session_id} ({len(store[session_id].messages)} messages)")
    
    return store[session_id]

def main():
    print("Hello from LCEL Conversation Summary Memory!")
    print("🧠 This version automatically summarizes long conversations")
    print("📏 Conversations are summarized when they exceed 12 messages")
    print("🔄 Recent 4 messages are always kept for context\n")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create a prompt template that works well with summaries
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the conversation history (including any summaries) to provide contextual and personalized responses."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{content}")
    ])

    # Create a chain
    chain = prompt | llm | StrOutputParser()

    # Create a chain that will use the summarizing memory
    chain_with_memory = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="content",
        history_messages_key="history"
    )

    # Get session ID from user
    session_id = input("Enter session ID (or press Enter for 'default'): ").strip()
    if not session_id:
        session_id = "default"
    
    print(f"📂 Using session: {session_id}")
    print("💬 Start chatting! Type 'exit' to quit, 'history' to see conversation state.\n")

    while True:
        user_input = input(">> ")
        
        if user_input == "exit":
            print("👋 Goodbye! Your conversation summary is preserved.")
            break
        elif user_input == "history":
            # Show current conversation state
            history = store.get(session_id)
            if history:
                print(f"\n📊 Conversation state: {len(history.messages)} messages")
                for i, msg in enumerate(history.messages):
                    prefix = "📄" if msg.content.startswith("[CONVERSATION SUMMARY]") else "💬"
                    content = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
                    print(f"   {i+1}. {prefix} {msg.type}: {content}")
                print()
            else:
                print("📭 No conversation history yet.\n")
            continue
        
        result = chain_with_memory.invoke(
            {"content": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        print(f"AI: {result}")

if __name__ == "__main__":
    main()
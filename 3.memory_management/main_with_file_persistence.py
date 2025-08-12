"""
Memory management with file persistence using modern LCEL approach
"""
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import FileChatMessageHistory
from dotenv import load_dotenv
import os

load_dotenv()
#   How does the message get saved automatically in FileChatMessageHistory?
#   The magic is simple: FileChatMessageHistory overrides the add_message() method to save to disk every
#   time it's called. Since RunnableWithMessageHistory calls this method for both user input and AI
#   responses, everything gets saved automatically!
def get_session_history(session_id: str) -> FileChatMessageHistory:
    """Get or create a file-based chat message history for the given session ID"""
    # Create a directory for conversation files if it doesn't exist
    conversations_dir = "conversations"
    if not os.path.exists(conversations_dir):
        os.makedirs(conversations_dir)
    
    # Each session gets its own file
    file_path = os.path.join(conversations_dir, f"conversation_{session_id}.json")
    
    print(f"💾 Using conversation file: {file_path}")
    
    # FileChatMessageHistory automatically loads existing conversations from the file
    return FileChatMessageHistory(file_path)

def main():
    print("Hello from 3-memory-management with file persistence!")
    print("💾 Conversations will be saved to files automatically")
    print("🔄 Previous conversations will be loaded when you restart")
    
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
        get_session_history,  # Now returns FileChatMessageHistory instead of ChatMessageHistory
        input_messages_key="content",
        history_messages_key="history"
    )

    # Show existing conversations
    conversations_dir = "conversations"
    if os.path.exists(conversations_dir):
        existing_files = [f for f in os.listdir(conversations_dir) if f.endswith('.json')]
        if existing_files:
            print(f"\n📁 Found {len(existing_files)} existing conversation(s):")
            for file in existing_files:
                session_id = file.replace('conversation_', '').replace('.json', '')
                print(f"   - Session ID: {session_id}")

    print("\n💬 Start chatting! Type 'exit' to quit.")
    print("🔧 Try different session IDs to create separate conversations.")
    
    # Get session ID from user
    session_id = input("Enter session ID (or press Enter for 'default'): ").strip()
    if not session_id:
        session_id = "default"
    
    print(f"📂 Using session: {session_id}")

    while True:
        user_input = input(">> ")
        if user_input == "exit":
            print("💾 Conversation automatically saved to file!")
            print("🔄 Restart the program to load this conversation again.")
            break
        
        result = chain_with_memory.invoke(
            {"content": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        print(f"AI response: {result}")

if __name__ == "__main__":
    main()
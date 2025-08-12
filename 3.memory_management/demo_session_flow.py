"""
Demonstration of how get_session_history works internally
"""
from langchain_community.chat_message_histories import ChatMessageHistory

# Simulate the global store
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Get or create a chat message history for the given session ID"""
    print(f"🔍 Looking for session: {session_id}")
    
    if session_id not in store:
        print(f"✨ Creating new history for session: {session_id}")
        store[session_id] = ChatMessageHistory()
    else:
        print(f"📚 Found existing history for session: {session_id}")
        print(f"   Messages in history: {len(store[session_id].messages)}")
    
    return store[session_id]

def demonstrate_session_flow():
    print("=== Session History Demonstration ===\n")
    
    # First call - creates new session
    print("1. First call to get_session_history('alice'):")
    alice_history_1 = get_session_history("alice")
    print(f"   History object: {alice_history_1}")
    print(f"   Messages count: {len(alice_history_1.messages)}")
    
    # Add a message to Alice's history
    print("\n2. Adding a message to Alice's history:")
    alice_history_1.add_user_message("Hello, I'm Alice!")
    alice_history_1.add_ai_message("Hi Alice, nice to meet you!")
    print(f"   Messages count: {len(alice_history_1.messages)}")
    
    # Second call - retrieves existing session
    print("\n3. Second call to get_session_history('alice'):")
    alice_history_2 = get_session_history("alice")
    print(f"   Same object? {alice_history_1 is alice_history_2}")
    print(f"   Messages count: {len(alice_history_2.messages)}")
    
    # Different session
    print("\n4. Call to get_session_history('bob'):")
    bob_history = get_session_history("bob")
    print(f"   Messages count: {len(bob_history.messages)}")
    
    # Show store contents
    print("\n5. Current store contents:")
    for session_id, history in store.items():
        print(f"   Session '{session_id}': {len(history.messages)} messages")
        for i, msg in enumerate(history.messages):
            print(f"     {i+1}. {msg.type}: {msg.content}")

if __name__ == "__main__":
    demonstrate_session_flow()
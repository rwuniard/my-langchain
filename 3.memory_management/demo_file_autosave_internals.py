"""
Demonstration of how FileChatMessageHistory automatically saves conversations
"""
from langchain_community.chat_message_histories import FileChatMessageHistory
import os
import json
import time

def demo_automatic_saving():
    print("=== How FileChatMessageHistory Auto-saves ===\n")
    
    # Clean up any existing demo file
    demo_file = "demo_autosave.json"
    if os.path.exists(demo_file):
        os.remove(demo_file)
    
    print("📝 Creating FileChatMessageHistory...")
    history = FileChatMessageHistory(demo_file)
    
    print(f"📁 File exists before adding messages: {os.path.exists(demo_file)}")
    
    print("\n🔍 Let's watch what happens when we add messages...")
    
    # Add first message and check file immediately
    print("\n1️⃣ Adding first user message...")
    history.add_user_message("Hello! This is my first message.")
    
    print(f"   📁 File exists after add_user_message: {os.path.exists(demo_file)}")
    if os.path.exists(demo_file):
        with open(demo_file, 'r') as f:
            content = f.read()
        print(f"   📄 File size: {len(content)} characters")
        print(f"   📊 Messages in memory: {len(history.messages)}")
    
    time.sleep(0.1)  # Small delay for demonstration
    
    # Add AI message
    print("\n2️⃣ Adding AI response...")
    history.add_ai_message("Hi there! I received your message.")
    
    print(f"   📁 File exists after add_ai_message: {os.path.exists(demo_file)}")
    if os.path.exists(demo_file):
        with open(demo_file, 'r') as f:
            content = f.read()
        print(f"   📄 File size: {len(content)} characters")
        print(f"   📊 Messages in memory: {len(history.messages)}")
    
    # Show actual file contents
    print("\n📄 Current file contents:")
    if os.path.exists(demo_file):
        with open(demo_file, 'r') as f:
            data = json.load(f)
        print(f"   Found {len(data)} messages in file:")
        for i, msg in enumerate(data):
            print(f"     {i+1}. {msg['type']}: {msg['data']['content']}")
    
    print("\n🔄 Testing persistence - creating NEW FileChatMessageHistory with same file...")
    
    # Create a completely new instance pointing to the same file
    history2 = FileChatMessageHistory(demo_file)
    print(f"   📊 Messages loaded automatically: {len(history2.messages)}")
    print("   🎯 Loaded messages:")
    for i, msg in enumerate(history2.messages):
        print(f"     {i+1}. {msg.type}: {msg.content}")
    
    # Clean up
    if os.path.exists(demo_file):
        os.remove(demo_file)

def show_when_saving_happens():
    print("\n\n=== When Does Auto-saving Happen? ===\n")
    
    print("🔍 FileChatMessageHistory saves to file IMMEDIATELY when:")
    print("   ✅ add_user_message() is called")
    print("   ✅ add_ai_message() is called") 
    print("   ✅ add_message() is called")
    print("   ✅ Any message is added to the history")
    
    print("\n📋 The saving process:")
    print("   1. You call: history.add_user_message('Hello')")
    print("   2. Message added to internal messages list")
    print("   3. IMMEDIATELY: Entire messages list serialized to JSON")
    print("   4. IMMEDIATELY: JSON written to file")
    print("   5. File is closed and persisted")
    
    print("\n💡 This means:")
    print("   ✅ No manual save() method needed")
    print("   ✅ No risk of losing messages")
    print("   ✅ Conversation persists even if program crashes")
    print("   ✅ Multiple instances can share the same file")

def demo_crash_safety():
    print("\n\n=== Crash Safety Demonstration ===\n")
    
    demo_file = "demo_crash_safety.json"
    if os.path.exists(demo_file):
        os.remove(demo_file)
    
    print("💥 Simulating program that might crash...")
    
    # Start conversation
    history = FileChatMessageHistory(demo_file)
    history.add_user_message("I hope this doesn't get lost!")
    
    print(f"   📁 File saved: {os.path.exists(demo_file)}")
    
    # Simulate crash by deleting the history object
    print("   💥 Simulating crash... (deleting history object)")
    del history
    
    print("   🔄 'Restarting' program... (creating new FileChatMessageHistory)")
    
    # "Restart" by creating new instance
    history_after_crash = FileChatMessageHistory(demo_file)
    print(f"   📊 Messages recovered: {len(history_after_crash.messages)}")
    
    if history_after_crash.messages:
        print(f"   💾 Recovered message: {history_after_crash.messages[0].content}")
        print("   ✅ No data lost!")
    
    # Clean up
    if os.path.exists(demo_file):
        os.remove(demo_file)

def show_internal_mechanism():
    print("\n\n=== Internal Mechanism (Simplified) ===\n")
    
    print("🔧 How FileChatMessageHistory works internally:")
    print("""
class FileChatMessageHistory:
    def __init__(self, file_path):
        self.file_path = file_path
        self.messages = self._load_messages()  # Load existing messages
    
    def add_message(self, message):
        self.messages.append(message)  # Add to memory
        self._save_messages()          # IMMEDIATELY save to file
    
    def _save_messages(self):
        # Convert messages to JSON and write to file
        with open(self.file_path, 'w') as f:
            json.dump(self.messages, f)
    
    def _load_messages(self):
        # Load messages from file if it exists
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                return json.load(f)
        return []
""")
    
    print("🎯 Key insight: Every add_message() triggers _save_messages()")
    print("   This is why saving is automatic - it's built into every write operation!")

if __name__ == "__main__":
    demo_automatic_saving()
    show_when_saving_happens()
    demo_crash_safety()
    show_internal_mechanism()
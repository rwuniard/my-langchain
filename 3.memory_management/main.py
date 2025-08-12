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


def main():
    print("Hello from 3-memory-management!")

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
        history_messages_key="history" # This must match the variable_name in the MessagesPlaceholder
    )

    while True:
        user_input = input(">> ")
        if user_input == "exit":
            break
        result = chain_with_memory.invoke(
            {"content": user_input},
            config={"configurable": {"session_id": "1"}} #the session id in real world context would be a user id
        )
        print(f"AI response: {result}")


if __name__ == "__main__":
    main()

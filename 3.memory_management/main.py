from langchain.prompts import HumanMessagePromptTemplate, ChatMessagePromptTemplate
from langchain.chat_models import ChatOpenAI


def main():
    print("Hello from 3-memory-management!")
    while True:
        user_input = input(">> ")
        if user_input == "exit":
            break
        print(f"You entered: {user_input}")


if __name__ == "__main__":
    main()

from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()


def main():
    print("Hello from 3-memory-management!")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create a prompt template that has input variables content and human message prompt template.
    prompt = ChatPromptTemplate(
        input_variables=["content"],
        messages = [
            HumanMessagePromptTemplate.from_template("{content}")
        ]
    )

    # Create a chain
    chain = prompt | llm | StrOutputParser()

    while True:
        user_input = input(">> ")
        if user_input == "exit":
            break
        result = chain.invoke({"content": user_input})
        print(f"AI response: {result}")


if __name__ == "__main__":
    main()

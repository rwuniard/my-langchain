import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

load_dotenv()


code_prompt = PromptTemplate(
    input_variables=["language", "task"],
    template="Write a very short {language} function that will {task}"
)
code_check_prompt = PromptTemplate(
    input_variables=["language", "code"],
    template="Write a test code for the following {language} code:\n {code}"
)


def main():
    print("Hello from 2-usingchain!")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OpenAI API key not found!")
        print("Please set your OPENAI_API_KEY environment variable:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    try:
        llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.2)
        code_chain = LLMChain(
            llm=llm, 
            prompt=code_prompt,
            output_key="code"
        )
        # response = code_chain.invoke({"language": "Python", "task": "print 10 numbers"})
        # print(response)
        # print(f"🤖 AI Response: {response['code']}")

        code_check_chain = LLMChain(
            llm=llm,
            prompt=code_check_prompt,
            output_key="test_code"
        )

        # Create a sequential chain to check the code
        chain = SequentialChain(
            chains=[code_chain, code_check_chain],
            input_variables=["language", "task"],
            output_variables=["code", "test_code"]
        )
        response = chain.invoke({"language": "Python", "task": "print 10 numbers"})
      
        response_code_check = code_check_chain.invoke({"language": "Python", "code": response['code']})
        print(response_code_check)
        print(f"🤖 AI Response: {response_code_check['test_code']}")
    except Exception as e:
        print(f"❌ Error: {e}")



if __name__ == "__main__":
    main()

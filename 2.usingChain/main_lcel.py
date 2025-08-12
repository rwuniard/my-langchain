import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()


code_prompt = PromptTemplate(
    input_variables=["language", "task"],
    template="Write a very short {language} function that will {task}"
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
        # Instead of this.
        # code_chain = LLMChain(llm=llm, prompt=code_prompt)
        # The more modern way is to use LCEL
        code_chain = code_prompt | llm
        response = code_chain.invoke({"language": "Python", "task":"print 10 numbers"})
        print(f"🤖 AI Response: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")



if __name__ == "__main__":
    main()

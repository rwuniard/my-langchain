import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import argparse

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, default="Python")
parser.add_argument("--task", type=str, default="print 10 numbers")
args = parser.parse_args()

# Simple prompts without Pydantic parsers
code_prompt = PromptTemplate(
    input_variables=["language", "task"],
    template="Write a very short {language} function that will {task}"
)

code_check_prompt = PromptTemplate(
    input_variables=["language", "code"],
    template="Write a test for the following {language} code:\n{code}"
)

def main():
    print("Hello from simplified version!")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OpenAI API key not found!")
        print("Please set your OPENAI_API_KEY environment variable:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Simple chains without structured output
        generate_code_chain = code_prompt | llm
        code_check_chain = code_check_prompt | llm
        
        # Method 1: Simple chaining with RunnableLambda
        def simple_chain_function(inputs):
            """Simple function to chain the operations"""
            # Execute first chain: Generate code
            code_result = generate_code_chain.invoke(inputs)
            # Execute second chain: Check/test the generated code
            test_result = code_check_chain.invoke({
                "code": code_result.content,  # Note: .content instead of .code
                "language": inputs["language"]
            })
            return {
                "original_code": code_result.content,
                "test_code": test_result.content
            }
        
        simple_chain = RunnableLambda(simple_chain_function)
        
        # Method 2: Complex chaining with RunnablePassthrough
        def extract_code_and_add_language(data):
            """Extract code from the first chain and add language for the second chain"""
            return {
                "code": data["generated"].content,  # Note: .content instead of .code
                "language": data["language"]
            }
        
        complex_chain = (
            RunnablePassthrough.assign(generated=generate_code_chain)
            | RunnableLambda(extract_code_and_add_language)
            | code_check_chain
        )
        
        # Execute different approaches
        print("🔗 Method 1: Simple chaining (WITHOUT Pydantic)...")
        result1 = simple_chain.invoke({"language": args.language, "task": args.task})
        print(f"📝 Original Code: {result1['original_code']}")
        print(f"🧪 Test Code: {result1['test_code']}")
        
        print("\n" + "="*50)
        print("🔗 Method 2: Complex chaining (WITHOUT Pydantic)...")
        result2 = complex_chain.invoke({"language": args.language, "task": args.task})
        print(f"🤖 Final Result: {result2.content}")
        
        print("\n" + "="*50)
        print("🔍 Manual step-by-step execution (WITHOUT Pydantic):")
        
        # Manual execution
        step1 = generate_code_chain.invoke({"language": args.language, "task": args.task})
        print(f"📝 Generated Code: {step1.content}")
        
        step2 = code_check_chain.invoke({"code": step1.content, "language": args.language})
        print(f"🧪 Test Code: {step2.content}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
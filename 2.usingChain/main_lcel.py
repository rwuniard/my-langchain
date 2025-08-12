import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from langchain.chains import LLMChain
import argparse

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, default="Python")
parser.add_argument("--task", type=str, default="print 10 numbers")
args = parser.parse_args()
args.language = args.language.lower()
args.task = args.task.lower()


# Define the models
class Code(BaseModel):
    code: str

class CodeCheck(BaseModel):
    final_code: str

# Create parser first
code_parser = PydanticOutputParser(pydantic_object=Code)
code_check_parser = PydanticOutputParser(pydantic_object=CodeCheck)

code_prompt = PromptTemplate(
    input_variables=["language", "task"],
    template="Write a very short {language} function that will {task}\n{format_instructions}\n\nIMPORTANT: Return ONLY valid JSON, no other text!",
    partial_variables={"format_instructions": code_parser.get_format_instructions()}
)

code_check_prompt = PromptTemplate(
    input_variables=["language", "code"],
    template="Write a test for the following {language} code:\n {code}\n{format_instructions}",
    partial_variables={"format_instructions": code_check_parser.get_format_instructions()}
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
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        # Instead of this.
        # code_chain = LLMChain(llm=llm, prompt=code_prompt)
        # The more modern way is to use LCEL
        # Create individual chains first
        generate_code_chain = code_prompt | llm.with_structured_output(Code)
        code_check_chain = code_check_prompt | llm.with_structured_output(CodeCheck)
        
        # Import required classes for chaining
        from langchain_core.runnables import RunnablePassthrough, RunnableLambda
        
        # Method 1: Simple Chaining with RunnableLambda
        def simple_chain_function(inputs):
            """Simple function to chain the operations into a single runnable"""
            # Execute first chain: Generate code
            code_result = generate_code_chain.invoke(inputs)
            # Execute second chain: Check/test the generated code
            test_result = code_check_chain.invoke({
                "code": code_result.code,
                "language": inputs["language"]
            })
            return {
                "original_code": code_result.code,
                "test_code": test_result.final_code
            }
        
        simple_chain = RunnableLambda(simple_chain_function)
        
        # Method 2: Complex Chaining with RunnablePassthrough
        def extract_code_and_add_language(data):
            """Extract code from the first chain and add language for the second chain"""
            return {
                "code": data["generated"].code,
                "language": data["language"]
            }
        
        complex_chain = (
            RunnablePassthrough.assign(generated=generate_code_chain)
            | RunnableLambda(extract_code_and_add_language)
            | code_check_chain
        )
        
        # Execute different chaining methods
        print("🔗 Method 1: Simple chaining with RunnableLambda...")
        result1 = simple_chain.invoke({"language": args.language, "task": args.task})
        print(f"📝 Original Code: {result1['original_code']}")
        print(f"🧪 Test Code: {result1['test_code']}")
        
        print("\n" + "="*50)
        print("🔗 Method 2: Complex chaining with RunnablePassthrough...")
        result2 = complex_chain.invoke({"language": args.language, "task": args.task})
        print(f"🤖 Final Result: {result2.final_code}")
        
        print("\n" + "="*50)
        print("🔍 Manual step-by-step execution (NOT chaining - for comparison):")
        
        # Manual execution - NOT chaining, just sequential calls
        step1 = generate_code_chain.invoke({"language": args.language, "task": args.task})
        print(f"📝 Generated Code: {step1.code}")
        
        step2 = code_check_chain.invoke({"code": step1.code, "language": args.language})
        print(f"🧪 Test Code: {step2.final_code}")
    except Exception as e:
        print(f"❌ Error: {e}")



if __name__ == "__main__":
    main()

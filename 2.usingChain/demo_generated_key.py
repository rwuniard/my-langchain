from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Demonstration showing exactly how "generated" key is created

def mock_generate_code_chain(data):
    """Simulates the generate_code_chain"""
    return {"code": f"def {data['task']}(): pass"}

def extract_code_and_add_language(data):
    """This is the exact function from main_lcel.py"""
    print("🔍 Data received by extract_code_and_add_language:")
    print(f"   Full data: {data}")
    print(f"   Available keys: {list(data.keys())}")
    print(f"   data['generated']: {data['generated']}")
    print(f"   data['language']: {data['language']}")
    
    return {
        "code": data["generated"]["code"],  # Extract code from the "generated" key
        "language": data["language"]        # Pass through the language
    }

def mock_code_check_chain(data):
    """Simulates the final chain"""
    return f"Test for {data['language']} code: {data['code']}"

# Create the chains
generate_chain = RunnableLambda(mock_generate_code_chain)
extract_func = RunnableLambda(extract_code_and_add_language)
check_chain = RunnableLambda(mock_code_check_chain)

# This is exactly how it works in main_lcel.py
demo_chain = (
    RunnablePassthrough.assign(generated=generate_chain)  # 👈 Creates "generated" key!
    | extract_func
    | check_chain
)

print("🚀 Running the exact same pattern as main_lcel.py...")
result = demo_chain.invoke({"language": "Python", "task": "add_numbers"})
print(f"📋 Final result: {result}")
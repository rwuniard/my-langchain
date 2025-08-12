from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Simple demonstration of RunnablePassthrough.assign()

def step1_processor(data):
    """Simulates the first chain - just doubles a number"""
    return {"result": data["number"] * 2}

def step2_processor(data):
    """Show what data looks like after RunnablePassthrough.assign()"""
    print("🔍 Data received by step2_processor:")
    print(f"   Original input: {data}")
    print(f"   Keys available: {list(data.keys())}")
    print(f"   step1_result: {data['step1_result']}")
    return f"Final result: {data['step1_result']['result']} (from {data['number']})"

# Create runnables
step1_chain = RunnableLambda(step1_processor)
step2_chain = RunnableLambda(step2_processor)

# This is the key part - assign() adds the result with a custom key
demo_chain = (
    RunnablePassthrough.assign(step1_result=step1_chain)  # 👈 Creates "step1_result" key
    | step2_chain
)

print("🚀 Running demonstration...")
result = demo_chain.invoke({"number": 5, "original_data": "preserved"})
print(f"📋 Final result: {result}")
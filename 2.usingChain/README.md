# 2-usingchain

LangChain project focused on exploring and implementing chains.

## Setup the Environment

1. **Initialize the project**:
   ```bash
   uv init
   ```

2. **Add dependencies**:
   ```bash
   uv add python-dotenv
   uv add langchain langchain-openai langchain-core
   ```

3. **Set up OpenAI API key**:
   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your-api-key-here
   ```

4. **Run the project**:
   
   **Legacy approach** (using deprecated LLMChain):
   ```bash
   uv run main.py
   ```
   
   **Modern approach** (using LCEL with structured output):
   ```bash
   # Basic usage
   uv run main_lcel.py
   
   # With custom parameters
   uv run main_lcel.py --language JavaScript --task "create a todo list"
   ```
   
   **Simple approach** (using LCEL without Pydantic):
   ```bash
   # Basic usage
   uv run main_simple.py
   
   # With custom parameters
   uv run main_simple.py --language Python --task "calculate factorial"
   ```
   
   **Understanding RunnablePassthrough** (educational demos):
   ```bash
   # See how RunnablePassthrough.assign() creates keys
   uv run demo_passthrough.py
   
   # See specifically how "generated" key is created
   uv run demo_generated_key.py
   ```

## Features

This project demonstrates:
- Basic LangChain chain implementation (legacy and modern approaches)
- Prompt templates with variables
- Integration with OpenAI models
- Environment variable management
- LCEL (LangChain Expression Language) syntax
- Command line argument parsing for dynamic inputs
- Two approaches to chain multiple operations:
  - **Method 1**: Simple chaining with `RunnableLambda`
  - **Method 2**: Complex chaining with `RunnablePassthrough`
- Structured output with Pydantic models vs. simple text output
- Sequential chains for multi-step workflows

## Files

- `main.py` - Legacy implementation using `LLMChain` and `SequentialChain` (deprecated)
- `main_lcel.py` - Modern implementation using LCEL syntax with Pydantic structured output
- `main_simple.py` - Simple LCEL implementation without Pydantic (cleaner setup, verbose output)
- `demo_passthrough.py` - Demonstration of how `RunnablePassthrough.assign()` works
- `demo_generated_key.py` - Specific example showing how the "generated" key is created

## Chaining Approaches

### Method 1: Simple Chaining with RunnableLambda
```python
def simple_chain_function(inputs):
    code_result = generate_code_chain.invoke(inputs)
    test_result = code_check_chain.invoke({"code": code_result.code, "language": inputs["language"]})
    return {"original_code": code_result.code, "test_code": test_result.final_code}

simple_chain = RunnableLambda(simple_chain_function)
```

### Method 2: Complex Chaining with RunnablePassthrough
```python
complex_chain = (
    RunnablePassthrough.assign(generated=generate_code_chain)  # Creates "generated" key
    | RunnableLambda(extract_code_and_add_language)
    | code_check_chain
)

def extract_code_and_add_language(data):
    """Data transformation function"""
    return {
        "code": data["generated"].code,  # Access the "generated" key created by assign()
        "language": data["language"]     # Pass through original language
    }
```

**How `RunnablePassthrough.assign()` works:**
- Takes original input: `{"language": "Python", "task": "add numbers"}`
- Runs `generate_code_chain` on this input
- Adds result to original data with the key name you specify: `generated`
- Output: `{"language": "Python", "task": "add numbers", "generated": Code(...)}`

## Structured Output Comparison

### With Pydantic (Recommended for Production)
- ✅ Clean, consistent output
- ✅ Type safety and validation
- ✅ Easy programmatic access
- ❌ More complex setup

### Without Pydantic (Good for Learning/Prototyping)
- ✅ Simpler setup
- ✅ Human-readable explanations
- ❌ Verbose, inconsistent output
- ❌ Requires manual parsing

## Notes

- `LLMChain` is deprecated in LangChain 0.3+ and will be removed in 1.0
- The modern approach uses LCEL syntax: `prompt | llm`
- LCEL provides better performance and composability
- Use `gpt-4o-mini` or `gpt-4o` for structured output support 
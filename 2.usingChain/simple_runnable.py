from langchain_core.runnables import RunnableLambda

# A RunnableSequence constructed using the `|` operator
sequence1 = RunnableLambda(lambda x: x + 1) | RunnableLambda(lambda x: x + 2)
result1 =    sequence1.invoke(1)
print(result1)

# A RunnableParallel constructed using a dict literal
# mul_2 and mul_5 are run in parallel
sequence2 = RunnableLambda(lambda x: x + 1) | {
    'mul_2': RunnableLambda(lambda x: x * 2),
    'mul_5': RunnableLambda(lambda x: x * 5)
}
result2 = sequence2.invoke(1)
print(result2)
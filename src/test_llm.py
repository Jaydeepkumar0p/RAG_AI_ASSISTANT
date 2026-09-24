from src.services.llm_service import ask_llm

question = "Explain RAG in simple terms."

answer = ask_llm(question)

print(answer)

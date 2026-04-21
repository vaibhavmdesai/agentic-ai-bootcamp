from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

open_ai_model = ChatOpenAI(model="gpt-4.1-nano")
qroq_model = ChatGroq(model="openai/gpt-oss-20b")


m1 = "Who is the president of the United States?"
m2 = "What is his age?"

def main(m1, m2):
    print("Hello from agentic-ai-bootcamp!")

    res1 = open_ai_model.invoke(m1)
    print(f"Response to m1: {res1.content}")

    res2 = open_ai_model.invoke(m2)
    print(f"Response to m2: {res2.content}")


if __name__ == "__main__":
    main(m1, m2)

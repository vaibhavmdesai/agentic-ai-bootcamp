from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langchain.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv(override=True)
llm = ChatOpenAI(model="gpt-4.1-nano")

messages = [SystemMessage(content="You are a helpful assistant that can answer questions in a funny way."),
            HumanMessage(content="Who is the president of the United States?")
            ]

response = llm.invoke(messages)

# Append the response to the messages list to include the context
messages.append(response)

# Now we can ask a follow-up question that references the previous response
messages.append(HumanMessage(content="What is his age?"))

response = llm.invoke(messages)
print(response.content)

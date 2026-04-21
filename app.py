from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langchain.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv(override=True)
llm = ChatOpenAI(model="gpt-4.1-nano")

resp1 = llm.invoke("We are building an AI system for processing medical insurance claims.")

resp2 = llm.invoke("What are the main risks in this system?")

print(f"Response to first message: {resp1.content}")
print(f"Response to second message: {resp2.content}")

# The second response might behave inconsistently because it doesn't have the context of the first message. 
# So, the second message is treated as an independent query to the LLM.


############# Example of maintaining context across multiple interactions with the LLM #############
messages = [
    SystemMessage(content="You are a senior AI architect reviewing production systems."),
    HumanMessage(content="We are building an AI system for processing medical insurance claims."),
    HumanMessage(content="What are the main risks in this system?")
]

response = llm.invoke(messages)
print(response.content)


"""
Reflection:
1. Why did the string based invocation fail?
The string-based invocation failed because it treated each message as an independent input without any context. 
The second message, "What are the main risks in this system?", did not have the required context from the first message.

2. Why did the message-based invocation work?
The message-based invocation worked because it maintained the context of the conversation. 
By using a list of messages, the LLM model is able to understand the relationship between the messages and respond with the combined context.

3. What would break in production AI System if we ignore message history?
Ignoring message history in a production AI system can lead to several issues:
- Inconsistent responses: The AI might provide answers that do not align with previous interactions, leading to confusion for users.
- Poor user experience: Users may become frustrated if the AI fails to remember previous interactions, making it less effective and less engaging.
"""
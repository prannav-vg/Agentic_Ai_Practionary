from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain.agents import create_agent
import math

load_dotenv()

# LLM
model = init_chat_model("openai:gpt-5.6-luna")

# Tools
@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

@tool
def divide(a: float, b: float) -> str:
    """Divide a by b."""
    if b == 0:
        return "Error: Cannot divide by zero"
    return str(a / b)

@tool
def square_root(number: float) -> str:
    """Calculate the square root of a number."""
    if number < 0:
        return "Error: Cannot calculate square root of negative number"
    return str(math.sqrt(number))

tools = [add, multiply, divide, square_root]

# Create agent
agent = create_agent(
    model=model,
    tools=tools
)

# Run agent
def run_agent(question: str):

    result = agent.invoke({
        "messages": [("user", question)]
    })

    for msg in result["messages"]:

        if msg.type == "human":
            print("User:", msg.content)

        elif msg.type == "ai" and msg.tool_calls:
            for call in msg.tool_calls:
                print("Agent calls:", call["name"])
                print("Arguments:", call["args"])

        elif msg.type == "tool":
            print("Tool result:", msg.content)

        elif msg.type == "ai" and msg.content:
            print("Final answer:", msg.content)


run_agent("What is 1 + 2?")

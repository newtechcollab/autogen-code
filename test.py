from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import MaxMessageTermination, ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.teams import SelectorGroupChat
from autogen_core.tools import FunctionTool
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.messages import ChatMessage
from typing import Callable, Sequence
from autogen_core import CancellationToken
from autogen_agentchat.base import Response


import requests
import asyncio
load_dotenv()

import logging

from autogen_core import TRACE_LOGGER_NAME

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(TRACE_LOGGER_NAME)
logger.setLevel(logging.DEBUG)

# Define a model client. You can use other model client that implements
# the `ChatCompletionClient` interface.
model_client = OpenAIChatCompletionClient(
    model="llama3.2:latest",
    base_url="http://localhost:11434/v1",
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": "unknown",
    },
)

omodel_client = OpenAIChatCompletionClient(
    model="gemini-1.5-flash-8b",
    api_key="AIzaSyBZrz2hrwVib3FdJhs_hfei5-qb0oaYBGM"
)

# Function that finds Inventory for a given Order
async def get_inventory(orderId: str) -> int:
    """
    Get Inventory information for a given Order
    """
    print("Received Order Id for Inventory Check -> ", orderId)
    if (orderId == "100"):
        print("Inventory available -> ", 100)
        return 100
    print("Inventory available -> ", 0)
    return 0

# Function tool for Inventory availability
get_inventory_tool = FunctionTool(get_inventory, description = "Finds Inventory for a given Order")
 
# Inventory Agent - A Custom Agent that checks Inventory for a given Order Id
class InventoryAgent(BaseChatAgent):
    def __init__(self, name: str, description: str):
        super().__init__(name, description)

    @property
    def produced_message_types(self) -> Sequence[type[ChatMessage]]:
        return (TextMessage,)

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        print("Received on_messages call -> ", messages)
        for message in messages:
            print("Message Content -> ", message.content)
        return Response(chat_message=message)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

# Create Inventory Agent
inventory_agent = InventoryAgent("Inventory_Agent", "An Agent that checks the Inventory for a given Order Id")
"""
inventory_agent = AssistantAgent(
    name="inventory_agent", 
    description="A helpful Assistant that checks whether Inventory is available or not for the given Order",
    model_client=model_client, 
    tools=[get_inventory_tool],
    system_message="You are an AI Assistant that finds whether Inventory is available or not for the given Order. Use your tool to do the task. Do not use your knowledge or information from Internet.")
"""
# Function that fulfills a given Order
async def fulfill_order(orderId: str) -> str:
    """
    Fulfills a given Order
    """
    print("Received Order for Fulfillment -> ", orderId)
    if (orderId == "100"):
        print("Order fulfilled")
        return "Order fulfilled"
    print("Order not fulfilled")
    return "Order not fulfilled"

# Function tool for Order fulfillment
fulfill_order_tool = FunctionTool(fulfill_order, description = "Fulfills a given Order")

# Agent that fulfills an order
fulfillment_agent = AssistantAgent(
    name="fulfillment_agent",
    description="A helpful Assistant that fulfills an order",
    model_client=model_client,
    tools=[fulfill_order_tool],
    system_message="You are a helpful AI assistant that fulfills an order. Use your tools to complete your task. You should do the task only if Inventory is available."
)

# Agent that processes an Order
orderprocess_agent = AssistantAgent(
    name="orderprocess_agent", 
    model_client=model_client, 
    description="A helpful Assistant that processes an order. This should be called first when a task is given. It will use other agents to process the order.", 
    system_message="""
    You are an AI Assistant that processes an order. It involves checking Inventory of the order.  
    Your team members are: 
        inventory_agent: Finds inventory details for a given order 
    When assigning tasks, use this format:
    1. <agent> : <task>
    Do not perform the subtasks yourself. 
    After all the sub-tasks are completed by the corresponding team members, you should generate a message that Order has been processeed and then say TERMINATE.
    """
    )

text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=25)

termination = text_mention_termination | max_messages_termination

#team = SelectorGroupChat([orderprocess_agent, fulfillment_agent, inventory_agent], model_client=model_client, termination_condition=termination, allow_repeated_speaker=False)
#team = SelectorGroupChat([orderprocess_agent, inventory_agent], model_client=model_client, termination_condition=termination, allow_repeated_speaker=False)
team = RoundRobinGroupChat([orderprocess_agent, inventory_agent], termination_condition=termination)
#team = RoundRobinGroupChat([orderprocess_agent, fulfillment_agent, inventory_agent], termination_condition=termination)

# Run the agent and stream the messages to the console.
async def main() -> None:
    await Console(team.run_stream(task=input("What do you want to ask?\n")))

# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
asyncio.run(main())

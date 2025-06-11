
import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_custom.agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def test_console() -> None:
    """
    Test the CustomConsole with a simple message stream.
    """

    agent = AssistantAgent(
        name="TestAgent",
        model_client=OpenAIChatCompletionClient(model="gpt-4o-mini"),
        system_message="You are a helpful assistant.",
    )

    # Create an instance of CustomConsole
    await Console(agent.run_stream(task="Say hello!"))


asyncio.run(test_console())

import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_custom.agentchat.ui import Console as CustomConsole
from autogen_custom.utils import patch_module
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Patch the Console class to use CustomConsole
patch_module("autogen_agentchat.ui", "Console", CustomConsole)

async def test_console() -> None:
    """
    Test patch_module with CustomConsole with a simple message stream.
    """
    from autogen_agentchat.ui import Console

    agent = AssistantAgent(
        name="TestAgent",
        model_client=OpenAIChatCompletionClient(model="gpt-4o-mini"),
        system_message="You are a helpful assistant.",
    )

    # Create an instance of CustomConsole
    await Console(agent.run_stream(task="Say hello!"))

asyncio.run(test_console())

import asyncio
import time
from typing import Any, AsyncGenerator, Awaitable, List, Optional, cast

from autogen_agentchat.base import Response, TaskResult
from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
    ModelClientStreamingChunkEvent,
    MultiModalMessage,
    UserInputRequestedEvent,
)
from autogen_agentchat.ui._console import (
    T,
    UserInputManager,
    _is_output_a_tty,
    _is_running_in_iterm,
)
from autogen_core.models import RequestUsage
from rich.console import Console as RichConsole
from rich.panel import Panel
from rich.text import Text


class CustomUserInputManager(UserInputManager):
    def notify_event_received(self, request_id: str) -> None:
        if request_id in self.input_events:
            self.input_events[request_id].set()
        else:
            event = asyncio.Event()
            event.set()  # Set immediately if not already waited on
            self.input_events[request_id] = event

def aprint(output: Any, **kwargs: Any) -> Awaitable[None]:
    """Asynchronously prints output using rich.console.print."""
    return asyncio.to_thread(RichConsole().print, output, **kwargs)


async def Console(
    stream: AsyncGenerator[BaseAgentEvent | BaseChatMessage | T, None],
    *,
    no_inline_images: bool = False,
    output_stats: bool = False,
    user_input_manager: CustomUserInputManager | None = None,
) -> T:
    """
    Consumes the message stream from :meth:`~autogen_agentchat.base.TaskRunner.run_stream`
    or :meth:`~autogen_agentchat.base.ChatAgent.on_messages_stream` and renders the messages to the console using Rich.
    Returns the last processed TaskResult or Response.

    .. note::

        `output_stats` is experimental and the stats may not be accurate.
        It will be improved in future releases.

    Args:
        stream (AsyncGenerator[BaseAgentEvent | BaseChatMessage | TaskResult, None] | AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]): Message stream to render.
        no_inline_images (bool, optional): If terminal is iTerm2 will render images inline. Use this to disable this behavior. Defaults to False.
        output_stats (bool, optional): (Experimental) If True, will output a summary of the messages and inline token usage info. Defaults to False.

    Returns:
        last_processed: A :class:`~autogen_agentchat.base.TaskResult` or :class:`~autogen_agentchat.base.Response`.
    """
    render_image_iterm = _is_running_in_iterm() and _is_output_a_tty() and not no_inline_images
    start_time = time.time()
    total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    last_processed: Optional[T] = None
    streaming_chunks: List[str] = []
    current_stream_header_printed = False

    async for message in stream:
        # Finalize any ongoing stream if the new message is not a chunk
        if streaming_chunks and not isinstance(message, ModelClientStreamingChunkEvent):
            await aprint("")  # Newline after streamed content
            streaming_chunks.clear()
            current_stream_header_printed = False

        if isinstance(message, TaskResult):
            duration = time.time() - start_time
            if output_stats:
                summary_content = (
                    f"Number of messages: {len(message.messages)}\n"
                    f"Finish reason: {message.stop_reason}\n"
                    f"Total prompt tokens: {total_usage.prompt_tokens}\n"
                    f"Total completion tokens: {total_usage.completion_tokens}\n"
                    f"Duration: {duration:.2f} seconds"
                )
                await aprint(Panel(Text(summary_content), title="[bold green]📊 Task Summary[/bold green]", expand=False, border_style="green"))

            last_processed = message  # type: ignore

        elif isinstance(message, Response):
            duration = time.time() - start_time

            # Print final response.
            if isinstance(message.chat_message, MultiModalMessage):
                final_content = message.chat_message.to_text(iterm=render_image_iterm)
            else:
                final_content = message.chat_message.to_text()

            response_text = Text(final_content)
            panel_title = f"[bold blue]💬 {message.chat_message.source}[/bold blue]"

            if message.chat_message.models_usage:
                if output_stats:
                    usage_info = f"\n[Prompt tokens: {message.chat_message.models_usage.prompt_tokens}, Completion tokens: {message.chat_message.models_usage.completion_tokens}]"
                    response_text.append(usage_info, style="dim")
                total_usage.completion_tokens += message.chat_message.models_usage.completion_tokens
                total_usage.prompt_tokens += message.chat_message.models_usage.prompt_tokens

            await aprint(Panel(response_text, title=panel_title, expand=False, border_style="blue"))

            # Print summary.
            if output_stats:
                if message.inner_messages is not None:
                    num_inner_messages = len(message.inner_messages)
                else:
                    num_inner_messages = 0
                summary_content = (
                    f"Number of inner messages: {num_inner_messages}\n"
                    f"Total prompt tokens: {total_usage.prompt_tokens}\n"
                    f"Total completion tokens: {total_usage.completion_tokens}\n"
                    f"Duration: {duration:.2f} seconds"
                )
                await aprint(Panel(Text(summary_content), title="[bold green]📊 Response Summary[/bold green]", expand=False, border_style="green"))

            last_processed = message  # type: ignore
        # We don't want to print UserInputRequestedEvent messages, we just use them to signal the user input event.
        elif isinstance(message, UserInputRequestedEvent):
            if user_input_manager is not None:
                user_input_manager.notify_event_received(message.request_id)
        else:
            # Cast required for mypy to be happy
            message = cast(BaseAgentEvent | BaseChatMessage, message)  # type: ignore

            if isinstance(message, ModelClientStreamingChunkEvent):
                if not current_stream_header_printed:
                    header_text = Text(f"🔄 Streaming from {message.source} ({message.__class__.__name__})", style="italic cyan")
                    await aprint(header_text)
                    current_stream_header_printed = True

                await aprint(message.to_text(), end="")
                streaming_chunks.append(message.content)
            else:
                # Handle non-streaming messages
                msg_content = ""
                if isinstance(message, MultiModalMessage):
                    msg_content = message.to_text(iterm=render_image_iterm)
                else:
                    msg_content = message.to_text()

                msg_text = Text(msg_content)
                panel_title = f"[cyan]🤖 {message.__class__.__name__} ({message.source})[/cyan]"

                if message.models_usage:
                    if output_stats:
                        usage_info = f"\n[Prompt tokens: {message.models_usage.prompt_tokens}, Completion tokens: {message.models_usage.completion_tokens}]"
                        msg_text.append(usage_info, style="dim")
                    total_usage.completion_tokens += message.models_usage.completion_tokens
                    total_usage.prompt_tokens += message.models_usage.prompt_tokens

                await aprint(Panel(msg_text, title=panel_title, expand=False, border_style="cyan"))

    # Ensure newline if loop ends mid-stream
    if streaming_chunks:
        await aprint("")

    if last_processed is None:
        raise ValueError("No TaskResult or Response was processed.")

    return last_processed

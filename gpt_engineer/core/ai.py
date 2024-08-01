"""
AI Module

This module provides an AI class that interfaces with language models to perform various tasks such as
starting a conversation, advancing the conversation, and handling message serialization. It also includes
backoff strategies for handling rate limit errors from the Llama API.

Classes:
    AI: A class that interfaces with language models for conversation management and message serialization.

Functions:
    serialize_messages(messages: List[Message]) -> str
        Serialize a list of messages to a JSON string.
"""

import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Union

import backoff
import pyperclip
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, HumanMessage, SystemMessage, messages_from_dict, messages_to_dict
from langchain_experimental.chat_models import Llama2Chat

from gpt_engineer.core.token_usage import TokenUsageLog

# Type hint for a chat message
Message = Union[AIMessage, HumanMessage, SystemMessage]

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class AI:
    """
    A class that interfaces with language models for conversation management and message serialization.

    This class provides methods to start and advance conversations, handle message serialization,
    and implement backoff strategies for rate limit errors when interacting with the Llama API.

    Attributes
    ----------
    temperature : float
        The temperature setting for the language model.
    azure_endpoint : str
        The endpoint URL for the Azure-hosted language model.
    model_name : str
        The name of the language model to use.
    streaming : bool
        A flag indicating whether to use streaming for the language model.
    llm : BaseChatModel
        The language model instance for conversation management.
    token_usage_log : TokenUsageLog
        A log for tracking token usage during conversations.

    Methods
    -------
    start(system: str, user: str, step_name: str) -> List[Message]
        Start the conversation with a system message and a user message.
    next(messages: List[Message], prompt: Optional[str], step_name: str) -> List[Message]
        Advances the conversation by sending message history to LLM and updating with the response.
    backoff_inference(messages: List[Message]) -> Any
        Perform inference using the language model with an exponential backoff strategy.
    serialize_messages(messages: List[Message]) -> str
        Serialize a list of messages to a JSON string.
    deserialize_messages(jsondictstr: str) -> List[Message]
        Deserialize a JSON string to a list of messages.
    _create_chat_model() -> BaseChatModel
        Create a chat model with the specified model name and temperature.
    """

    def __init__(
        self,
        model_name='llama-2-13b-chat', 
        temperature=0.1,
        azure_endpoint=None,
        streaming=True,
        vision=False,
    ):
        """
        Initialize the AI class.

        Parameters
        ----------
        model_name : str, optional
            The name of the model to use, by default "llama-2-13b-chat".
        temperature : float, optional
            The temperature to use for the model, by default 0.1.
        """
        self.temperature = temperature
        self.azure_endpoint = azure_endpoint
        self.model_name = model_name
        self.streaming = streaming
        self.vision = vision
        self.llm = self._create_chat_model()
        self.token_usage_log = TokenUsageLog(model_name)
        logger.debug(f"Using model {self.model_name}")

    def start(self, system: str, user: Any, *, step_name: str) -> List[Message]:
        """
        Start the conversation with a system message and a user message.

        Parameters
        ----------
        system : str
            The content of the system message.
        user : str
            The content of the user message.
        step_name : str
            The name of the step.

        Returns
        -------
        List[Message]
            The list of messages in the conversation.
        """

        messages: List[Message] = [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]
        return self.next(messages, step_name=step_name)

    def _extract_content(self, content):
        """
        Extracts text content from a message, supporting both string and list types.

        Parameters
        ----------
        content : Union[str, List[dict]]
            The content of a message, which could be a string or a list.

        Returns
        -------
        str
            The extracted text content.
        """
        if isinstance(content, str):
            return content
        elif isinstance(content, list) and content and "text" in content[0]:
            return content[0]["text"]
        else:
            return ""

    def _collapse_text_messages(self, messages: List[Message]):
        """
        Combine consecutive messages of the same type into a single message.

        This method iterates through the list of messages, combining consecutive messages of the same type
        by joining their content with a newline character. If the content is a list, it extracts text from the first
        text element's content. This reduces the number of messages and simplifies the conversation for processing.

        Parameters
        ----------
        messages : List[Message]
            The list of messages to collapse.

        Returns
        -------
        List[Message]
            The list of messages after collapsing consecutive messages of the same type.
        """
        collapsed_messages = []
        if not messages:
            return collapsed_messages

        previous_message = messages[0]
        combined_content = self._extract_content(previous_message.content)

        for current_message in messages[1:]:
            if current_message.type == previous_message.type:
                combined_content += "\n\n" + self._extract_content(current_message.content)
            else:
                collapsed_messages.append(previous_message.__class__(content=combined_content))
                previous_message = current_message
                combined_content = self._extract_content(current_message.content)

        collapsed_messages.append(previous_message.__class__(content=combined_content))
        return collapsed_messages

    def next(self, messages: List[Message], prompt: Optional[str] = None, *, step_name: str) -> List[Message]:
        """
        Advances the conversation by sending message history to LLM and updating with the response.

        Parameters
        ----------
        messages : List[Message]
            The list of messages in the conversation.
        prompt : Optional[str], optional
            The prompt to use, by default None.
        step_name : str
            The name of the step.

        Returns
        -------
        List[Message]
            The updated list of messages in the conversation.
        """

        if prompt:
            messages.append(HumanMessage(content=prompt))

        logger.debug("Creating a new chat completion: %s", "\n".join([m.pretty_repr() for m in messages]))

        if not self.vision:
            messages = self._collapse_text_messages(messages)

        response = self.backoff_inference(messages)

        self.token_usage_log.update_log(messages=messages, answer=response.content, step_name=step_name)
        messages.append(response)
        logger.debug(f"Chat completion finished: {messages}")

        return messages

    @backoff.on_exception(backoff.expo, Exception, max_tries=7, max_time=45)
    def backoff_inference(self, messages):
        """
        Perform inference using the language model with an exponential backoff strategy.

        Parameters
        ----------
        messages : List[Message]
            A list of chat messages which will be passed to the language model for processing.

        Returns
        -------
        Any
            The output from the language model after processing the provided messages.
        """
        return self.llm.invoke(messages)  # type: ignore

    @staticmethod
    def serialize_messages(messages: List[Message]) -> str:
        """
        Serialize a list of messages to a JSON string.

        Parameters
        ----------
        messages : List[Message]
            The list of messages to serialize.

        Returns
        -------
        str
            The serialized messages as a JSON string.
        """
        return json.dumps(messages_to_dict(messages))

    @staticmethod
    def deserialize_messages(jsondictstr: str) -> List[Message]:
        """
        Deserialize a JSON string to a list of messages.

        Parameters
        ----------
        jsondictstr : str
            The JSON string to deserialize.

        Returns
        -------
        List[Message]
            The deserialized list of messages.
        """
        data = json.loads(jsondictstr)
        prevalidated_data = [{**item, "tools": {**item.get("tools", {}), "is_chunk": False}} for item in data]
        return list(messages_from_dict(prevalidated_data))  # type: ignore

    def _create_chat_model(self) -> BaseChatModel:
        """
        Create a chat model with the specified model name and temperature.

        Parameters
        ----------
        model : str
            The name of the model to create.
        temperature : float
            The temperature to use for the model.

        Returns
        -------
        BaseChatModel
            The created chat model.
        """
        return Llama2Chat(
            model=self.model_name,
            max_tokens=None,
            timeout=None,
            max_retries=5,
            streaming=self.streaming,
            temperature=self.temperature,
            callbacks=[StreamingStdOutCallbackHandler()],
        )


def serialize_messages(messages: List[Message]) -> str:
    """
    Serialize a list of messages to a JSON string.

    Parameters
    ----------
    messages : List[Message]
        The list of messages to serialize.

    Returns
    -------
    str
        The serialized messages as a JSON string.
    """
    return AI.serialize_messages(messages)


class ClipboardAI(AI):
    """
    A class that extends the AI class to handle message input and output via the clipboard.

    This class provides methods to serialize and copy messages to the clipboard, as well as
    read multi-line input from the user for simpler scenarios or testing purposes.
    """

    def __init__(self, **_):  # type: ignore
        self.vision = False
        self.token_usage_log = TokenUsageLog("clipboard_llm")

    @staticmethod
    def serialize_messages(messages: List[Message]) -> str:
        """
        Serialize a list of messages to a text string with each message on a new line.

        Parameters
        ----------
        messages : List[Message]
            The list of messages to serialize.

        Returns
        -------
        str
            The serialized messages as a text string.
        """
        return "\n\n".join([f"{m.type}:\n{m.content}" for m in messages])

    @staticmethod
    def multiline_input():
        """
        Read multi-line input from the user.

        Returns
        -------
        str
            The input content from the user.
        """
        print("Enter/Paste your content. Ctrl-D or Ctrl-Z ( windows ) to save it.")
        content = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            content.append(line)
        return "\n".join(content)

    def next(self, messages: List[Message], prompt: Optional[str] = None, *, step_name: str) -> List[Message]:
        """
        Advances the conversation by copying the message history to the clipboard and waiting for user input.

        Parameters
        ----------
        messages : List[Message]
            The list of messages in the conversation.
        prompt : Optional[str], optional
            The prompt to use, by default None.
        step_name : str
            The name of the step.

        Returns
        -------
        List[Message]
            The updated list of messages in the conversation.
        """
        if prompt:
            messages.append(HumanMessage(content=prompt))

        logger.debug(f"Creating a new chat completion: {messages}")

        msgs = self.serialize_messages(messages)
        pyperclip.copy(msgs)
        Path("clipboard.txt").write_text(msgs)
        print("Messages copied to clipboard and written to clipboard.txt,", len(msgs), "characters in total")

        response = self.multiline_input()

        messages.append(AIMessage(content=response))
        logger.debug(f"Chat completion finished: {messages}")

        return messages

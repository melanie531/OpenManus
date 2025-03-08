from typing import Dict, List, Literal, Optional, Union
import json

from openai import (
    APIError,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
    AsyncAzureOpenAI
)
import boto3
from tenacity import retry, stop_after_attempt, wait_random_exponential

from app.config import LLMSettings, config
from app.logger import logger  # Assuming a logger is set up in your app
from app.schema import Message


class LLM:
    _instances: Dict[str, "LLM"] = {}

    def __new__(
        cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]

    def __init__(
        self, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if not hasattr(self, "client"):  # Only initialize if not already initialized
            llm_config = llm_config or config.llm
            llm_config = llm_config.get(config_name, llm_config["default"])
            self.model = llm_config.model
            self.max_tokens = llm_config.max_tokens
            self.temperature = llm_config.temperature
            self.api_type = llm_config.api_type
            self.api_key = llm_config.api_key
            self.api_version = llm_config.api_version
            self.base_url = llm_config.base_url
            self.region = llm_config.region
            self.profile = llm_config.profile

            if self.api_type == "azure":
                self.client = AsyncAzureOpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    api_version=self.api_version
                )
            elif self.api_type == "bedrock":
                session = boto3.Session(profile_name=self.profile) if self.profile else boto3.Session()
                self.client = session.client(
                    service_name='bedrock-runtime',
                    region_name=self.region
                )
            else:
                self.client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )

    @staticmethod
    def format_messages(messages: List[Union[dict, Message]]) -> List[dict]:
        """
        Format messages for LLM by converting them to OpenAI message format.

        Args:
            messages: List of messages that can be either dict or Message objects

        Returns:
            List[dict]: List of formatted messages in OpenAI format

        Raises:
            ValueError: If messages are invalid or missing required fields
            TypeError: If unsupported message types are provided

        Examples:
            >>> msgs = [
            ...     Message.system_message("You are a helpful assistant"),
            ...     {"role": "user", "content": "Hello"},
            ...     Message.user_message("How are you?")
            ... ]
            >>> formatted = LLM.format_messages(msgs)
        """
        formatted_messages = []

        for message in messages:
            if isinstance(message, dict):
                # If message is already a dict, ensure it has required fields
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")
                formatted_messages.append(message)
            elif isinstance(message, Message):
                # If message is a Message object, convert it to dict
                formatted_messages.append(message.to_dict())
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        # Validate all messages have required fields
        for msg in formatted_messages:
            if msg["role"] not in ["system", "user", "assistant", "tool"]:
                raise ValueError(f"Invalid role: {msg['role']}")
            if "content" not in msg and "tool_calls" not in msg:
                raise ValueError(
                    "Message must contain either 'content' or 'tool_calls'"
                )

        return formatted_messages

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        try:
            # Format system and user messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)

            if self.api_type == "bedrock":
                # Convert messages to Bedrock format for Claude
                formatted_messages = []
                for msg in messages:
                    if msg.get('role') == 'system':
                        formatted_messages.append({"role": "user", "content": f"System instruction: {msg['content']}"})
                    elif msg.get('role') == 'user':
                        formatted_messages.append({"role": "user", "content": msg['content']})
                    elif msg.get('role') == 'assistant':
                        formatted_messages.append({"role": "assistant", "content": msg['content']})
                    elif msg.get('role') == 'tool':
                        # Handle tool messages appropriately
                        formatted_messages.append({"role": "assistant", "content": f"Tool response: {msg['content']}"})

                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": formatted_messages,
                    "max_tokens": self.max_tokens,
                    "temperature": temperature or self.temperature,
                }

                request_body = {
                    "modelId": self.model,
                    "contentType": "application/json",
                    "accept": "application/json",
                    "body": json.dumps(body)
                }
                
                try:
                    response = self.client.invoke_model(**request_body)
                    response_body = json.loads(response.get('body').read())
                    
                    # Extract content from the response
                    if isinstance(response_body.get('content'), list):
                        # Handle list of content items
                        content = ''.join(item.get('text', '') for item in response_body.get('content', []) if item.get('type') == 'text')
                    else:
                        # Handle direct string content
                        content = response_body.get('content', '')
                    
                    return content
                except Exception as e:
                    logger.error(f"Bedrock API error: {str(e)}")
                    raise

            if not stream:
                # Non-streaming request for OpenAI/Azure
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=temperature or self.temperature,
                    stream=False,
                )
                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty or invalid response from LLM")
                return response.choices[0].message.content

            # Streaming request for OpenAI/Azure
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=temperature or self.temperature,
                stream=True,
            )

            collected_messages = []
            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                print(chunk_message, end="", flush=True)

            print()  # Newline after streaming
            full_response = "".join(collected_messages).strip()
            if not full_response:
                raise ValueError("Empty response from streaming LLM")
            return full_response

        except ValueError as ve:
            logger.error(f"Validation error: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask: {e}")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 60,
        tools: Optional[List[dict]] = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
        temperature: Optional[float] = None,
        **kwargs,
    ):
        try:
            # Validate tool_choice
            if tool_choice not in ["none", "auto", "required"]:
                raise ValueError(f"Invalid tool_choice: {tool_choice}")

            # Format messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)

            # Validate tools if provided
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError("Each tool must be a dict with 'type' field")

            if self.api_type == "bedrock":
                # Convert messages to Bedrock format for Claude
                formatted_messages = []
                for msg in messages:
                    if msg.get('role') == 'system':
                        formatted_messages.append({"role": "user", "content": f"System instruction: {msg['content']}"})
                    elif msg.get('role') == 'user':
                        formatted_messages.append({"role": "user", "content": msg['content']})
                    elif msg.get('role') == 'assistant':
                        formatted_messages.append({"role": "assistant", "content": msg['content']})
                    elif msg.get('role') == 'tool':
                        # Handle tool messages appropriately
                        formatted_messages.append({"role": "assistant", "content": f"Tool response: {msg['content']}"})

                # For Bedrock, we'll use a special prompt to handle tool-like behavior
                if tools:
                    tool_descriptions = []
                    for tool in tools:
                        if tool["type"] == "function":
                            function_info = tool["function"]
                            tool_descriptions.append(
                                f"Tool: {function_info['name']}\n"
                                f"Description: {function_info.get('description', '')}\n"
                                f"Parameters: {json.dumps(function_info.get('parameters', {}), indent=2)}\n"
                            )
                    
                    # Add tool descriptions to the last user message
                    last_msg = formatted_messages[-1]
                    if last_msg["role"] == "user":
                        last_msg["content"] = (
                            f"{last_msg['content']}\n\n"
                            f"Available tools:\n{''.join(tool_descriptions)}\n"
                            f"If you need to use a tool, format your response as:\n"
                            f"TOOL_CALL: <tool_name>\n"
                            f"ARGUMENTS: <json_arguments>"
                        )

                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": formatted_messages,
                    "max_tokens": self.max_tokens,
                    "temperature": temperature or self.temperature,
                }

                request_body = {
                    "modelId": self.model,
                    "contentType": "application/json",
                    "accept": "application/json",
                    "body": json.dumps(body)
                }
                
                try:
                    response = self.client.invoke_model(**request_body)
                    response_body = json.loads(response.get('body').read())
                    
                    # Extract content from the response
                    if isinstance(response_body.get('content'), list):
                        # Handle list of content items
                        content = ''.join(item.get('text', '') for item in response_body.get('content', []) if item.get('type') == 'text')
                    else:
                        # Handle direct string content
                        content = response_body.get('content', '')
                    
                    # Parse the response for tool calls
                    tool_calls = []
                    if "TOOL_CALL:" in content:
                        parts = content.split("TOOL_CALL:", 1)
                        pre_content = parts[0].strip()
                        tool_part = parts[1]
                        
                        # Try to extract tool name and arguments
                        try:
                            tool_lines = tool_part.strip().split("\n")
                            tool_name = tool_lines[0].strip()
                            args_text = tool_part.split("ARGUMENTS:", 1)[1].strip()
                            args = json.loads(args_text)
                            
                            tool_calls.append({
                                "id": "call_0",
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(args)
                                }
                            })
                            content = pre_content
                        except Exception as e:
                            logger.warning(f"Failed to parse tool call: {e}")
                    
                    # Create and return the Message object
                    return Message(
                        role="assistant",
                        content=content,
                        tool_calls=tool_calls if tool_calls else None
                    )
                except Exception as e:
                    logger.error(f"Bedrock API error: {str(e)}")
                    raise

            # Set up the completion request for OpenAI/Azure
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=self.max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                timeout=timeout,
                **kwargs,
            )

            # Check if response is valid
            if not response.choices or not response.choices[0].message:
                print(response)
                raise ValueError("Invalid or empty response from LLM")

            return response.choices[0].message

        except ValueError as ve:
            logger.error(f"Validation error in ask_tool: {ve}")
            raise
        except OpenAIError as oe:
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_tool: {e}")
            raise

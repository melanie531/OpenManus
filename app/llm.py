from typing import Dict, List, Literal, Optional, Union
import json
import re

from openai import (
    APIError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from tenacity import retry, stop_after_attempt, wait_random_exponential
import boto3

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
                if not self.api_key:
                    raise ValueError("Azure API key must be provided")
                self.client = AsyncAzureOpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    api_version=self.api_version,
                )
            elif self.api_type == "bedrock":
                session = boto3.Session(profile_name=self.profile) if self.profile else boto3.Session()
                self.client = session.client(
                    service_name='bedrock-runtime',
                    region_name=self.region
                )
            elif self.api_type == "openai":
                if not self.api_key:
                    raise ValueError("OpenAI API key must be provided")
                self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
            else:
                raise ValueError(f"Unsupported API type: {self.api_type}")

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
        """
        Send a prompt to the LLM and get the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response

        Returns:
            str: The generated response

        Raises:
            ValueError: If messages are invalid or response is empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
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
                    logger.debug(f"Raw Bedrock response: {json.dumps(response_body, indent=2)}")
                    
                    # Extract content from the response
                    if isinstance(response_body.get('content'), list):
                        # Handle list of content items
                        content = ''.join(item.get('text', '') for item in response_body.get('content', []) if isinstance(item, dict) and item.get('type') == 'text')
                    else:
                        # Handle direct string content
                        content = str(response_body.get('content', ''))
                    
                    if not content:
                        raise ValueError("Empty response from Bedrock")
                    return content
                except Exception as e:
                    logger.error(f"Bedrock API error: {str(e)}")
                    raise

            # OpenAI/Azure handling
            if not stream:
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
        """
        Ask LLM using functions/tools and return the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            timeout: Request timeout in seconds
            tools: List of tools to use
            tool_choice: Tool choice strategy
            temperature: Sampling temperature for the response
            **kwargs: Additional completion arguments

        Returns:
            ChatCompletionMessage: The model's response

        Raises:
            ValueError: If tools, tool_choice, or messages are invalid
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
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
                        formatted_messages.append({"role": "assistant", "content": f"Tool response: {msg['content']}"})

                # Add tool descriptions to the last user message
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
                    logger.debug(f"Raw Bedrock response: {json.dumps(response_body, indent=2)}")
                    
                    # Extract content from the response
                    content = ""
                    if isinstance(response_body.get('content'), list):
                        # Handle list of content items
                        content = ''.join(item.get('text', '') for item in response_body.get('content', []) if isinstance(item, dict) and item.get('type') == 'text')
                    else:
                        # Handle direct string content
                        content = str(response_body.get('content', ''))
                    
                    if not content:
                        logger.warning("No content extracted from Bedrock response")
                        content = "No response content available"
                    
                    # Parse the response for tool calls
                    tool_calls = []
                    if "TOOL_CALL:" in content:
                        parts = content.split("TOOL_CALL:")
                        pre_content = parts[0].strip()
                        
                        # Process each tool call
                        for i, tool_part in enumerate(parts[1:], 1):
                            try:
                                # Split into lines for processing
                                lines = [line.strip() for line in tool_part.strip().split("\n") if line.strip()]
                                if not lines:
                                    continue
                                
                                # First line is the tool name
                                tool_name = lines[0].strip()
                                
                                # Find ARGUMENTS: section and collect all argument lines
                                args_lines = []
                                in_args = False
                                for line in lines[1:]:  # Skip the tool name line
                                    if "ARGUMENTS:" in line:
                                        in_args = True
                                        # Get the part after ARGUMENTS:
                                        args_start = line.index("ARGUMENTS:") + len("ARGUMENTS:")
                                        args_line = line[args_start:].strip()
                                        if args_line:  # Only add if there's content after ARGUMENTS:
                                            args_lines.append(args_line)
                                    elif in_args and not line.startswith("TOOL_CALL:"):
                                        args_lines.append(line)
                                    elif line.startswith("TOOL_CALL:"):
                                        break  # Stop at next tool call
                                
                                if args_lines:
                                    # Join all lines and clean up the text
                                    args_text = " ".join(args_lines)
                                    args_text = args_text.strip()
                                    
                                    # Try to parse JSON arguments with fallback strategies
                                    args = self._parse_json_arguments(args_text)
                                    logger.debug(f"type of args: {type(args)}")
                                    if args:
                                        logger.debug(f"Successfully parsed tool {tool_name} arguments: {args}")
                                        
                                        tool_calls.append({
                                            "id": f"call_{len(tool_calls)}",
                                            "type": "function",
                                            "function": {
                                                "name": tool_name,
                                                "arguments": json.dumps(args)
                                            }
                                        })
                                    else:
                                        logger.warning(f"Failed to parse arguments for tool {tool_name}")
                                        logger.debug(f"Problematic JSON string: {args_text}")
                                else:
                                    logger.warning(f"No ARGUMENTS section found for tool {tool_name}")
                            except Exception as e:
                                logger.error(f"Failed to parse tool call: {str(e)}")
                                logger.debug(f"Tool part causing error: {tool_part}")
                        
                        # Set content to everything before the first tool call
                        content = pre_content
                    
                    logger.debug(f"Final content: {content}")
                    logger.debug(f"Final tool calls: {tool_calls}")
                    
                    # Create and return the Message object
                    return Message(
                        role="assistant",
                        content=content if not tool_calls else "",  # Empty content if we have tool calls
                        tool_calls=tool_calls if tool_calls else None
                    )
                except Exception as e:
                    logger.error(f"Bedrock API error: {str(e)}")
                    raise

            # OpenAI/Azure handling
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

    @staticmethod
    def _clean_json_text(text: str) -> str:
        """Clean and prepare JSON text for parsing."""
        # First handle Python string formatting
        def handle_datetime_format(match):
            from datetime import datetime
            format_str = match.group(1) if match.group(1) else "%Y-%m-%d"
            return datetime.now().strftime(format_str)

        # Replace datetime.now() format strings
        text = re.sub(r'datetime\.now\(\)\.strftime\(["\']([^"\']*)["\']?\)', handle_datetime_format, text)
        text = re.sub(r'datetime\.now\(\)', lambda x: f'"{datetime.now().strftime("%Y-%m-%d")}"', text)
        
        # Handle string format() calls
        def handle_format_call(match):
            content = match.group(1)
            # If there's a format() call, evaluate it
            if content.endswith(".format(datetime.now().strftime('%Y-%m-%d'))"):
                from datetime import datetime
                content = content[:-len(".format(datetime.now().strftime('%Y-%m-%d'))")].replace("{}", datetime.now().strftime("%Y-%m-%d"))
            return content

        # Replace format() calls
        text = re.sub(r'"([^"]*?{.*?}.*?\.format\([^)]*\))"', handle_format_call, text)
        
        # Handle triple-quoted Python code blocks
        def handle_triple_quotes(match):
            code = match.group(1)
            # Handle any datetime formatting in the code
            code = re.sub(r'datetime\.now\(\)\.strftime\(["\']([^"\']*)["\']?\)', handle_datetime_format, code)
            code = re.sub(r'datetime\.now\(\)', lambda x: f'"{datetime.now().strftime("%Y-%m-%d")}"', code)
            # Handle string formatting
            if "{}" in code and ".format(" in code:
                code = handle_format_call(f'"{code}"')
            # Escape quotes and preserve newlines
            code = code.replace('"', '\\"')
            return f'"{code}"'
        
        # Replace triple-quoted blocks with properly escaped strings
        text = re.sub(r'"""(.*?)"""', handle_triple_quotes, text, flags=re.DOTALL)
        
        # Remove any trailing text after the last closing brace
        last_brace = text.rfind('}')
        if last_brace >= 0:
            text = text[:last_brace + 1]
        
        # Remove any text before the first opening brace
        first_brace = text.find('{')
        if first_brace >= 0:
            text = text[first_brace:]
        
        # Fix common JSON formatting issues
        text = text.replace("'", '"')  # Replace single quotes with double quotes
        text = re.sub(r'([{,])\s*(\w+):', r'\1"\2":', text)  # Quote unquoted keys
        
        return text.strip()

    @staticmethod
    def _extract_json_object(text: str) -> Optional[str]:
        """Extract the first valid JSON object from text."""
        # First handle triple-quoted Python code blocks
        def handle_triple_quotes(match):
            code = match.group(1)
            # Preserve newlines but escape quotes
            code = code.replace('"', '\\"')
            # Join lines with explicit \n to preserve formatting
            return f'"{code}"'
        
        # Replace triple-quoted blocks with properly escaped strings
        text = re.sub(r'"""(.*?)"""', handle_triple_quotes, text, flags=re.DOTALL)
        
        stack = []
        in_string = False
        escape = False
        start = -1
        
        for i, char in enumerate(text):
            if char == '\\' and not escape:
                escape = True
                continue
                
            if char == '"' and not escape:
                in_string = not in_string
                
            if not in_string and not escape:
                if char == '{':
                    if not stack:  # First opening brace
                        start = i
                    stack.append(char)
                elif char == '}':
                    if stack:
                        stack.pop()
                        if not stack and start >= 0:  # Found complete object
                            return text[start:i + 1]
                            
            escape = False
        
        return None

    def _parse_json_arguments(self, args_text: str) -> Optional[dict]:
        """Parse JSON arguments with multiple fallback strategies."""
        def process_code_value(obj):
            """Process code values in the parsed JSON object."""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str):
                        # Handle any remaining format strings or datetime calls
                        if "{}" in value and ".format(" in value:
                            from datetime import datetime
                            value = value.replace("{}", datetime.now().strftime("%Y-%m-%d"))
                            value = re.sub(r'\.format\([^)]*\)', '', value)
                        if key == "code":
                            # Remove any leading/trailing quotes
                            code = value.strip('"\'')
                            # Unescape any escaped quotes
                            code = code.replace('\\"', '"')
                            # Split into lines and process
                            lines = code.split('\\n')
                            # Remove common leading whitespace
                            if lines:
                                # Find minimum indentation from non-empty lines
                                min_indent = float('inf')
                                for line in lines:
                                    if line.strip():
                                        indent = len(line) - len(line.lstrip())
                                        min_indent = min(min_indent, indent) if indent < min_indent else min_indent
                                if min_indent == float('inf'):
                                    min_indent = 0
                                # Remove common indentation
                                lines = [line[min_indent:] if line.strip() else '' for line in lines]
                                # Remove leading/trailing empty lines
                                while lines and not lines[0].strip():
                                    lines.pop(0)
                                while lines and not lines[-1].strip():
                                    lines.pop()
                                # Join lines back together
                                value = '\n'.join(lines)
                        obj[key] = value
                    elif isinstance(value, (dict, list)):
                        process_code_value(value)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        process_code_value(item)
            return obj

        try:
            # First attempt: direct parse
            args = json.loads(args_text)
            return process_code_value(args)
        except json.JSONDecodeError:
            try:
                # Second attempt: clean and parse
                cleaned_text = self._clean_json_text(args_text)
                args = json.loads(cleaned_text)
                return process_code_value(args)
            except json.JSONDecodeError:
                try:
                    # Third attempt: extract and parse first JSON object
                    json_obj = self._extract_json_object(args_text)
                    if json_obj:
                        args = json.loads(json_obj)
                        return process_code_value(args)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON arguments: {args_text}")
                    return None

import math
from typing import Dict, List, Optional, Union

import tiktoken
from openai import (
    APIError as OpenAIAPIError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AuthenticationError as OpenAIAuthenticationError,
    OpenAIError,
    RateLimitError as OpenAIRateLimitError,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessage as OpenAIChatCompletionMessage
from anthropic import (
    AsyncAnthropic, 
    AnthropicError, # Base error
    APIError as AnthropicAPIError, # Specific aliased import
    AuthenticationError as AnthropicAuthenticationError, # Specific aliased import
    RateLimitError as AnthropicRateLimitError # Specific aliased import
)
from anthropic.types import MessageParam as AnthropicMessageParam
from anthropic.types import MessageStreamEvent, TextBlockParam

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from .bedrock import BedrockClient
from .config import LLMSettings, config
from .exceptions import TokenLimitExceeded
from .logger import logger
from .schema import (
    ROLE_VALUES,
    TOOL_CHOICE_TYPE,
    TOOL_CHOICE_VALUES,
    Message,
    ToolChoice,
    ToolCall
)


REASONING_MODELS = ["o1", "o3-mini"]
MULTIMODAL_MODELS = [
    "gpt-4-vision-preview",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]


class TokenCounter:
    # Token constants
    BASE_MESSAGE_TOKENS = 4
    FORMAT_TOKENS = 2
    LOW_DETAIL_IMAGE_TOKENS = 85
    HIGH_DETAIL_TILE_TOKENS = 170

    # Image processing constants
    MAX_SIZE = 2048
    HIGH_DETAIL_TARGET_SHORT_SIDE = 768
    TILE_SIZE = 512

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def count_text(self, text: str) -> int:
        """Calculate tokens for a text string"""
        return 0 if not text else len(self.tokenizer.encode(text))

    def count_image(self, image_item: dict) -> int:
        """
        Calculate tokens for an image based on detail level and dimensions

        For "low" detail: fixed 85 tokens
        For "high" detail:
        1. Scale to fit in 2048x2048 square
        2. Scale shortest side to 768px
        3. Count 512px tiles (170 tokens each)
        4. Add 85 tokens
        """
        detail = image_item.get("detail", "medium")

        # For low detail, always return fixed token count
        if detail == "low":
            return self.LOW_DETAIL_IMAGE_TOKENS

        # For medium detail (default in OpenAI), use high detail calculation
        # OpenAI doesn't specify a separate calculation for medium

        # For high detail, calculate based on dimensions if available
        if detail == "high" or detail == "medium":
            # If dimensions are provided in the image_item
            if "dimensions" in image_item:
                width, height = image_item["dimensions"]
                return self._calculate_high_detail_tokens(width, height)

        return (
            self._calculate_high_detail_tokens(1024, 1024) if detail == "high" else 1024
        )

    def _calculate_high_detail_tokens(self, width: int, height: int) -> int:
        """Calculate tokens for high detail images based on dimensions"""
        # Step 1: Scale to fit in MAX_SIZE x MAX_SIZE square
        if width > self.MAX_SIZE or height > self.MAX_SIZE:
            scale = self.MAX_SIZE / max(width, height)
            width = int(width * scale)
            height = int(height * scale)

        # Step 2: Scale so shortest side is HIGH_DETAIL_TARGET_SHORT_SIDE
        scale = self.HIGH_DETAIL_TARGET_SHORT_SIDE / min(width, height)
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)

        # Step 3: Count number of 512px tiles
        tiles_x = math.ceil(scaled_width / self.TILE_SIZE)
        tiles_y = math.ceil(scaled_height / self.TILE_SIZE)
        total_tiles = tiles_x * tiles_y

        # Step 4: Calculate final token count
        return (
            total_tiles * self.HIGH_DETAIL_TILE_TOKENS
        ) + self.LOW_DETAIL_IMAGE_TOKENS

    def count_content(self, content: Union[str, List[Union[str, dict]]]) -> int:
        """Calculate tokens for message content"""
        if not content:
            return 0

        if isinstance(content, str):
            return self.count_text(content)

        token_count = 0
        for item in content:
            if isinstance(item, str):
                token_count += self.count_text(item)
            elif isinstance(item, dict):
                if "text" in item:
                    token_count += self.count_text(item["text"])
                elif "image_url" in item:
                    token_count += self.count_image(item)
        return token_count

    def count_tool_calls(self, tool_calls: List[dict]) -> int:
        """Calculate tokens for tool calls"""
        token_count = 0
        for tool_call in tool_calls:
            if "function" in tool_call:
                function = tool_call["function"]
                token_count += self.count_text(function.get("name", ""))
                token_count += self.count_text(function.get("arguments", ""))
        return token_count

    def count_message_tokens(self, messages: List[dict]) -> int:
        """Calculate the total number of tokens in a message list"""
        total_tokens = self.FORMAT_TOKENS  # Base format tokens

        for message in messages:
            tokens = self.BASE_MESSAGE_TOKENS  # Base tokens per message

            # Add role tokens
            tokens += self.count_text(message.get("role", ""))

            # Add content tokens
            if "content" in message:
                tokens += self.count_content(message["content"])

            # Add tool calls tokens
            if "tool_calls" in message:
                tokens += self.count_tool_calls(message["tool_calls"])

            # Add name and tool_call_id tokens
            tokens += self.count_text(message.get("name", ""))
            tokens += self.count_text(message.get("tool_call_id", ""))

            total_tokens += tokens

        return total_tokens


class LLM:
    _instances: Dict[str, "LLM"] = {}
    # Anthropic specific headers
    ANTHROPIC_DEFAULT_HEADERS = {"anthropic-version": "2023-06-01"}

    def __new__(cls, config_name: str = "default", llm_config_override: Optional[LLMSettings] = None):
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            # Call __init__ only once per instance name, passing the override
            instance._initialized_llm = False # Flag to ensure __init__ runs once effectively
            instance.__init__(config_name, llm_config_override)
            cls._instances[config_name] = instance
        return cls._instances[config_name]

    def __init__(self, config_name: str = "default", llm_config_override: Optional[LLMSettings] = None):
        if hasattr(self, '_initialized_llm') and self._initialized_llm:
            return

        current_llm_config = llm_config_override or config.llm.get(config_name, config.llm["default"])
        
        self.model = current_llm_config.model
        self.max_tokens = current_llm_config.max_tokens
        self.temperature = current_llm_config.temperature
        self.api_type = current_llm_config.api_type
        self.api_key = current_llm_config.api_key # This should now be resolved by Config class
        self.api_version = current_llm_config.api_version
        self.base_url = current_llm_config.base_url

        self.total_input_tokens = 0
        self.total_completion_tokens = 0
        self.max_input_tokens = current_llm_config.max_input_tokens

        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        if self.api_type == "azure":
            self.client = AsyncAzureOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                api_version=self.api_version,
            )
        elif self.api_type == "aws":
            self.client = BedrockClient()
        elif self.api_type == "anthropic": # Added Anthropic client initialization
            self.client = AsyncAnthropic(
                api_key=self.api_key,
                base_url=self.base_url # base_url might not be used by anthropic client if it defaults correctly
            )
        else: # Defaults to OpenAI compatible
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

        self.token_counter = TokenCounter(self.tokenizer)
        self._initialized_llm = True # Mark as initialized

    def count_tokens(self, text: str) -> int:
        """Calculate the number of tokens in a text"""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def count_message_tokens(self, messages: List[dict]) -> int:
        return self.token_counter.count_message_tokens(messages)

    def update_token_count(self, input_tokens: int, completion_tokens: int = 0) -> None:
        """Update token counts"""
        # Only track tokens if max_input_tokens is set
        self.total_input_tokens += input_tokens
        self.total_completion_tokens += completion_tokens
        logger.info(
            f"Token usage: Input={input_tokens}, Completion={completion_tokens}, "
            f"Cumulative Input={self.total_input_tokens}, Cumulative Completion={self.total_completion_tokens}, "
            f"Total={input_tokens + completion_tokens}, Cumulative Total={self.total_input_tokens + self.total_completion_tokens}"
        )

    def check_token_limit(self, input_tokens: int) -> bool:
        """Check if token limits are exceeded"""
        if self.max_input_tokens is not None:
            return (self.total_input_tokens + input_tokens) <= self.max_input_tokens
        # If max_input_tokens is not set, always return True
        return True

    def get_limit_error_message(self, input_tokens: int) -> str:
        """Generate error message for token limit exceeded"""
        if (
            self.max_input_tokens is not None
            and (self.total_input_tokens + input_tokens) > self.max_input_tokens
        ):
            return f"Request may exceed input token limit (Current: {self.total_input_tokens}, Needed: {input_tokens}, Max: {self.max_input_tokens})"

        return "Token limit exceeded"

    @staticmethod
    def _format_openai_messages(messages: List[Union[dict, Message]], supports_images: bool = False) -> List[dict]:
        # This existing method is for OpenAI, renamed for clarity
        formatted_messages = []
        for message_item in messages:
            message_dict = message_item.to_dict() if isinstance(message_item, Message) else message_item
            if "role" not in message_dict: raise ValueError("Message dict must contain 'role' field")
            if supports_images and message_dict.get("base64_image"):
                content = message_dict.get("content", [])
                if isinstance(content, str): content = [{"type": "text", "text": content}]
                elif isinstance(content, list):
                    content = [{"type": "text", "text": item} if isinstance(item, str) else item for item in content]
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{message_dict['base64_image']}"}})
                message_dict["content"] = content
                del message_dict["base64_image"]
            elif not supports_images and message_dict.get("base64_image"): del message_dict["base64_image"]
            if "content" in message_dict or "tool_calls" in message_dict: formatted_messages.append(message_dict)
        for msg in formatted_messages: 
            if msg["role"] not in ROLE_VALUES: raise ValueError(f"Invalid role: {msg['role']}")
        return formatted_messages

    @staticmethod
    def _format_anthropic_messages(messages: List[Union[dict, Message]], supports_images: bool = False) -> List[AnthropicMessageParam]:
        anthropic_messages: List[AnthropicMessageParam] = []
        for message_item in messages:
            message_dict = message_item.to_dict() if isinstance(message_item, Message) else message_item
            role = message_dict.get("role")
            content = message_dict.get("content") # Can be string or list for multimodal
            
            # Anthropic expects user/assistant. System prompt is handled separately.
            # Tool calls and results also have specific formats.
            if role not in ["user", "assistant"]:
                logger.debug(f"Skipping message with role '{role}' for Anthropic messages list.")
                continue

            # Content conversion for Anthropic (text or list of content blocks)
            anthropic_content: Union[str, List[Dict[str, Any]]] = []
            if isinstance(content, str):
                anthropic_content.append({"type": "text", "text": content})
            elif isinstance(content, list): # OpenAI multimodal format
                for item in content:
                    if isinstance(item, str): # Should not happen if properly formatted before
                        anthropic_content.append({"type": "text", "text": item})
                    elif isinstance(item, dict):
                        if item.get("type") == "text":
                            anthropic_content.append({"type": "text", "text": item.get("text", "")})
                        elif item.get("type") == "image_url" and supports_images:
                            base64_data = item["image_url"]["url"].split(",")[-1]
                            media_type = item["image_url"]["url"].split(";")[0].split(":")[-1]
                            anthropic_content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type, # e.g., "image/jpeg"
                                    "data": base64_data,
                                }
                            })
            # If message has base64_image directly (OpenManus specific)
            elif message_dict.get("base64_image") and supports_images:
                anthropic_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg", # Assuming jpeg, could be improved
                        "data": message_dict["base64_image"],
                    }
                })
                if isinstance(content, str) and not any(c.get("type") == "text" for c in anthropic_content):
                     anthropic_content.insert(0, {"type": "text", "text": content})
            
            # Handle tool_calls (from assistant) and tool_results (from user, mapped from OpenManus 'tool' role)
            # This part needs careful mapping if/when tool use is fully implemented for Anthropic.
            # For now, basic message content is prioritized.

            if not anthropic_content:
                 if isinstance(content, str): # Fallback for simple string content if list processing yielded empty
                    anthropic_content = content
                 else:
                    logger.warning(f"Message content for role '{role}' could not be formatted for Anthropic: {message_dict.get('content')}")
                    continue # Skip message if content is not formattable to text or image block

            anthropic_messages.append(AnthropicMessageParam(role=role, content=anthropic_content))
        return anthropic_messages

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type((OpenAIError, AnthropicError, Exception, ValueError))
    )
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        try:
            supports_images = self.model in MULTIMODAL_MODELS
            input_tokens = 0 # Will be calculated based on client type

            if isinstance(self.client, AsyncAnthropic):
                system_prompt_str = ""
                if system_msgs:
                    # Concatenate system messages content for Anthropic
                    for msg_item in system_msgs:
                        msg_dict = msg_item.to_dict() if isinstance(msg_item, Message) else msg_item
                        if msg_dict.get("role") == "system" and msg_dict.get("content"):
                            system_prompt_str += str(msg_dict.get("content")) + "\n"
                system_prompt_str = system_prompt_str.strip()
                
                formatted_anth_messages = self._format_anthropic_messages(messages, supports_images)
                # Anthropic token counting is complex, using a rough estimate or relying on API errors for now.
                # For simplicity, we'll estimate based on combined text for the check_token_limit.
                temp_text_for_token_check = system_prompt_str + " ".join([str(m.get("content", "")) for m in formatted_anth_messages])
                input_tokens = self.count_tokens(temp_text_for_token_check) 

                if not self.check_token_limit(input_tokens):
                    raise TokenLimitExceeded(self.get_limit_error_message(input_tokens))
                
                self.update_token_count(input_tokens) # Update before call for streaming

                if stream:
                    completion_text = ""
                    async with self.client.messages.stream(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        temperature=temperature if temperature is not None else self.temperature,
                        messages=formatted_anth_messages,
                        system=system_prompt_str if system_prompt_str else None,
                        extra_headers=self.ANTHROPIC_DEFAULT_HEADERS
                    ) as stream_obj:
                        async for event in stream_obj:
                            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                                chunk_text = event.delta.text
                                print(chunk_text, end="", flush=True)
                                completion_text += chunk_text
                    print() # Newline after streaming
                    if not completion_text:
                        # Check for error in stream_obj if possible or just raise
                        # final_message = await stream_obj.get_final_message() # if this was available
                        # if final_message.stop_reason == "error": raise ValueError("Anthropic stream ended with error")
                        raise ValueError("Empty response from Anthropic streaming LLM")
                    self.total_completion_tokens += self.count_tokens(completion_text)
                    return completion_text
                else: # Non-streaming Anthropic
                    response = await self.client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        temperature=temperature if temperature is not None else self.temperature,
                        messages=formatted_anth_messages,
                        system=system_prompt_str if system_prompt_str else None,
                        extra_headers=self.ANTHROPIC_DEFAULT_HEADERS
                    )
                    # Assuming response.content is a list of blocks, and we want the text from the first text block
                    response_text = ""
                    if response.content:
                        for block in response.content:
                            if hasattr(block, 'text'): # Check if block is TextBlock
                                response_text += block.text
                    if not response_text: raise ValueError("Empty or invalid response from Anthropic LLM")
                    # Anthropic API v1 returns usage in response.usage
                    self.update_token_count(response.usage.input_tokens, response.usage.output_tokens)
                    return response_text

            else: # OpenAI or compatible client
                formatted_oa_messages = self._format_openai_messages(messages, supports_images)
                if system_msgs:
                    formatted_oa_system_msgs = self._format_openai_messages(system_msgs, supports_images)
                    all_oa_messages = formatted_oa_system_msgs + formatted_oa_messages
                else:
                    all_oa_messages = formatted_oa_messages
                
                input_tokens = self.count_message_tokens(all_oa_messages)
                if not self.check_token_limit(input_tokens):
                    raise TokenLimitExceeded(self.get_limit_error_message(input_tokens))

                params = {"model": self.model, "messages": all_oa_messages}
                if self.model in REASONING_MODELS: params["max_completion_tokens"] = self.max_tokens # OpenAI specific field
                else: params["max_tokens"] = self.max_tokens
                params["temperature"] = temperature if temperature is not None else self.temperature

                if not stream:
                    response = await self.client.chat.completions.create(**params, stream=False)
                    if not response.choices or not response.choices[0].message.content:
                        raise ValueError("Empty or invalid response from LLM")
                    self.update_token_count(response.usage.prompt_tokens, response.usage.completion_tokens)
                    return response.choices[0].message.content
                else: # OpenAI Streaming
                    self.update_token_count(input_tokens)
                    response_stream = await self.client.chat.completions.create(**params, stream=True)
                    collected_messages_text = []
                    completion_text = ""
                    async for chunk in response_stream:
                        chunk_content = chunk.choices[0].delta.content or ""
                        collected_messages_text.append(chunk_content)
                        completion_text += chunk_content
                        print(chunk_content, end="", flush=True)
                    print()
                    full_response = "".join(collected_messages_text).strip()
                    if not full_response: raise ValueError("Empty response from streaming LLM")
                    self.total_completion_tokens += self.count_tokens(completion_text)
                    return full_response

        except TokenLimitExceeded: raise
        except (ValueError, AnthropicError, OpenAIError) as e:
            logger.exception(f"LLM API error in ask method: {e}")
            logger.info("DEBUG: ask - In except (ValueError, AnthropicError, OpenAIError)")
            if isinstance(e, OpenAIAuthenticationError) or isinstance(e, AnthropicAuthenticationError):
                logger.error("Authentication failed. Check API key and client configuration.")
            elif isinstance(e, OpenAIRateLimitError) or isinstance(e, AnthropicRateLimitError):
                logger.error("Rate limit exceeded.")
            elif isinstance(e, OpenAIAPIError) or isinstance(e, AnthropicAPIError):
                 logger.error(f"Generic API error: {e}")
                 logger.info(f"DEBUG: ask - Caught APIError type: {type(e)}")
            raise
        except Exception: logger.exception(f"Unexpected error in ask"); raise

    @retry(
        wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6),
        retry=retry_if_exception_type((OpenAIError, AnthropicError, Exception, ValueError))
    )
    async def ask_with_images(
        self,
        messages: List[Union[dict, Message]],
        images: List[Union[str, dict]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        if self.model not in MULTIMODAL_MODELS:
            raise ValueError(f"Model {self.model} does not support images. Use a model from {MULTIMODAL_MODELS}")
        
        logger.info(f"ask_with_images called. Client type: {type(self.client)}")
        if isinstance(self.client, AsyncAnthropic):
            # Simplified Anthropic image handling - assumes images are base64 strings provided in `images` list
            # A more robust implementation would handle URLs by fetching them.
            # Anthropic expects image content blocks directly in the messages.
            system_prompt_str = ""
            if system_msgs:
                for msg_item in system_msgs:
                    msg_dict = msg_item.to_dict() if isinstance(msg_item, Message) else msg_item
                    if msg_dict.get("role") == "system" and msg_dict.get("content"):
                        system_prompt_str += str(msg_dict.get("content")) + "\n"
            system_prompt_str = system_prompt_str.strip()

            formatted_anth_messages = self._format_anthropic_messages(messages, supports_images=True)
            
            # Find the last user message to append images to its content list
            last_user_message_index = -1
            for i in range(len(formatted_anth_messages) - 1, -1, -1):
                if formatted_anth_messages[i]["role"] == "user":
                    last_user_message_index = i
                    break
            
            if last_user_message_index == -1:
                 # If no user message, create one or decide on error handling.
                 # For now, let's assume the prompt implies a user message and add to the last message if it exists.
                 if not formatted_anth_messages: # Create a dummy user message if none exist
                     formatted_anth_messages.append(AnthropicMessageParam(role="user", content=[]))
                 last_user_message_index = len(formatted_anth_messages) -1
            
            # Ensure content is a list for the target message
            if isinstance(formatted_anth_messages[last_user_message_index]["content"], str):
                 formatted_anth_messages[last_user_message_index]["content"] = [{"type": "text", "text": formatted_anth_messages[last_user_message_index]["content"]}]
            elif not isinstance(formatted_anth_messages[last_user_message_index]["content"], list):
                 formatted_anth_messages[last_user_message_index]["content"] = []

            for image_data in images: # Assuming images are base64 strings or dicts with url/base64 for simplicity
                if isinstance(image_data, str): # Assuming base64 string
                    formatted_anth_messages[last_user_message_index]["content"].append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}
                    })
                elif isinstance(image_data, dict) and "url" in image_data.get("image_url", {}):
                    # For URLs, a robust solution would fetch and base64 encode.
                    # Here, we assume direct base64 if not a simple string for this example pass-through.
                    # This part needs careful real-world implementation of URL fetching & type detection.
                    img_url_data = image_data["image_url"]["url"]
                    if img_url_data.startswith("data:"):
                        media_type = img_url_data.split(";")[0].split(":")[-1]
                        base64_str = img_url_data.split(",")[-1]
                        formatted_anth_messages[last_user_message_index]["content"].append({
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": base64_str}
                        })
                    else:
                        logger.warning(f"Image URL handling not fully implemented for Anthropic in this example: {img_url_data}")
                        # Potentially add as text or skip
                        formatted_anth_messages[last_user_message_index]["content"].append({"type": "text", "text": f"[Image URL: {img_url_data}]"})
                else:
                    logger.warning(f"Unsupported image data format for Anthropic: {image_data}")
            
            # (Token counting and limit checks would go here, similar to .ask())
            temp_text_for_token_check = system_prompt_str + " ".join([str(m.get("content", "")) for m in formatted_anth_messages])
            input_tokens = self.count_tokens(temp_text_for_token_check)
            if not self.check_token_limit(input_tokens): raise TokenLimitExceeded(self.get_limit_error_message(input_tokens))
            self.update_token_count(input_tokens)

            if stream:
                # Anthropic streaming for images follows the same pattern as text
                completion_text = ""
                async with self.client.messages.stream(
                    model=self.model, max_tokens=self.max_tokens,
                    temperature=temperature if temperature is not None else self.temperature,
                    messages=formatted_anth_messages, system=system_prompt_str if system_prompt_str else None,
                    extra_headers=self.ANTHROPIC_DEFAULT_HEADERS
                ) as stream_obj:
                    async for event in stream_obj:
                        if event.type == "content_block_delta" and event.delta.type == "text_delta":
                            chunk_text = event.delta.text
                            print(chunk_text, end="", flush=True); completion_text += chunk_text
                print()
                if not completion_text: raise ValueError("Empty response from Anthropic streaming LLM (images)")
                self.total_completion_tokens += self.count_tokens(completion_text)
                return completion_text
            else: # Non-streaming Anthropic with images
                response = await self.client.messages.create(
                    model=self.model, max_tokens=self.max_tokens,
                    temperature=temperature if temperature is not None else self.temperature,
                    messages=formatted_anth_messages, system=system_prompt_str if system_prompt_str else None,
                    extra_headers=self.ANTHROPIC_DEFAULT_HEADERS
                )
                response_text = ""
                if response.content: 
                    for block in response.content: 
                        if hasattr(block, 'text'): response_text += block.text
                if not response_text: raise ValueError("Empty or invalid response from Anthropic LLM (images)")
                self.update_token_count(response.usage.input_tokens, response.usage.output_tokens)
                return response_text
        else: # OpenAI or compatible for images
            # ... (original OpenAI ask_with_images logic using self._format_openai_messages) ...
            # This part is complex and needs careful merging with the original OpenAI logic for images.
            # For now, I'll just put a placeholder indicating it uses the OpenAI path.
            logger.info("Using OpenAI-compatible path for ask_with_images")
            formatted_oa_system_msgs = self._format_openai_messages(system_msgs, True) if system_msgs else []
            formatted_oa_user_msgs = self._format_openai_messages(messages, True)
            
            # Original logic to inject images into the last user message for OpenAI
            if not formatted_oa_user_msgs or formatted_oa_user_msgs[-1]["role"] != "user":
                raise ValueError("The last message must be from the user to attach images for OpenAI client")
            last_message = formatted_oa_user_msgs[-1]
            content = last_message["content"]
            multimodal_content = ([{"type": "text", "text": content}] if isinstance(content, str) 
                                 else content if isinstance(content, list) else [])
            for image_spec in images:
                if isinstance(image_spec, str): # Assumed to be a URL or base64 data URI
                    multimodal_content.append({"type": "image_url", "image_url": {"url": image_spec}})
                elif isinstance(image_spec, dict) and "url" in image_spec: # OpenAI format for image_url dict
                     multimodal_content.append({"type": "image_url", "image_url": image_spec})
                elif isinstance(image_spec, dict) and "image_url" in image_spec: # Handling nested image_url
                     multimodal_content.append(image_spec)
                else: raise ValueError(f"Unsupported image format for OpenAI: {image_spec}")
            last_message["content"] = multimodal_content
            all_oa_messages = formatted_oa_system_msgs + formatted_oa_user_msgs

            input_tokens = self.count_message_tokens(all_oa_messages)
            if not self.check_token_limit(input_tokens): raise TokenLimitExceeded(self.get_limit_error_message(input_tokens))
            
            params = {"model": self.model, "messages": all_oa_messages, "stream": stream}
            if self.model in REASONING_MODELS: params["max_completion_tokens"] = self.max_tokens
            else: params["max_tokens"] = self.max_tokens
            params["temperature"] = temperature if temperature is not None else self.temperature

            if not stream:
                response = await self.client.chat.completions.create(**params)
                if not response.choices or not response.choices[0].message.content: raise ValueError("Empty response from LLM")
                self.update_token_count(response.usage.prompt_tokens, response.usage.completion_tokens)
                return response.choices[0].message.content
            else: # OpenAI streaming with images
                self.update_token_count(input_tokens)
                response_stream = await self.client.chat.completions.create(**params)
                collected_messages_text = []
                completion_text = ""
                async for chunk in response_stream:
                    chunk_content = chunk.choices[0].delta.content or ""
                    collected_messages_text.append(chunk_content); completion_text += chunk_content
                    print(chunk_content, end="", flush=True)
                print()
                full_response = "".join(collected_messages_text).strip()
                if not full_response: raise ValueError("Empty response from streaming LLM")
                self.total_completion_tokens += self.count_tokens(completion_text)
                return full_response

    @retry(
        wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6),
        retry=retry_if_exception_type((OpenAIError, AnthropicError, Exception, ValueError))
    )
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 300,
        tools: Optional[List[dict]] = None,
        tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> Union[OpenAIChatCompletionMessage, AnthropicMessageParam, None]:
        try:
            if tool_choice not in TOOL_CHOICE_VALUES: raise ValueError(f"Invalid tool_choice: {tool_choice}")
            supports_images = self.model in MULTIMODAL_MODELS

            if isinstance(self.client, AsyncAnthropic):
                if tools:
                    logger.warning("Anthropic tool use is experimental and requires specific formatting. Basic pass-through attempted.")
                    # TODO: Convert OpenAI-style tools to Anthropic's tool definition format if possible.
                    # This is a placeholder for actual Anthropic tool definition and handling.
                    # For now, we might just pass tools if the SDK supports it, or ignore.
                
                system_prompt_str = ""
                if system_msgs:
                    for msg_item in system_msgs:
                        msg_dict = msg_item.to_dict() if isinstance(msg_item, Message) else msg_item
                        if msg_dict.get("role") == "system" and msg_dict.get("content"):
                            system_prompt_str += str(msg_dict.get("content")) + "\n"
                system_prompt_str = system_prompt_str.strip()

                formatted_anth_messages = self._format_anthropic_messages(messages, supports_images)
                # (Token counting and limit check would be here)
                temp_text_for_token_check = system_prompt_str + " ".join([str(m.get("content", "")) for m in formatted_anth_messages])
                input_tokens = self.count_tokens(temp_text_for_token_check) 
                if not self.check_token_limit(input_tokens): raise TokenLimitExceeded(self.get_limit_error_message(input_tokens))
                self.update_token_count(input_tokens) 

                # Simplified: Anthropic client `messages.create` doesn't directly take `tool_choice` like OpenAI.
                # It takes a `tools` parameter. The response will contain `tool_use` blocks if a tool is invoked.
                # This is a very basic attempt and likely needs more sophisticated tool definition and response parsing.
                anthropic_tools_param = []
                if tools:
                    # Attempt a naive conversion - this will likely NOT work directly for complex tools
                    for t in tools:
                        if t.get("type") == "function" and t.get("function"): 
                            anthropic_tools_param.append({
                                "name": t["function"].get("name"),
                                "description": t["function"].get("description"),
                                "input_schema": t["function"].get("parameters", {"type": "object", "properties": {}})
                            })
                
                api_params = {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "messages": formatted_anth_messages,
                    "system": system_prompt_str if system_prompt_str else None,
                    "temperature": temperature if temperature is not None else self.temperature,
                    "extra_headers": self.ANTHROPIC_DEFAULT_HEADERS
                }
                if anthropic_tools_param:
                    api_params["tools"] = anthropic_tools_param
                    # Anthropic uses tool_choice at a higher level or implicitly by providing tools
                    # For specific tool choice like 'required', that needs more complex logic or SDK support.
                    # if tool_choice == ToolChoice.REQUIRED: # This kind of direct mapping is not available
                    #    pass # May need to handle by checking response or structuring the call differently

                response = await self.client.messages.create(**api_params)

                self.update_token_count(response.usage.input_tokens, response.usage.output_tokens)

                # Convert Anthropic response to a somewhat compatible ChatCompletionMessage-like structure or a new type.
                # This is a major simplification. Real tool use requires parsing `response.content` for `tool_use` blocks.
                response_content_text = ""
                anthropic_tool_calls = []
                if response.content:
                    for block in response.content:
                        if block.type == 'text':
                            response_content_text += block.text
                        elif block.type == 'tool_use': # Basic tool_use block handling
                            anthropic_tool_calls.append(
                                ToolCall( # Using OpenManus ToolCall for now
                                    id=block.id,
                                    type='function', # Assuming function for now
                                    function={
                                        'name': block.name,
                                        'arguments': json.dumps(block.input) # input is already a dict
                                    }
                                )
                            )
                
                # Create a pseudo OpenAIChatCompletionMessage. A dedicated AnthropicMessage type would be better.
                # For now, this is a hack to fit the existing structure.
                # The `MessageParam` from anthropic could be used but it doesn't have `tool_calls` field in the same way.
                pseudo_openai_message = OpenAIChatCompletionMessage(
                    role='assistant',
                    content=response_content_text if response_content_text else None, # Content can be None if only tool_calls
                    tool_calls=[tc.model_dump() for tc in anthropic_tool_calls] if anthropic_tool_calls else None
                )
                return pseudo_openai_message

            else: # OpenAI or compatible client
                formatted_oa_messages = self._format_openai_messages(messages, supports_images)
                if system_msgs:
                    formatted_oa_system_msgs = self._format_openai_messages(system_msgs, supports_images)
                    all_oa_messages = formatted_oa_system_msgs + formatted_oa_messages
                else:
                    all_oa_messages = formatted_oa_messages

                input_tokens = self.count_message_tokens(all_oa_messages)
                tools_tokens = 0
                if tools: tools_tokens = sum(self.count_tokens(str(tool)) for tool in tools)
                input_tokens += tools_tokens
                if not self.check_token_limit(input_tokens): raise TokenLimitExceeded(self.get_limit_error_message(input_tokens))
                self.update_token_count(input_tokens)
                
                if tools: 
                    for tool in tools: 
                        if not isinstance(tool, dict) or "type" not in tool: raise ValueError("Each tool must be a dict with 'type' field")

                params = {
                    "model": self.model, "messages": all_oa_messages,
                    "tools": tools, "tool_choice": tool_choice,
                    "timeout": timeout, **kwargs
                }
                if self.model in REASONING_MODELS: params["max_completion_tokens"] = self.max_tokens
                else: params["max_tokens"] = self.max_tokens
                params["temperature"] = temperature if temperature is not None else self.temperature
                params["stream"] = False

                response: OpenAIChatCompletion = await self.client.chat.completions.create(**params)
                if not response.choices or not response.choices[0].message: return None
                self.update_token_count(response.usage.prompt_tokens, response.usage.completion_tokens)
                return response.choices[0].message

        except TokenLimitExceeded: raise
        except (ValueError, AnthropicError, OpenAIError) as e:
            logger.exception(f"LLM API error in ask_tool method: {e}")
            logger.info("DEBUG: ask_tool - In except (ValueError, AnthropicError, OpenAIError)")
            if isinstance(e, OpenAIAuthenticationError) or isinstance(e, AnthropicAuthenticationError):
                logger.error("Authentication failed. Check API key and client configuration.")
            elif isinstance(e, OpenAIRateLimitError) or isinstance(e, AnthropicRateLimitError):
                logger.error("Rate limit exceeded.")
            elif isinstance(e, OpenAIAPIError) or isinstance(e, AnthropicAPIError):
                 logger.error(f"Generic API error: {e}")
                 logger.info(f"DEBUG: ask_tool - Caught APIError type: {type(e)}")
            raise
        except Exception: logger.exception(f"Unexpected error in ask_tool"); raise

"""OpenAI API integration with structured output support."""
import json
import logging
import os
import time
from typing import Any, Dict, Optional

import openai
from openai import OpenAI

logger = logging.getLogger("evolia")

# Error messages for different OpenAI exceptions
ERROR_MESSAGES = {
    "rate_limit": "OpenAI API rate limit exceeded. Please try again in {retry_after} seconds.",
    "auth": "Authentication failed. Please check your API key.",
    "invalid_request": "Invalid request to OpenAI API: {details}",
    "api_connection": "Failed to connect to OpenAI API. Please check your network connection.",
    "internal_server": "OpenAI API internal server error. Please try again later.",
    "bad_request": "Bad request to OpenAI API: {details}",
    "empty_response": "OpenAI API returned empty response",
    "invalid_json": "Invalid JSON in OpenAI response: {details}",
    "max_retries": "Failed to get valid response after {max_retries} attempts: {error}",
    "unsupported_model": "Model {model} does not support structured output. Please use a supported model.",
}


def supports_structured_output(model_name: str) -> bool:
    """Check if the model supports structured output.

    Args:
        model_name: Name of the OpenAI model

    Returns:
        bool: True if the model supports structured output, False otherwise
    """
    # Extract version from model name
    if "gpt-4o-mini" in model_name:
        version = model_name.split("gpt-4o-mini-")[-1]
        return version >= "2024-07-18"
    elif "gpt-4o" in model_name:
        version = model_name.split("gpt-4o-")[-1]
        return version >= "2024-08-06"
    elif "o1" in model_name:
        version = model_name.split("o1-")[-1]
        return version >= "2024-12-17"
    return False


def call_openai_structured(
    api_key: str,
    model: str,
    json_schema: Dict[str, Any],
    user_prompt: str,
    system_prompt: str,
    max_retries: int = 5,
    retry_delay: int = 20,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> Dict[str, Any]:
    """Call OpenAI with structured output using json_schema response format.

    Args:
        api_key: OpenAI API key
        model: Model to use (e.g. "gpt-4o-2024-08-06")
        json_schema: JSON Schema for the expected response
        user_prompt: User prompt
        system_prompt: System prompt
        max_retries: Maximum number of retries
        retry_delay: Delay between retries in seconds
        temperature: Temperature for sampling
        max_tokens: Maximum tokens in response
        top_p: Top-p sampling parameter
        frequency_penalty: Frequency penalty parameter
        presence_penalty: Presence penalty parameter

    Returns:
        Dict matching the provided JSON schema

    Raises:
        openai.RateLimitError: If API rate limit is exceeded
        openai.AuthenticationError: If API key is invalid
        openai.BadRequestError: If request is malformed
        openai.APIConnectionError: If connection to API fails
        openai.InternalServerError: If OpenAI has internal error
        RuntimeError: For other errors or if max retries exceeded
        ValueError: If model does not support structured output
    """
    # Check if model supports structured output
    if not supports_structured_output(model):
        error_msg = ERROR_MESSAGES["unsupported_model"].format(model=model)
        logger.error(error_msg)
        raise ValueError(error_msg)

    client = OpenAI(api_key=api_key)

    logger.info(
        "Starting OpenAI API call",
        extra={
            "payload": {
                "component": "openai",
                "operation": "structured_call",
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "schema_type": json_schema.get("type", "unknown"),
            }
        },
    )

    for attempt in range(max_retries):
        try:
            logger.debug(
                f"Attempt {attempt + 1}/{max_retries}",
                extra={
                    "payload": {
                        "component": "openai",
                        "operation": "structured_call",
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                    }
                },
            )

            # Log request payload
            logger.debug(
                "OpenAI API request",
                extra={
                    "payload": {
                        "component": "openai",
                        "operation": "structured_call",
                        "request": {
                            "model": model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            "response_format": {
                                "type": "json_schema",
                                "json_schema": {
                                    "name": "structured_output",
                                    "schema": json_schema,
                                },
                            },
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "top_p": top_p,
                            "frequency_penalty": frequency_penalty,
                            "presence_penalty": presence_penalty,
                        },
                    }
                },
            )

            # Log full request details
            logger.debug(
                "FULL OPENAI REQUEST",
                extra={
                    "payload": {
                        "component": "openai",
                        "operation": "structured_call",
                        "full_request": {
                            "system_prompt": system_prompt,
                            "user_prompt": user_prompt,
                            "json_schema": json_schema,
                            "model": model,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "top_p": top_p,
                            "frequency_penalty": frequency_penalty,
                            "presence_penalty": presence_penalty,
                        },
                    }
                },
            )

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_output",
                        "schema": json_schema,
                    },
                },
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )

            # Log successful API call with response content
            logger.debug(
                "Received OpenAI API response",
                extra={
                    "payload": {
                        "component": "openai",
                        "operation": "structured_call",
                        "attempt": attempt + 1,
                        "response_id": response.id,
                        "model": response.model,
                        "usage": {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens,
                        }
                        if response.usage
                        else None,
                        "response_content": response.choices[0].message.content
                        if response.choices
                        else None,
                    }
                },
            )

            # Extract JSON response
            content = response.choices[0].message.content
            if not content:
                logger.error(
                    ERROR_MESSAGES["empty_response"],
                    extra={
                        "payload": {
                            "component": "openai",
                            "operation": "structured_call",
                            "response_id": response.id,
                        }
                    },
                )
                raise RuntimeError(ERROR_MESSAGES["empty_response"])

            # Parse JSON response
            try:
                result = json.loads(content)
                logger.info(
                    "Successfully parsed OpenAI response",
                    extra={
                        "payload": {
                            "component": "openai",
                            "operation": "structured_call",
                            "response_id": response.id,
                        }
                    },
                )
                return result

            except json.JSONDecodeError as e:
                error_msg = ERROR_MESSAGES["invalid_json"].format(details=str(e))
                logger.error(
                    error_msg,
                    extra={
                        "payload": {
                            "component": "openai",
                            "operation": "structured_call",
                            "error": str(e),
                            "content": content[
                                :1000
                            ],  # Log first 1000 chars of invalid JSON
                        }
                    },
                )
                raise RuntimeError(error_msg)

        except openai.RateLimitError as e:
            retry_after = getattr(e, "retry_after", retry_delay)
            error_msg = ERROR_MESSAGES["rate_limit"].format(retry_after=retry_after)
            logger.warning(
                error_msg,
                extra={
                    "payload": {
                        "component": "openai",
                        "operation": "structured_call",
                        "retry_after": retry_after,
                        "attempt": attempt + 1,
                    }
                },
            )
            if attempt < max_retries - 1:
                time.sleep(retry_after or retry_delay)
                continue
            raise

        except openai.AuthenticationError as e:
            error_msg = ERROR_MESSAGES["auth"]
            logger.error(
                error_msg,
                extra={
                    "payload": {
                        "component": "openai",
                        "operation": "structured_call",
                        "error": str(e),
                    }
                },
            )
            raise

        except openai.BadRequestError as e:
            error_msg = ERROR_MESSAGES["invalid_request"].format(details=str(e))
            logger.error(
                error_msg,
                extra={
                    "payload": {
                        "component": "openai",
                        "operation": "structured_call",
                        "error": str(e),
                    }
                },
            )
            raise

        except openai.APIConnectionError as e:
            error_msg = ERROR_MESSAGES["api_connection"]
            logger.error(
                error_msg,
                extra={
                    "payload": {
                        "component": "openai",
                        "operation": "structured_call",
                        "error": str(e),
                        "attempt": attempt + 1,
                    }
                },
            )
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            raise

        except openai.InternalServerError as e:
            error_msg = ERROR_MESSAGES["internal_server"]
            logger.error(
                error_msg,
                extra={
                    "payload": {
                        "component": "openai",
                        "operation": "structured_call",
                        "error": str(e),
                        "attempt": attempt + 1,
                    }
                },
            )
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            raise

        except Exception as e:
            error_msg = ERROR_MESSAGES["max_retries"].format(
                max_retries=max_retries, error=str(e)
            )
            logger.error(
                error_msg,
                extra={
                    "payload": {
                        "component": "openai",
                        "operation": "structured_call",
                        "error": str(e),
                        "attempt": attempt + 1,
                    }
                },
            )
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            raise RuntimeError(error_msg)

    raise RuntimeError(
        ERROR_MESSAGES["max_retries"].format(
            max_retries=max_retries, error="Unknown error"
        )
    )

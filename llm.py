"""
Utilities for selecting browser tabs with the help of a ChatGPT model.

This module leans on the OpenAI API and expects the `OPENAI_API_KEY`
environment variable to be present. Downstream callers can invoke
`select_tabs_with_llm` to obtain a ranked list of bookmarks that should
be opened for a particular prompt.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pydantic import BaseModel, HttpUrl, ValidationError
from openai import AsyncOpenAI

DEFAULT_MODEL = os.getenv("TAB_PLANNER_OPENAI_MODEL", "gpt-4.1-mini")

RESPONSE_SCHEMA = {
    "name": "tab_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "tabs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "reason": {"type": "string"},
                        "score": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["title", "url", "reason", "score"],
                },
            }
        },
        "required": ["tabs"],
    },
}

SYSTEM_PROMPT = """You are a research assistant tasked with helping a user decide
which saved bookmarks to open in new browser tabs for their current task.

General rules:
- Only choose from the bookmarks provided. Do not invent new URLs.
- Choose the smallest set of tabs that still covers the user's intent.
- Prioritize relevance to the prompt, freshness implied by the prompt, and topic coverage.
- Provide a concise reason for each selection. Mention why the bookmark helps.
- Provide a confidence score between 0 and 1. Use higher scores for better matches.
- Return answers strictly as JSON that matches the provided schema with a top-level `tabs` array.
"""


class BookmarkPayload(BaseModel):
    title: str
    url: HttpUrl
    tags: list[str] | None = None
    description: str | None = None


class BrowsingHistoryPayload(BaseModel):
    title: str
    url: HttpUrl
    last_visited: str | None = None


class OpenTabPayload(BaseModel):
    title: str
    url: HttpUrl
    opened_at: str | None = None
    pinned: bool | None = None


@dataclass(frozen=True)
class TabSuggestion:
    title: str
    url: HttpUrl
    reason: str
    score: float


class LLMSuggestionError(RuntimeError):
    """Raised when the language model could not produce a usable response."""


def _get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMSuggestionError(
            "Missing OPENAI_API_KEY environment variable; cannot call the language model."
        )
    return AsyncOpenAI(api_key=api_key)


def _build_user_message(
    prompt: str,
    bookmarks: list[BookmarkPayload],
    max_tabs: int,
    history: list[BrowsingHistoryPayload] | None = None,
    open_tabs: list[OpenTabPayload] | None = None,
) -> str:
    bookmark_lines = []
    for index, bookmark in enumerate(bookmarks, start=1):
        tags_text = f" | tags: {', '.join(bookmark.tags)}" if bookmark.tags else ""
        description_text = (
            f"\n    description: {bookmark.description}" if bookmark.description else ""
        )
        bookmark_lines.append(
            f"{index}. {bookmark.title}\n    url: {bookmark.url}{tags_text}{description_text}"
        )

    bookmarks_block = (
        "\n".join(bookmark_lines) if bookmark_lines else "No bookmarks supplied."
    )

    history_block = "No history items supplied."
    if history:
        history_lines = []
        for index, entry in enumerate(history, start=1):
            last_visited_text = (
                f" (last visited: {entry.last_visited})" if entry.last_visited else ""
            )
            history_lines.append(
                f"{index}. {entry.title} — {entry.url}{last_visited_text}"
            )
        history_block = "\n".join(history_lines)

    open_tabs_block = "No open tabs supplied."
    if open_tabs:
        open_tab_lines = []
        for index, tab in enumerate(open_tabs, start=1):
            status_parts = []
            if tab.pinned is not None:
                status_parts.append("pinned" if tab.pinned else "unpinned")
            if tab.opened_at:
                status_parts.append(f"opened at {tab.opened_at}")
            status_suffix = f" ({', '.join(status_parts)})" if status_parts else ""
            open_tab_lines.append(f"{index}. {tab.title} — {tab.url}{status_suffix}")
        open_tabs_block = "\n".join(open_tab_lines)

    return (
        "User prompt:\n"
        f"{prompt.strip()}\n\n"
        f"Bookmarks:\n{bookmarks_block}\n\n"
        f"Recent history entries:\n{history_block}\n\n"
        f"Currently open tabs:\n{open_tabs_block}\n\n"
        f"Select up to {max_tabs} bookmarks that best satisfy the user's prompt. "
        "You may reference history or existing tabs when explaining your choices."
    )


async def select_tabs_with_llm(
    *,
    prompt: str,
    bookmarks: list[BookmarkPayload],
    history: list[BrowsingHistoryPayload] | None = None,
    open_tabs: list[OpenTabPayload] | None = None,
    max_tabs: int,
    model: str | None = None,
    temperature: float = 0.2,
) -> list[TabSuggestion]:
    """
    Ask a ChatGPT model to decide which bookmarks should be opened as tabs.

    Parameters
    ----------
    prompt:
        The user's goal or task description.
    bookmarks:
        The list of available bookmarks to choose from (must not be empty).
    history:
        Optional list of recent browser history items to provide extra context.
    open_tabs:
        Optional list of currently open tabs to avoid duplication or leverage existing pages.
    max_tabs:
        The upper bound on the number of tabs to return.
    model:
        Override for the OpenAI model name to use. Falls back to DEFAULT_MODEL.
    temperature:
        Creativity parameter passed through to the model.
    """
    cleaned_prompt = prompt.strip()
    if not cleaned_prompt:
        raise ValueError("Prompt must not be empty.")

    if not bookmarks:
        raise ValueError("At least one bookmark must be provided.")

    if max_tabs <= 0:
        raise ValueError("max_tabs must be greater than zero.")

    try:
        validated_bookmarks = [BookmarkPayload.model_validate(b) for b in bookmarks]
    except ValidationError as exc:
        raise ValueError(f"Invalid bookmark payload: {exc}") from exc

    validated_history = None
    if history is not None:
        try:
            validated_history = [
                BrowsingHistoryPayload.model_validate(item) for item in history
            ]
        except ValidationError as exc:
            raise ValueError(f"Invalid browsing history payload: {exc}") from exc

    validated_open_tabs = None
    if open_tabs is not None:
        try:
            validated_open_tabs = [
                OpenTabPayload.model_validate(item) for item in open_tabs
            ]
        except ValidationError as exc:
            raise ValueError(f"Invalid open tab payload: {exc}") from exc

    client = _get_client()
    user_message = _build_user_message(
        cleaned_prompt,
        validated_bookmarks,
        max_tabs,
        history=validated_history,
        open_tabs=validated_open_tabs,
    )
    chosen_model = model or DEFAULT_MODEL

    try:
        response = await client.chat.completions.create(
            model=chosen_model,
            temperature=temperature,
            response_format={
                "type": "json_schema",
                "json_schema": RESPONSE_SCHEMA,
            },
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
    except Exception as exc:  # noqa: BLE001 - bubble up as domain-specific error
        raise LLMSuggestionError(f"OpenAI API request failed: {exc}") from exc

    if not response.choices:
        raise LLMSuggestionError("Language model returned no choices.")

    message = response.choices[0].message
    if message is None or message.content is None:
        raise LLMSuggestionError("Language model returned an empty message.")

    try:
        payload = json.loads(message.content)
    except json.JSONDecodeError as exc:
        raise LLMSuggestionError("Language model response was not valid JSON.") from exc

    tabs_data = payload.get("tabs")
    if not isinstance(tabs_data, list):
        raise LLMSuggestionError(
            "Language model response did not include a 'tabs' list."
        )

    suggestions: list[TabSuggestion] = []
    for item in tabs_data:
        if not isinstance(item, dict):
            continue
        try:
            suggestion = TabSuggestion(
                title=item["title"],
                url=item["url"],
                reason=item.get("reason", "No reason provided."),
                score=float(item.get("score", 0.5)),
            )
        except (KeyError, TypeError, ValueError):
            continue
        suggestions.append(suggestion)

    return suggestions[:max_tabs]

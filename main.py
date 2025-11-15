import re
from typing import Iterable

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

app = FastAPI(
    title="Bookmark Tab Planner",
    description="Suggests which browser tabs to open based on a prompt and saved bookmarks.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Bookmark(BaseModel):
    title: str
    url: HttpUrl
    tags: list[str] | None = None
    description: str | None = None


class TabPlanRequest(BaseModel):
    prompt: str
    bookmarks: list[Bookmark]


class Tab(BaseModel):
    title: str
    url: HttpUrl
    reason: str
    score: float


class TabPlanResponse(BaseModel):
    tabs: list[Tab]


MAX_TABS_DEFAULT = 5


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"\b[\w-]+\b", text.lower())
    return {token for token in tokens if token}


def _combine_fields(bookmark: Bookmark) -> Iterable[str]:
    yield bookmark.title
    if bookmark.description:
        yield bookmark.description
    if bookmark.tags:
        yield from bookmark.tags


def _bookmark_tokens(bookmark: Bookmark) -> set[str]:
    combined_text = " ".join(_combine_fields(bookmark))
    return _tokenize(combined_text)


def _score_bookmark(bookmark: Bookmark, prompt_tokens: set[str]) -> tuple[float, str]:
    if not prompt_tokens:
        return 0.0, "No prompt keywords; preserving original order."

    bookmark_tokens = _bookmark_tokens(bookmark)
    if not bookmark_tokens:
        return 0.0, "Bookmark has no descriptive text to compare."

    matched_tokens = prompt_tokens & bookmark_tokens
    if matched_tokens:
        score = float(len(matched_tokens))
        return score, f"Matched keywords: {', '.join(sorted(matched_tokens))}"

    title_lower = bookmark.title.lower()
    if any(token in title_lower for token in prompt_tokens):
        return 0.5, "Prompt keywords appear within the bookmark title."

    return 0.0, "No keyword matches; keeping bookmark for context."


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/tabs", response_model=TabPlanResponse)
async def plan_tabs(
    request: TabPlanRequest, limit: int = MAX_TABS_DEFAULT
) -> TabPlanResponse:
    prompt_tokens = _tokenize(request.prompt)
    evaluated: list[tuple[float, str, int, Bookmark]] = []

    for index, bookmark in enumerate(request.bookmarks):
        score, reason = _score_bookmark(bookmark, prompt_tokens)
        evaluated.append((score, reason, index, bookmark))

    evaluated.sort(key=lambda item: (-item[0], item[2]))

    if limit <= 0:
        limit = len(evaluated)

    selected = evaluated[:limit]
    tabs = [
        Tab(title=bookmark.title, url=bookmark.url, reason=reason, score=score)
        for score, reason, _, bookmark in selected
    ]

    if not tabs:
        raise HTTPException(status_code=400, detail="No bookmarks provided")

    return TabPlanResponse(tabs=tabs)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

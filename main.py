from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

from llm import (
    BookmarkPayload,
    BrowsingHistoryPayload,
    LLMSuggestionError,
    OpenTabPayload,
    select_tabs_with_llm,
)

from dotenv import load_dotenv

load_dotenv()

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


class HistoryEntry(BaseModel):
    title: str
    url: HttpUrl
    last_visited: str | None = None


class OpenTab(BaseModel):
    title: str
    url: HttpUrl
    opened_at: str | None = None
    pinned: bool | None = None


class TabPlanRequest(BaseModel):
    prompt: str
    bookmarks: list[Bookmark]
    history: list[HistoryEntry] | None = None
    open_tabs: list[OpenTab] | None = None


class Tab(BaseModel):
    title: str
    url: HttpUrl
    reason: str
    score: float


class TabPlanResponse(BaseModel):
    tabs: list[Tab]


MAX_TABS_DEFAULT = 5


@app.post("/tabs", response_model=TabPlanResponse)
async def plan_tabs_llm(
    request: TabPlanRequest,
    limit: int = MAX_TABS_DEFAULT,
    model: str | None = None,
    temperature: float = 0.2,
) -> TabPlanResponse:
    max_tabs = limit if limit > 0 else MAX_TABS_DEFAULT
    try:
        bookmarks_payload = [
            BookmarkPayload.model_validate(bookmark.model_dump())
            for bookmark in request.bookmarks
        ]
        history_payload = (
            [
                BrowsingHistoryPayload.model_validate(item.model_dump())
                for item in request.history
            ]
            if request.history
            else None
        )
        open_tabs_payload = (
            [
                OpenTabPayload.model_validate(item.model_dump())
                for item in request.open_tabs
            ]
            if request.open_tabs
            else None
        )

        suggestions = await select_tabs_with_llm(
            prompt=request.prompt,
            bookmarks=bookmarks_payload,
            history=history_payload,
            open_tabs=open_tabs_payload,
            max_tabs=max_tabs,
            model=model,
            temperature=temperature,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMSuggestionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if not suggestions:
        raise HTTPException(
            status_code=502,
            detail="Language model returned no tab suggestions.",
        )

    tabs = [
        Tab(
            title=suggestion.title,
            url=suggestion.url,
            reason=suggestion.reason,
            score=suggestion.score,
        )
        for suggestion in suggestions
    ]

    return TabPlanResponse(tabs=tabs)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

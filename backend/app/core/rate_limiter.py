# medical-triage-agent-ai-poc/backend/app/core/rate_limiter.py

import time

from collections import defaultdict
from fastapi import HTTPException, Request

from backend.app.core.config import settings


REQUEST_HISTORY = defaultdict(list)


async def rate_limit(request: Request):

    client_ip = request.client.host

    current_time = time.time()

    window_start = (
        current_time -
        settings.RATE_LIMIT_WINDOW_SECONDS
    )

    REQUEST_HISTORY[client_ip] = [
        ts
        for ts in REQUEST_HISTORY[client_ip]
        if ts > window_start
    ]

    if (
        len(REQUEST_HISTORY[client_ip])
        >= settings.RATE_LIMIT_REQUESTS
    ):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded"
        )

    REQUEST_HISTORY[client_ip].append(current_time)

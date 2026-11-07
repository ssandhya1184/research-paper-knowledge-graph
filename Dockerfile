FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY . .

RUN uv sync --frozen --no-dev

EXPOSE 7860

CMD ["uv", "run", "--no-sync", "python", "-m", "research_assistant.ui.app"]
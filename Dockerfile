FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_BIND_HOST=0.0.0.0

WORKDIR /app

# Install dependencies separately so source changes can reuse this build layer.
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && groupadd --gid 10001 agent \
    && useradd --uid 10001 --gid agent --no-create-home agent \
    && mkdir /app/.agent-data \
    && chown agent:agent /app/.agent-data

# Explicit source list: no secrets, local history, previews, tests or virtual environment.
COPY config.py demo.py email_delivery.py email_format.py main.py \
    official_sources.py openrouter_client.py research_agent.py \
    run_tracking.py stages.py web_app.py ./

USER 10001:10001
EXPOSE 8765

# Checks the local form only; never invokes research, an API or email delivery.
HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/', timeout=3).close()"]

CMD ["python", "web_app.py"]

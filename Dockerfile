FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY crew ./crew
RUN pip install ".[api]" && useradd -m crew && mkdir -p /data && chown crew /data
USER crew
ENV CREW_TRACE_DIR=/data/runs
EXPOSE 8000
ENTRYPOINT ["crew"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml README.md ./
COPY rolescout ./rolescout
RUN pip install --no-cache-dir .
RUN useradd -m app && mkdir -p /data && chown app /data
USER app
ENV DATABASE_URL=sqlite:////data/rolescout.db ROLESCOUT_OUTBOX_DIR=/data/outbox ROLESCOUT_SECRET_FILE=/data/.secret
WORKDIR /data
EXPOSE 8000
CMD ["rolescout", "serve", "--host", "0.0.0.0", "--port", "8000"]

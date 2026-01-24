# Lethus Backend Dockerfile
# Optimized for low-resource servers

FROM python:3.11-slim

WORKDIR /app

# Install minimal runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy and install Python dependencies first (better caching)
COPY pyproject.toml ./
COPY src ./src

# Install the package (makes lethus module importable)
RUN pip install --no-cache-dir .

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Create non-root user
RUN useradd --create-home --shell /bin/bash lethus
USER lethus

# Expose port
EXPOSE 8000

# Health check - longer start period for initialization
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "lethus.main"]

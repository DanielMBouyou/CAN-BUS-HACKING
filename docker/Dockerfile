FROM python:3.11-slim

LABEL maintainer="DanielMBouyou"
LABEL description="CAN-Stealth-Attack-AI Lab"

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    can-utils \
    iproute2 \
    kmod \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier les fichiers de config d'abord (cache layer)
COPY pyproject.toml README.md ./

# Copier le code source
COPY src/ src/
COPY data/ data/

# Installer le projet
RUN pip install --no-cache-dir -e ".[dev]"

# Exposer les ports API + Dashboard
EXPOSE 8000 8501

# Script d'entrée
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["api"]

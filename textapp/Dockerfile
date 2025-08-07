FROM python:3.13.5-alpine3.21

# Install dependencies required for Python packages like torch, pydub, etc.
RUN apk add --no-cache \
    gcc \
    g++ \
    musl-dev \
    ffmpeg \
    libffi-dev \
    build-base \
    python3-dev \
    openssl-dev \
    portaudio-dev \
    libstdc++ \
    libmagic \
    bash \
    curl \
    git

# Set working directory
WORKDIR /app/

# Copy project files
COPY . /app/

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Run the Django development server (you may want to change host/port)
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

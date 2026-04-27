FROM python:3.10-slim

# Set up a new user named "user" with user ID 1000
# Hugging Face requires this for security
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy your code into the container
COPY --chown=user . /app

# Install your requirements
RUN pip install --no-cache-dir -r requirements.txt

# Open the port Hugging Face uses
EXPOSE 7860

# Start your FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]

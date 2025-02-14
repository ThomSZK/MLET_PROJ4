FROM python:3.12-slim

# Set the working directory
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

ENTRYPOINT [ "streamlit", "run", "front.py", "--server.port", "8501" ]a
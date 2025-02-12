FROM python:3.12.3

# Set the working directory
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . . 

CMD uvicorn main:app --host=0.0.0.0 --reload
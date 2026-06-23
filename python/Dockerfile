FROM python:3.13.3-slim

RUN apt update
RUN apt install -y --no-install-recommends unzip wget curl build-essential libicu-dev fonts-noto-cjk && apt clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -U langchain-community
RUN pip install -U langchain-google-genai
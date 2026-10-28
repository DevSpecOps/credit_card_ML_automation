FROM python:3.10-slim

WORKDIR /app

ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple

COPY requirements.api.txt ./
RUN pip install --no-cache-dir -r requirements.api.txt

COPY src ./src

EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

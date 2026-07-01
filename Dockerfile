FROM python:3.12-slim

WORKDIR /tennis-match-predictor

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p logs

COPY . .

CMD ["python", "main.py"]

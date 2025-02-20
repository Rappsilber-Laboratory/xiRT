FROM python:3.7

RUN mkdir -p /app
WORKDIR /app

COPY setup.py .
COPY xirt/ xirt/
COPY requirements.txt .

RUN pip install -r requirements.txt
RUN pip install .

ENTRYPOINT ["python", "-m", "xirt"]

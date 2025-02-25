FROM python:3.7

RUN mkdir -p /app
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY setup.py .
COPY setup.cfg .
COPY xirt/ xirt/

RUN pip install .

ENTRYPOINT ["python", "-m", "xirt"]

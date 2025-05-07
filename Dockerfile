FROM python:3.10

RUN mkdir -p /app
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY setup.py .
COPY versioneer.py .
COPY setup.cfg .
COPY xirt/ xirt/

RUN pip install .

RUN chmod -R a+rw /app

ENTRYPOINT ["python", "-m", "xirt"]

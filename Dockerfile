FROM python:3.9

RUN mkdir -p /app
WORKDIR /app

COPY requirements.txt .
RUN ulimit -n 65536 && CYTHON_NTHREADS=8 pip install --prefer-binary -r requirements.txt

COPY setup.py .
COPY setup.cfg .
COPY xirt/ xirt/

RUN ulimit -n 65536 && CYTHON_NTHREADS=8 pip install .

RUN chmod -R a+rw /app

ENTRYPOINT ["python", "-m", "xirt"]

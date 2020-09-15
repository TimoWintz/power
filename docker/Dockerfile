FROM tiangolo/uwsgi-nginx:python3.8
RUN useradd -ms /bin/bash power
ENV PATH=${HOME}/.local/bin:${PATH}
COPY python/requirements.txt /app/
RUN pip install -r /app/requirements.txt
COPY python/* /app/
COPY secrets/*.txt /app/
COPY uwsgi.ini /app/
COPY uwsgi.py /app/

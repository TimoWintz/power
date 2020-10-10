FROM tiangolo/uwsgi-nginx:python3.8
RUN useradd -ms /bin/bash power
ENV PATH=${HOME}/.local/bin:${PATH}
COPY python/requirements.txt /app/
RUN pip install -r /app/requirements.txt
COPY python/ /app/
COPY secrets/*.txt /app/
ENV UWSGI_CHEAPER 4
ENV UWSGI_PROCESSES 8


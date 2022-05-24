FROM ubuntu:22.04
RUN apt-get update && apt-get install -y python3-dev python3-pip python3-pybind11 cmake git libeigen3-dev
RUN useradd -ms /bin/bash power
ENV PATH=${HOME}/.local/bin:${PATH}
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt
RUN git clone https://github.com/autodiff/autodiff /tmp/autodiff
RUN cd /tmp/autodiff && mkdir build && cd build && cmake .. -DAUTODIFF_BUILD_TESTS=0 -DAUTODIFF_BUILD_PYTHON=0 -DAUTODIFF_BUILD_EXAMPLES=0 -D AUTODIFF_BUILD_DOCS=0 && make && make install
COPY src /app/src
COPY app/ /app/
COPY setup.py CMakeLists.txt /app/
WORKDIR /app/
RUN pip install .
CMD ["/usr/local/bin/uwsgi", "--http-socket",  "0.0.0.0:5000",  "--module", "opt_power.flask_app:app", "--check-static", "/app/static", "--workers", "4"]

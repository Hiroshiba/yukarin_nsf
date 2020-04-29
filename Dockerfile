FROM hiroshiba/hiho-deep-docker-base:pytorch1.5-cuda9.0

WORKDIR /app

# install requirements
COPY requirements.txt /app/
RUN pip install -r requirements.txt

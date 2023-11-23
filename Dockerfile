FROM python:3.6

RUN mkdir -p /data/tianxing/PycharmProjects/CallbotNLPServer

COPY . /data/tianxing/PycharmProjects/CallbotNLPServer --exclude /data/tianxing/PycharmProjects/CallbotNLPServer/server/callbot_nlp_server/dotenv/

WORKDIR /data/tianxing/PycharmProjects/CallbotNLPServer

RUN pip3 install --upgrade pip

RUN pip3 install --no-cache-dir --upgrade -r requirements.txt

RUN bash -c 'bash install.sh --stage 3 --stop_stage 5 --system_version ubuntu'

CMD ["bash", "/data/tianxing/PycharmProjects/CallbotNLPServer/start.sh"]

## CallbotNLPServer

Callbot NLP Server


### deploy
```text
git clone https://github.com/qgyd2021/CallbotNLPServer.git

cd CallbotNLPServer

docker build -t callbot_nlp:v20231123_1020 .

docker run --name nlp_hk_sit -dit -p 9070:9070 \
-e port=9070 \
-e environment=hk_sit \
-e num_processes=1 \
-v /data/tianxing/PycharmProjects/CallbotNLPServer/server/callbot_nlp_server/dotenv:/data/tianxing/PycharmProjects/CallbotNLPServer/server/callbot_nlp_server/dotenv \
callbot_nlp:v20231123_1020 /bin/bash \
-c 'bash /data/tianxing/PycharmProjects/CallbotNLPServer/start.sh --port 9070'

```

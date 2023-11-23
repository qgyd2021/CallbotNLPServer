## CallbotNLPServer

Callbot NLP Server


```text
git clone https://github.com/qgyd2021/CallbotNLPServer.git

cd CallbotNLPServer

docker build -t callbot_nlp:v20231123_1020 .

docker run --name callbot_nlp -dit -p 9080:9080 \
-e port 9080 \
-e environment hk_dev \
callbot_nlp:v20231123_1020 /bin/bash \
-c 'bash /data/tianxing/PycharmProjects/CallbotNLPServer/start.sh'

```

## CallbotNLPServer

Callbot NLP Server


### 部署

```text
mkdir -p /data/tianxing/PycharmProjects

cd /data/tianxing/PycharmProjects

git clone https://github.com/qgyd2021/CallbotNLPServer.git
```
宿主机上创建目录并拉取代码.

```text
cd /data/tianxing/PycharmProjects/CallbotNLPServer/server/callbot_nlp_server

rz -ve

unzip dotenv.zip
```
上传 dotenv.zip 文件并解压. 

```text
cd /data/tianxing/PycharmProjects/CallbotNLPServer

docker build -t callbot_nlp:v20240227_1735 .
```
构建镜像. 

```text
docker run --name nlp_id_sit_id -dit -p 9071:9071 \
-e port=9071 \
-e environment=id_sit_id \
-e num_processes=1 \
-v /data/tianxing/PycharmProjects/CallbotNLPServer/server/callbot_nlp_server/dotenv:/data/tianxing/PycharmProjects/CallbotNLPServer/server/callbot_nlp_server/dotenv \
callbot_nlp:v20231201_1510 /bin/bash \
-c 'bash /data/tianxing/PycharmProjects/CallbotNLPServer/start.sh --port 9071'

```
启动容器.


### 备注

nginx 重启
```text
cd /etc/nginx/conf.d

service nginx reload

nginx -t
```

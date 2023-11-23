#!/bin/sh
# apt-get install -y dos2unix
# apt-get install -y lrzsz

system_version="centos";

if [ ${system_version} = "centos" ]; then
#  apt-get remove docker
#  apt-get install -y docker

#  mkdir -p /data/lib/docker

  echo -e "\r{\n
  \r\"graph\": \"/data/lib/docker\"\n
  \r}\n" >/etc/docker/daemon.json

#  systemctl start docker
elif [ ${system_version} = "ubuntu" ]; then
  echo 0;
fi


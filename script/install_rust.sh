#!/usr/bin/env bash

# 参考链接:
# https://code84.com/2322.html

export profile="default"

rustc -v

mkdir -p /data/dep
cd /data/dep || exit 0;

yum -y update
yum -y install curl

curl https://sh.rustup.rs -sSf | sh

source "$HOME/.cargo/env"

rustc --version

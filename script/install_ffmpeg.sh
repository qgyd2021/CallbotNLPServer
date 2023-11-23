#!/usr/bin/env bash

# 参考链接:
# https://blog.csdn.net/weixin_44692055/article/details/128848638

mkdir -p /data/dep && cd /data/dep || exit 1;

# 安装 yasm
wget http://www.tortall.net/projects/yasm/releases/yasm-1.3.0.tar.gz
tar -zxvf yasm-1.3.0.tar.gz
cd yasm-1.3.0 || 1;
./configure
make && make install

cd ..

# 安装 ffmpeg
wget http://www.ffmpeg.org/releases/ffmpeg-4.2.tar.gz
tar -zxvf ffmpeg-4.2.tar.gz

cd ffmpeg-4.2 || exit 0;

./configure --prefix=/usr/local/ffmpeg
./configure --prefix=/usr/local/ffmpeg --enable-openssl --disable-x86asm
make && make install

# 查看版本
/usr/local/ffmpeg/bin/ffmpeg -version

# 文件末尾添加行
cat /etc/profile
echo "export PATH=$PATH:/usr/local/ffmpeg/bin" >> /etc/profile
cat /etc/profile

# 刷新资源,使其生效
source /etc/profile

# 查看版本
ffmpeg -version

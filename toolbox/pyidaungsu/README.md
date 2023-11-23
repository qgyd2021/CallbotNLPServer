## pyidaungsu 库

```text
缅甸语分词. 
```

```text
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple Cython==0.29.28
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple fasttext==0.9.2
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pyidaungsu==0.1.4

```

```text
由于 `fasttext==0.9.2` 在 linux 上报错： 
`gcc: error: unrecognized command line option '-std=c++14'`

因此, 考虑将该库代码 copy 出来, 只抽取需要的功能. 

```

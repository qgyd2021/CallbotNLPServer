#!/usr/bin/python3
# -*- coding: utf-8 -*-
import fasttext
import fasttext.util


ft = fasttext.load_model('cc.en.300.bin')
result = ft.get_dimension()
print(result)

fasttext.util.reduce_model(ft, 100)
result = ft.get_dimension()
print(result)


if __name__ == '__main__':
    pass

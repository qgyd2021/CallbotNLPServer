#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os

from toolbox.khmernltk.word_tokenize.features import create_kcc_features
from toolbox.khmernltk.utils.data import cleanup_str, seg_kcc
from toolbox.khmernltk.utils.file_utils import load_model


# sklearn_crf_ner_alt_0.9725.sav / sklearn_crf_ner_10000.sav
model_path = os.path.join(os.path.dirname(__file__), "sklearn_crf_ner_10000.sav")
crf_model = load_model(model_path)


def word_tokenize(text: str, separator: str = "-", return_tokens: bool = True):
    text = cleanup_str(text)
    skcc = seg_kcc(text)

    features = create_kcc_features(skcc)
    pred = crf_model.predict([features])

    tkcc = []
    for k in features:
        tkcc.append(k["kcc"])
    complete = ""
    tokens = []
    for i, p in enumerate(pred[0]):
        if p == "1" or i == 0:
            tokens.append(tkcc[i])
        else:
            tokens[-1] += tkcc[i]
    if return_tokens:
        return tokens

    complete = separator.join(tokens)
    complete = complete.replace(separator + " " + separator, " ")

    return complete

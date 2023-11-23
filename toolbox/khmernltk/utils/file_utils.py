#!/usr/bin/python3
# -*- coding: utf-8 -*-
import logging
import pickle


logger = logging.getLogger(__file__)


def load_model(fp):
    with open(fp, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Loaded model from {fp}")
    return model


if __name__ == '__main__':
    pass

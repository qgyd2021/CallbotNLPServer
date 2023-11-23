#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import datetime
import json
import logging
import os
from pathlib import Path
import pickle
import sys
import traceback
from tqdm import tqdm
import time

pwd = os.path.abspath(os.path.dirname(__file__))
project_path = Path(os.path.join(pwd, "../../"))
sys.path.append(project_path.as_posix())

os.environ["NLTK_DATA"] = (project_path / "data/nltk_data").as_posix()

import pandas as pd

from nxtech.nlpbot.node import Node
from nxtech.database.mysql_connect import MySqlConnect
from nxtech.table_lib.t_dialog_node_info import TDialogNodeInfo
from nxtech.table_lib.t_dialog_edge_info import TDialogEdgeInfo
from nxtech.table_lib.t_dialog_resource_info import TDialogResourceInfo
from server import log
from server.callbot_nlp_server import settings
from server.callbot_nlp_server.service.nlpbot import MakeConfig
from toolbox.design_patterns.singleton import ParamsSingleton
from toolbox.hashlib.misc import hash_json


log.setup(log_directory=settings.log_directory)
logger = logging.getLogger("apscheduler")


def get_scene_hash(
    global_config,
    product_id,
    scene_id,
):
    mysql_begin = time.time()
    # node
    t_dialog_node_info = TDialogNodeInfo(
        scene_id=scene_id,
        mysql_connect=MySqlConnect.from_json(
            global_config["mysql_connect"]
        ),
    )
    df: pd.DataFrame = t_dialog_node_info.data
    if len(df) == 0:
        return "scene_hash {}: len(df) == 0".format(scene_id)
    df = df[df["product_id"] == product_id]
    df = df[df["scene_id"] == scene_id]
    df = df.sort_values(by=["last_update_ts"])
    df = df.drop(labels=[
        "node_flag", "show_type", "node_priority", "others",
        "node_time", "intent_desc", "action_again_res_group_id",
        "create_ts", "version", "last_update_ts", "hold_before_lead",
        "hold_before_lead_on", "func_on", "max_repeat_time", "sms_id"
    ], axis=1)
    js_node = df.to_dict(orient="records")
    length_node = len(js_node)

    # edge
    t_dialog_edge_info = TDialogEdgeInfo(
        mysql_connect=MySqlConnect.from_json(
            global_config["mysql_connect"]
        ),
        scene_id=scene_id,
    )
    df: pd.DataFrame = t_dialog_edge_info.data
    df = df[df["product_id"] == product_id]
    df = df[df["scene_id"] == scene_id]
    df = df.sort_values(by=["last_update_ts"])
    df = df.drop(labels=[
        "edge_desc", "create_ts", "version", "last_update_ts", "edge_type"
    ], axis=1)
    js_edge = df.to_dict(orient="records")
    length_edge = len(js_edge)

    # resource
    t_dialog_resource_info = TDialogResourceInfo(
        mysql_connect=MySqlConnect.from_json(
            global_config["mysql_connect"]
        ),
        scene_id=scene_id,
    )
    df: pd.DataFrame = t_dialog_resource_info.data
    df = df[df["product_id"] == product_id]
    df = df[df["scene_id"] == scene_id]
    df = df.sort_values(by=["last_update_ts"])
    df = df.drop(labels=[
        "res_desc", "res_url", "create_ts", "version", "last_update_ts",
        "res_url_id", "prever_word", "res_idx", "func_type"
    ], axis=1)
    js_resource = df.to_dict(orient="records")
    length_resource = len(js_resource)

    js = [
        js_node, js_edge, js_resource
    ]

    scene_hash = hash_json(js)

    logger.info("scene id: {}; pull mysql time cost: {}; length node: {}, edge: {}, resource: {}".format(
        scene_id, time.time() - mysql_begin, length_node, length_edge, length_resource
    ))
    return scene_hash


def load_scene_id2scene_hash_dict(scene_id2scene_hash_filename):
    if os.path.exists(scene_id2scene_hash_filename):
        with open(scene_id2scene_hash_filename, "r", encoding="utf-8") as f:
            scene_id2scene_hash = json.load(f)
    else:
        scene_id2scene_hash = dict()
    return scene_id2scene_hash


def scene_initialization(product_id, scene_id, group, env, scene_hash, global_config_file, node_config_file):
    scene_config_file = settings.scenes_configs_path / scene_id / "{}_{}_{}.json".format(product_id, scene_id, env)

    scene_config = MakeConfig.make_scene_config_file(
        product_id, scene_id, env, global_config_file,
        node_config_file, scene_config_file
    )

    node_list = list()
    for config in tqdm(scene_config):
        params = config.pop("node")
        global_params = config
        node = Node.from_json(params=params, global_params=global_params)

        node_list.append({
            "node": node,
            "params": params,
            "global_params": global_params,
        })
    result = {
        "product_id": product_id,
        "scene_id": scene_id,
        "group": group,
        "env": env,
        "scene_hash": scene_hash,
        "node_list": node_list,
    }
    return result


def update():
    update_begin = time.time()
    scene_id2scene_hash_filename = os.path.join(settings.temp_update_directory, "scene_id2scene_hash.json")

    scene_id2scene_hash = load_scene_id2scene_hash_dict(scene_id2scene_hash_filename)

    # 更新
    scene_update_task_list = list()

    # 重新打开文件
    dotenv = settings.environment.open_dotenv()
    scenes_to_init = json.loads(dotenv["init_scenes"])

    for product_id, scene_id, group_config, env in scenes_to_init:
        scene_update_begin = time.time()
        global_config_file = settings.scenes_configs_path / "global_config.json"
        global_config = settings.json_config.sanitize_by_filename(global_config_file.as_posix())

        MakeConfig.make_sure_node_config_file(scene_id, group_config)

        node_config_file = settings.scenes_configs_path / scene_id / "node_config.json"

        try:
            scene_hash = get_scene_hash(
                global_config,
                product_id,
                scene_id
            )
        except Exception as e:
            logger.error("get scene hash failed. scene_id: {}".format(scene_id))
            logger.error(e)
            logger.error("traceback: {}".format(traceback.format_exc()))
            continue

        last_scene_hash = scene_id2scene_hash.get(scene_id, None)
        if scene_hash == last_scene_hash:
            logger.info("same hash, scene_id: {}, scene_hash: {}".format(scene_id, scene_hash))
            continue
        else:
            logger.info("init scene, scene_id: {}, scene_hash: {}, last_scene_hash: {}. ".format(scene_id, scene_hash, last_scene_hash))

            try:
                scene_instance = scene_initialization(
                    product_id, scene_id, group_config, env,
                    scene_hash,
                    global_config_file, node_config_file,
                )
                scene_pkl_filename = settings.temp_update_directory / "{}_{}_{}_{}.pkl".format(
                    product_id, scene_id, env, scene_hash)
                # 当有些实例不能被 pickle 保存时, 会报错.
                with open(scene_pkl_filename.as_posix(), "wb") as f:
                    pickle.dump(scene_instance, f)

            except Exception as e:
                logger.error("scene init failed. scene_id: {}".format(scene_id))
                logger.error(e)
                logger.error("traceback: {}".format(traceback.format_exc()))
                continue

            del scene_instance

            scene_id2scene_hash[scene_id] = scene_hash
            scene_update_task_list.append({
                "scene_pkl_filename": scene_pkl_filename.as_posix(),
                "operator": "update"
            })
        logger.info("scene: {}, update time cost: {}".format(scene_id, time.time() - scene_update_begin))

    # 保存 update 文件, run server 根据此文件判断是否需要更新.
    if len(scene_update_task_list) != 0:
        scene_update_task_list_json = os.path.join(settings.temp_update_directory, "scene_update_task_list.json")
        with open(scene_update_task_list_json, "w", encoding="utf-8") as f:
            json.dump(scene_update_task_list, f, ensure_ascii=False, indent=4)

    # 保存 scene_id2scene_hash, 将现有的场景与ID保存起来.
    with open(scene_id2scene_hash_filename, "w", encoding="utf-8") as f:
        json.dump(scene_id2scene_hash, f)

    # 删除单例.
    ParamsSingleton.flush()

    logger.info("update time cost: {}".format(time.time() - update_begin))
    return


if __name__ == "__main__":
    update()

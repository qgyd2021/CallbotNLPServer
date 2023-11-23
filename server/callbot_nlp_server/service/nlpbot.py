#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import logging
import os
import pickle
from typing import List
import time

from glob import glob
import shutil
from tqdm import tqdm

from nxtech.nlpbot.node import RequestContext, Node
from nxtech.database.mysql_connect import MySqlConnect
from server.exception import ExpectedError
from nxtech.table_lib.common import is_end_node
from nxtech.table_lib.t_dialog_node_info import TDialogNodeInfo
from nxtech.table_lib.t_dialog_edge_info import TDialogEdgeInfo
from server.callbot_nlp_server import settings
from toolbox.os.command import ps_ef_grep, Command

logger = logging.getLogger("server")


class MakeConfig(object):

    @classmethod
    def make_config_for_node(cls, product_id: str, scene_id: str, env: str,
                             node_id: str, node_type: str, node_desc: str,
                             global_config: dict, node_config: dict):
        """给定数据库, ES 连接信息, 给定 product_id, scene_id, node_id, 生成其模板. """
        result = dict()

        result.update({
            "product_id": product_id,
            "scene_id": scene_id,
            "env": env,
            "node_id": node_id,
            "node_type": node_type,
            "node_desc": node_desc,
        })
        result.update(global_config)
        result["node"] = node_config
        return result

    @classmethod
    def make_config_for_scene(cls, product_id: str, scene_id: str, env: str,
                              global_config: dict, node_config: dict,
                              t_dialog_node_info: TDialogNodeInfo,
                              t_dialog_edge_info: TDialogEdgeInfo,
                              ):

        df = t_dialog_node_info.data
        df = df[df["product_id"] == product_id]
        df = df[df["scene_id"] == scene_id]
        df = df[df["node_type"] == 1]

        result = list()
        for i, row in df.iterrows():
            product_id = row["product_id"]
            scene_id = row["scene_id"]
            node_id = row["node_id"]
            node_desc = row["node_desc"]
            node_type = row["node_type"]

            # 判断 node_id 是否是结束节点.
            # 如果 node_id 只有一个 dst_node_id, 且该 dst_node_id 的 node_type = 7. 则该 node_id 是结束节点.
            if is_end_node(
                product_id=product_id,
                scene_id=scene_id,
                node_id=node_id,
                t_dialog_node_info=t_dialog_node_info,
                t_dialog_edge_info=t_dialog_edge_info,
            ):
                continue

            config = cls.make_config_for_node(
                product_id=product_id,
                scene_id=scene_id,
                env=env,
                node_id=node_id,
                node_type=node_type,
                node_desc=node_desc,
                global_config=global_config,
                node_config=node_config,
            )
            result.append(config)
        return result

    @classmethod
    def make_scene_config(cls, product_id, scene_id, env, global_config_file, node_config_file):

        global_config = settings.json_config.sanitize_by_filename(filename=global_config_file)
        node_config = settings.json_config.sanitize_by_filename(filename=node_config_file)

        mysql_connect = global_config["mysql_connect"]

        mysql_connect = MySqlConnect.from_json(mysql_connect)

        result = cls.make_config_for_scene(
            product_id=product_id,
            scene_id=scene_id,
            env=env,
            global_config=global_config,
            node_config=node_config,
            t_dialog_node_info=TDialogNodeInfo(
                scene_id=scene_id,
                mysql_connect=mysql_connect
            ),
            t_dialog_edge_info=TDialogEdgeInfo(
                scene_id=scene_id,
                mysql_connect=mysql_connect
            )
        )
        return result

    @classmethod
    def make_scene_config_file(cls, product_id, scene_id, env, global_config_file, node_config_file, to_filename):

        # 从 node_config 创建 scene 的配置.
        scene_config = cls.make_scene_config(product_id, scene_id, env, global_config_file, node_config_file)

        with open(to_filename, "w", encoding="utf-8") as f:
            json.dump(scene_config, f, ensure_ascii=False, indent=4)
        return scene_config

    @classmethod
    def make_sure_node_config_file(cls, scene_id: str, group: str):
        """如果 node_config_file 不存在, 则 copy default 中对应 group 的配置. """
        node_config_file_path = settings.scenes_configs_path / scene_id
        if not node_config_file_path.exists():
            shutil.copytree(
                src=(settings.group_configs_path / group).as_posix(),
                dst=node_config_file_path.as_posix()
            )
        elif not (node_config_file_path / "node_config.json").exists():
            shutil.rmtree(
                path=node_config_file_path.as_posix(),
            )
            shutil.copytree(
                src=(settings.group_configs_path / group).as_posix(),
                dst=node_config_file_path.as_posix()
            )
        else:
            pass
        return


class NlpBot(object):
    def __init__(self):
        # node_dict:
        # key 格式为: "{}_{}_{}".format(product_id, scene_id, node_id)
        # value 是 Node 对象. 用于召回算分候选句子.
        self.node_dict = dict()

        self.no_update_count = 0
        self.max_no_update_count = 300

        self.update__check_and_init_node()

        self.last_scene_update_task_list = None

    def this_is_min_pid(self):
        """
        ps -ef | grep run_callbot_nlp_server.py | awk '{print $2}'
        :return:
        """
        cmd = "ps -ef | grep run_callbot_nlp_server.py | grep -v 'grep' | awk '{print $2}'"
        # logger.info(">>> {}".format(cmd))
        output_string = Command.popen(cmd)

        rows = str(output_string).split("\n")
        rows = [str(row).strip() for row in rows]

        pid_list = [int(row) for row in rows if len(row) != 0]

        if len(pid_list) == 1:
            min_pid = min(pid_list)
        else:
            min_pid = min(pid_list[1:])

        if min_pid == os.getpid():
            logger.info("\n{}".format(output_string))
            return True
        else:
            return False

    def _init(self):
        # tornado 多进程情况下, 只有一个进程会执行更新操作.
        filename = os.path.join(settings.temp_update_directory, "process_lock.txt")

        pid = os.getpid()
        with open(filename, "a+", encoding="utf-8") as f:
            f.write("{}\n".format(pid))
        with open(filename, "r", encoding="utf-8") as f:
            first_line = f.readline()
            first_line = str(first_line).strip()
        if first_line == pid:
            self.do_update = True
        return

    def _make_scene_config(self, product_id, scene_id, group, env):
        scene_config_file = os.path.join(
            settings.node_config_path,
            "scenes", scene_id,
            "{}_{}_{}.json".format(product_id, scene_id, env)
        )
        if settings.override_scene_config or not os.path.exists(scene_config_file):
            logger.info("make scene config file {}".format(scene_config_file))
            MakeConfig.make_sure_node_config_file(scene_id, group)
            MakeConfig.make_scene_config_file(
                product_id=product_id,
                scene_id=scene_id,
                env=env,
                global_config_file=os.path.join(
                    settings.node_config_path,
                    env, "global_config.json"
                ),
                node_config_file=os.path.join(
                    settings.node_config_path,
                    "scenes",
                    scene_id,
                    "node_config.json"
                ),
                to_filename=scene_config_file
            )
        return scene_config_file

    def _add_scene(self, scene_config: List[dict]):
        for config in scene_config:
            self._add_node(
                config=config,
            )
        return

    def _add_node(self, config: dict):
        product_id = config["product_id"]
        scene_id = config["scene_id"]
        env = config["env"]

        node_id = config["node_id"]

        params = config.pop("node")
        global_params = config

        key = "{}_{}_{}_{}".format(product_id, scene_id, env, node_id)
        if key in self.node_dict:
            raise ExpectedError(
                status_code=60405,
                message="node already exist. key: {}".format(key)
            )
        else:
            logger.info("add node: {}".format(key))
            node = Node.from_json(params=params, global_params=global_params)
            self.node_dict[key] = node
        return node

    async def recommend(self, context: RequestContext) -> RequestContext:
        product_id = context.product_id
        scene_id = context.scene_id
        node_id = context.node_id
        env = context.env

        key = "{}_{}_{}_{}".format(product_id, scene_id, env, node_id)
        node: Node = self.node_dict.get(key)
        if node is None:
            raise ExpectedError(
                status_code=60405,
                message="node not found. ",
                detail="product_id: {}, scene_id: {}, node_id: {}".format(product_id, scene_id, node_id)
            )

        recommend: RequestContext = await node.async_recommend(context)
        return recommend

    def add_scene(self, product_id: str, scene_id: str, group: str, env: str):
        scene_config_file = self._make_scene_config(product_id, scene_id, group, env)
        scene_config = settings.json_config.sanitize_by_filename(filename=scene_config_file)

        logger.info("init scene, node count: {}, scene_config_file: {}".format(len(scene_config), scene_config_file))
        self._add_scene(
            scene_config=scene_config,
        )
        return

    def delete_scene(self, product_id: str, scene_id: str, group: str, env: str):
        prefix = "{}_{}_{}".format(product_id, scene_id, env)

        result = list()
        for k, v in self.node_dict.items():
            if k.startswith(prefix):
                result.append(k)

        for k in result:
            self.node_dict.pop(k)
        return result

    def update__check_and_init_node(self):
        """
        先检查 update_callbot_nlp_server.py 进程是否存在, 如果存在, 则不更新.

        """
        if not self.this_is_min_pid():
            # 进程 id 最小的进程, 执行 update 操作.
            return

        logger.info("pid: {}, update__check_and_init_node".format(os.getpid()))

        rows = ps_ef_grep("update_callbot_nlp_server.py")
        if len(rows) == 0:
            cmd = "cd /data/tianxing/PycharmProjects/CallbotNLPServer/server/callbot_nlp_server/"
            logger.info(">>> {}".format(cmd))
            Command.popen(cmd)

            cmd = "nohup python3 update_callbot_nlp_server.py > nohup_update.out &"
            logger.info(">>> {}".format(cmd))
            Command.popen(cmd)
        else:
            logger.info("pid: process running".format(os.getpid()))
            for row in rows:
                logger.info(row)
        return

    def update__replace_node(self):
        logger.info("pid: {}, update__replace_node".format(os.getpid()))
        replace_begin = time.time()

        scene_update_task_list_json = os.path.join(settings.temp_update_directory, "scene_update_task_list.json")
        if not os.path.exists(scene_update_task_list_json):
            logger.info("pid: {}, scene_update_task_list.json not exist. no update task. ".format(os.getpid()))
            return

        with open(scene_update_task_list_json, "r", encoding="utf-8") as f:
            scene_update_task_list: List[dict] = json.load(f)

        if scene_update_task_list == self.last_scene_update_task_list:
            logger.info("pid: {}, scene_update_task_list.json is updated. no update task. ".format(os.getpid()))
            return

        # 将不在 scene_update_task_list 中的 pkl 文件都删掉.
        pkl_list = [task["scene_pkl_filename"] for task in scene_update_task_list]
        filename_pattern = os.path.join(settings.temp_update_directory, "*.pkl")
        for filename in glob(filename_pattern):
            if filename not in pkl_list:
                # server 多进程, 可能被其它进程删除.
                try:
                    os.remove(filename)
                except (FileNotFoundError,):
                    continue

        # pkl 更新.
        for task in tqdm(scene_update_task_list):
            scene_pkl_filename = task["scene_pkl_filename"]

            logger.info("pid: {}, load file: {}".format(os.getpid(), scene_pkl_filename))
            with open(scene_pkl_filename, "rb") as f:
                scene = pickle.load(f)
            # server 多进程, 不能删除.
            # os.remove(scene_pkl_filename)

            product_id = scene["product_id"]
            scene_id = scene["scene_id"]
            env = scene["env"]
            node_list = scene["node_list"]

            # 删除原有的节点 (如果是新添加节点, 则没有原有节点, 仍然是 OK 的).
            prefix = "{}_{}_{}".format(product_id, scene_id, env)
            key_list = list()
            for k, v in self.node_dict.items():
                if k.startswith(prefix):
                    key_list.append(k)
            for k in key_list:
                self.node_dict.pop(k)

            # 添加新的节点
            for node_dict in node_list:
                node = node_dict["node"]
                global_params = node_dict["global_params"]
                node_id = global_params["node_id"]

                key = "{}_{}_{}_{}".format(product_id, scene_id, env, node_id)
                self.node_dict[key] = node

        self.last_scene_update_task_list = scene_update_task_list
        logger.info("pid: {}, update__replace_node time cost: {}, task count: {}".format(
            os.getpid(), time.time() - replace_begin, len(scene_update_task_list)
        ))
        return


nlpbot = NlpBot()

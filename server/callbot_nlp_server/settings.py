# -*- encoding=UTF-8 -*-
import json
import os

from project_settings import project_path
from toolbox.os.environment import EnvironmentManager, JsonConfig

log_directory = project_path / "server/callbot_nlp_server/log"
log_directory.mkdir(exist_ok=True)
temp_directory = project_path / "server/callbot_nlp_server/temp"
temp_directory.mkdir(exist_ok=True)
temp_update_directory = project_path / temp_directory / "update"
temp_update_directory.mkdir(exist_ok=True)

environment = EnvironmentManager(
    path=os.path.join(project_path, "server/callbot_nlp_server/dotenv"),
    env=os.environ.get("environment", "hk_dev"),
    override=True
)

json_config = JsonConfig(
    environment=environment
)

port = environment.get(key="port", default=9080, dtype=int)

num_processes = environment.get(key="num_processes", default=1, dtype=int)

init_scenes = environment.get(key="init_scenes", dtype=json.loads)

node_config_path = environment.get(key="node_config_path", default="server/callbot_nlp_server/configs", dtype=str)
node_config_path = project_path / node_config_path
scenes_configs_path = node_config_path / "scenes_configs"
group_configs_path = scenes_configs_path / "group_configs"

override_scene_config = environment.get(key="node_config_path", default=True, dtype=bool)

server_update_interval = environment.get(key="server_update_interval", default=5 * 60, dtype=int)

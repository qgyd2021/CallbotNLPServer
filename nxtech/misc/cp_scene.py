#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os

from toolbox.os.command import Command

from project_settings import project_path


def cp_scene(from_scene_id, to_scene_id):
    """
    创建 shell 文件
    /data/callbot/cpscene/cp_scene.sh
    """
    cp_scene_sh = """
    #!/bin/bash

    mysql_host='10.20.251.13'
    mysql_port=3306
    mysql_user='callbot'
    mysql_pass='NxcloudAI2021!'

    db='callbot_ppe'
    product_id='callbot'
    from_scene_id='{from_scene_id}'
    to_scene_id='{to_scene_id}'

    from_sql="$from_scene_id.sql"
    to_sql="$to_scene_id.sql"


    tables=(
    t_callcentre_intent_info
    t_callcentre_intent_rule_info
    t_dialog_dial_strategy_action_info
    t_dialog_dial_strategy_info
    t_dialog_edge_info
    t_dialog_ignore_node_info
    t_dialog_language_info
    t_dialog_node_info
    t_dialog_params_info
    t_dialog_resource_info
    t_dialog_scene_info
    t_cb_process_master
    t_cb_question
    t_cb_record_feed
    t_cb_scenes
    )
    # t_cb_scene_category #sha

    #echo ${{tables[@]}}

    # -c: insert时候带上字段，-t：只需要数据不需要结构
    #mysqldump -h $mysql_host -P $mysql_port -u$mysql_user -p$mysql_pass -c -t \
    #	--where="product_id='${{product_id}}' and scene_id='${{from_scene_id}}'" $db ${{tables[@]}} > ./sql/$from_sql


    #sed -i "s/'$from_scene_id'/'$to_scene_id'/g" ./sql/${{from_sql}}
    #for table in ${{tables[@]}};do
    #	#echo $table
    #    mysql -h$mysql_host -P$mysql_port -u$mysql_user -p$mysql_pass $db -e"delete from $table"
    #done
    #
    #mysql -h$mysql_host -P$mysql_port -u$mysql_user -p$mysql_pass $db < ./sql/$from_sql
    #


    to_scene_dir=$to_scene_id
    mkdir ./$to_scene_dir

    # dump
    echo "---------------------------------"
    for table in ${{tables[@]}};do
        echo "mysqldump $table"
        mysqldump -h $mysql_host -P $mysql_port -u$mysql_user -p$mysql_pass -c -t --skip-disable-keys --skip-add-locks  --single-transaction \
    	    --where="product_id='${{product_id}}' and scene_id='${{from_scene_id}}'" $db ${{table}} > ./$to_scene_dir/${{table}}.sql
    done


    # replace
    echo "---------------------------------"
    for table in ${{tables[@]}};do
        echo "replace $table"
        sed -i "s/'$from_scene_id'/'$to_scene_id'/g" ./$to_scene_dir/${{table}}.sql
    done

    # insert
    echo "---------------------------------"
    for table in ${{tables[@]}};do
        echo "insert $table"
        mysql -h$mysql_host -P$mysql_port -u$mysql_user -p$mysql_pass $db < ./$to_scene_dir/${{table}}.sql
    done

    """.format(
        from_scene_id=from_scene_id,
        to_scene_id=to_scene_id,
    )
    pwd = os.path.abspath(os.path.dirname(__file__))

    filename = os.path.join(pwd, 'cp_scene.sh')

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(cp_scene_sh)

    Command.popen('chmod 777 {}'.format(filename))
    Command.popen('sh {}'.format(filename))
    return

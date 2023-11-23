#!/usr/bin/python3
# -*- coding: utf-8 -*-
from nxtech.table_lib.t_dialog_node_info import TDialogNodeInfo
from nxtech.table_lib.t_dialog_edge_info import TDialogEdgeInfo


def is_end_node(
    product_id: str,
    scene_id: str,
    node_id: str,
    t_dialog_node_info: TDialogNodeInfo,
    t_dialog_edge_info: TDialogEdgeInfo
) -> bool:
    """判断主流程中的节点是否是结束节点. """
    rows = t_dialog_edge_info.get_rows_by_node_id(
        product_id=product_id,
        scene_id=scene_id,
        node_id=node_id,
    )
    if len(rows) == 0:
        raise ValueError('node_id not exist. product_id: {}, scene_id: {}, node_id: {}'.format(
            product_id, scene_id, node_id
        ))
    if len(rows) != 1:
        return False
    dst_node_id = rows[0]['dst_node_id']

    rows = t_dialog_node_info.get_rows_by_node_id(
        product_id=product_id,
        scene_id=scene_id,
        node_id=dst_node_id,
    )
    node_type = rows[0]['node_type']
    if node_type == 7:
        return True
    return False


if __name__ == '__main__':
    pass

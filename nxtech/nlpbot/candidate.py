# -*- encoding=UTF-8 -*-
import copy
import os
import sys
import time
from typing import Dict, List

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../'))

from cacheout import Cache
import pandas as pd

from nxtech.common.params import Params
from nxtech.table_lib.t_dialog_node_info import TDialogNodeInfo
from nxtech.table_lib.t_dialog_edge_info import TDialogEdgeInfo
from nxtech.table_lib.t_dialog_resource_info import TDialogResourceInfo
from nxtech.table_lib.t_know_basic_info import TKnowBasicInfo
from nxtech.table_lib.t_know_word_node_info import TKnowWordNodeInfo
from nxtech.table_lib.t_know_word_info import TKnowWordInfo
from nxtech.database.mysql_connect import MySqlConnect


class RecallCandidates(Params):
    """
    召回器的候选.
    """
    def __init__(self):
        super().__init__()

    def get(self) -> List[dict]:
        raise NotImplementedError


@RecallCandidates.register('list')
class ListCandidates(RecallCandidates):
    def __init__(self, candidates: List[dict]):
        super().__init__()

        self.candidates = candidates

    def get(self) -> List[dict]:
        return self.candidates


@RecallCandidates.register('mysql_deprecated')
class MySqlCandidatesDeprecated(RecallCandidates):
    cache = Cache(maxsize=256, ttl=1 * 60, timer=time.time)

    @staticmethod
    def demo1():
        from nxtech.database.mysql_connect import MySqlConnect

        product_id = 'callbot'
        # scene_id = 'ad6e2oq406'
        # node_id = '51def72a-b086-4f6b-a3d5-45165d02dc10'

        # scene_id = 'u47kpyc43ttk'
        # node_id = '51def72a-b086-4f6b-a3d5-45165d02dc10'

        scene_id = '9qtb38k09cpb'
        node_id = '51def72a-b086-4f6b-a3d5-45165d02dc10'

        mysql_connect = MySqlConnect(
            host='10.20.251.13',
            # host='10.52.66.41',
            port=3306,
            user='callbot',
            password='NxcloudAI2021!',
            database='callbot_ppe',
            charset='utf8'
        )

        candidates: List[dict] = MySqlCandidates(
            product_id=product_id,
            scene_id=scene_id,
            node_id=node_id,
            mysql_connect=mysql_connect,
        ).get()

        # uy9oiixjjqw0hawr
        node_type_list = [candidate['node_type'] for candidate in candidates]
        print(list(set(node_type_list)))

        node_type_list = [candidate['node_id'] for candidate in candidates]
        print(list(set(node_type_list)))

        return

    @staticmethod
    def demo2():
        """
{
    'product_id': 'callbot', 'scene_id': 'ad6e2oq406', 'group_id': '4w44xdmbm4270b23',
    'resource_id': 'MUT6zsxHiZFCJhbvacI6D', 'resource_type': 3, 'node_id': 'ylvj075cipbuu23c',
    'node_desc': '询问审批通过率', 'node_type': 4, 'text': '.*?(审批|审核).*?(怎么|咋|如何|难不难|难度).*?'
}
{
    'product_id': 'callbot', 'scene_id': 'ad6e2oq406', 'group_id': '4w44xdmbm4270b23',
    'resource_id': 'rijMC8c1-iG6igEPNP8lj', 'resource_type': 3, 'node_id': 'ylvj075cipbuu23c',
    'node_desc': '询问审批通过率', 'node_type': 4, 'text': '.*?(通过率|成功率).*?(怎么|咋|如何).*?'
}

        """
        from nxtech.database.mysql_connect import MySqlConnect

        product_id = 'callbot'
        scene_id = 'ad6e2oq406'
        node_id = '51def72a-b086-4f6b-a3d5-45165d02dc10'

        mysql_connect = MySqlConnect(
            # host='10.20.251.13',
            host='10.52.66.41',
            port=3306,
            user='callbot',
            password='NxcloudAI2021!',
            database='callbot_ppe',
            charset='utf8'
        )

        candidates: List[dict] = MySqlCandidates(
            product_id=product_id,
            scene_id=scene_id,
            node_id=node_id,
            mysql_connect=mysql_connect,
            resource_type_list=[3],
        ).get()

        for candidate in candidates:
            print(candidate)

        return

    def _get_global_candidates(self,
                               add_product_id: str,
                               add_scene_id: str,
                               resource_type_list: List[int],
                               t_dialog_node_info: TDialogNodeInfo,
                               t_dialog_resource_info: TDialogResourceInfo):
        key = '{}_{}_{}'.format(add_product_id, add_scene_id, resource_type_list)
        value = self.cache.get(key)
        if value is not None:
            return value

        global_candidates = list()
        for i, row in t_dialog_node_info.data.iterrows():
            product_id = row['product_id']
            scene_id = row['scene_id']
            node_id = row['node_id']
            node_desc = row['node_desc']
            node_type = row['node_type']
            group_id = row['intent_res_group_id']

            if product_id != add_product_id:
                continue
            if scene_id != add_scene_id:
                continue
            # type, 1 是主流程, 11 是噪声节点.
            if node_type in (0, 1, 11):
                continue
            if pd.isna(group_id):
                continue
            if len(group_id) == 0:
                continue

            rows = t_dialog_resource_info.get_rows_by_group_id(
                product_id=product_id,
                scene_id=scene_id,
                group_id=group_id,
            )
            # if len(rows) == 0:
            #     raise AssertionError('group_id: {} invalid'.format(group_id))

            for resource_row in rows:
                text = resource_row['word']
                if pd.isna(text) or len(text) == 0:
                    continue

                resource_type = resource_row['res_type']
                # type 为 1 的是意图词, 2 是机器话术, 3 是正则表达式.
                if len(resource_type_list) != 0 and resource_type not in resource_type_list:
                    continue

                global_candidates.append({
                    'product_id': resource_row['product_id'],
                    'scene_id': resource_row['scene_id'],
                    'group_id': group_id,
                    'resource_id': resource_row['res_id'],
                    'resource_type': resource_type,
                    'node_id': node_id,
                    'node_desc': node_desc,
                    'node_type': node_type,
                    'text': resource_row['word'],
                })

        self.cache.set(key, global_candidates)
        return global_candidates

    def __init__(self,
                 product_id: str,
                 scene_id: str,
                 node_id: str,
                 mysql_connect: MySqlConnect,
                 resource_type_list: List[int] = None,
                 main_take_precedence: bool = True,
                 ):
        """

        :param product_id:
        :param scene_id:
        :param node_id:
        :param mysql_connect:
        :param resource_type_list: resource_type 为 1 的是意图词, 2 是机器话术, 3 是正则表达式. 默认是 [1], 可以传入 [3] 来获取正则表达式.
        :param resource_type_list: main_take_precedence
        """
        super().__init__()

        if len(node_id) == 0:
            raise AssertionError('node_id should not be length 0')
        self.product_id = product_id
        self.scene_id = scene_id
        self.node_id = node_id
        self.mysql_connect = mysql_connect
        # 默认生成意图句子的候选结果.
        self.resource_type_list = resource_type_list or [1]
        self.main_take_precedence = main_take_precedence

        # Table 表都是单例, 不存在重复下载数据的情况.
        self.t_dialog_node_info = TDialogNodeInfo(
            scene_id=scene_id,
            mysql_connect=self.mysql_connect
        )
        self.t_dialog_edge_info = TDialogEdgeInfo(
            scene_id=scene_id,
            mysql_connect=self.mysql_connect
        )
        self.t_dialog_resource_info = TDialogResourceInfo(
            scene_id=scene_id,
            mysql_connect=self.mysql_connect,
        )

    def get(self) -> List[dict]:
        global_candidates = self._get_global_candidates(
            add_product_id=self.product_id,
            add_scene_id=self.scene_id,
            resource_type_list=self.resource_type_list,
            t_dialog_node_info=self.t_dialog_node_info,
            t_dialog_resource_info=self.t_dialog_resource_info,
        )

        rows = self.t_dialog_edge_info.get_rows_by_node_id(
            product_id=self.product_id,
            scene_id=self.scene_id,
            node_id=self.node_id,
        )
        dst_node_id_list = [row['dst_node_id'] for row in rows]

        main_candidates = list()
        for dst_node_id in dst_node_id_list:
            rows = self.t_dialog_node_info.get_rows_by_node_id(
                product_id=self.product_id,
                scene_id=self.scene_id,
                node_id=dst_node_id,
            )
            if len(rows) == 0:
                raise AssertionError('node_id: {} invalid'.format(dst_node_id))
            row = rows[0]
            node_id = row['node_id']
            node_desc = row['node_desc']
            node_type = row['node_type']

            group_id_list = [row['intent_res_group_id'] for row in rows if len(row['intent_res_group_id']) != 0]
            group_id_list = [group_id for group_id in group_id_list if not pd.isna(group_id)]

            for group_id in group_id_list:
                rows = self.t_dialog_resource_info.get_rows_by_group_id(
                    product_id=self.product_id,
                    scene_id=self.scene_id,
                    group_id=group_id,
                )

                for row in rows:
                    resource_type = row['res_type']
                    # type 为 1 的是意图词, 2 是机器话术, 3 是正则表达式.
                    if len(self.resource_type_list) != 0 and resource_type not in self.resource_type_list:
                        continue

                    main_candidates.append({
                        'product_id': row['product_id'],
                        'scene_id': row['scene_id'],
                        'group_id': group_id,
                        'resource_id': row['res_id'],
                        'resource_type': resource_type,
                        'node_id': node_id,
                        'node_desc': node_desc,
                        'node_type': node_type,
                        'text': row['word'],
                    })

        # 如果主流程中没有下流话术, 则返回空列表.
        # if len(main_candidates) == 0:
        #     return list()

        if self.main_take_precedence:
            # 如果 global 中的话术已在主流程中出现, 则不要 global 中的该话术.
            result = copy.deepcopy(main_candidates)
            main_wordings = [main_candidate['text'] for main_candidate in main_candidates]
            for global_candidate in global_candidates:
                text = global_candidate['text']
                if text in main_wordings:
                    continue
                else:
                    result.append(global_candidate)
        else:
            result = copy.deepcopy(global_candidates)
            global_wordings = [global_candidate['text'] for global_candidate in global_candidates]
            for main_candidate in main_candidates:
                text = main_candidate['text']
                if text in global_wordings:
                    continue
                else:
                    result.append(main_candidate)
        return result


@RecallCandidates.register('mysql')
class MysqlCandidatesIncludeIntentLib(RecallCandidates):
    """
    Read candidates from mysql, include intent library.

    resource_type_list:
    There is [相似问, 白名单正则, 黑名单正则, 意图库-标准意图, 意图库-意图分类, 意图库-相似问, 意图库-正则表达式],
    It is named in english as : [
        similar_question , white_regex, black_regex,
        lib_intent, lib_class, lib_similar_question, lib_regex
    ].

    priority:
    There is a function to control the order of : [主流程, 业务问答, 通用对话].
    We name it as [main process, frequency question answer, dialogue control],
    shortening [main, faq, ctrl].

    main: node_type_list: [1]
    faq: node_type_list: [4]
    ctrl: node_type_list: [2, 5]

    TDialogNodeInfo
    node_type
    1 //主流程节点
    2 //非主流程节点之用户要求重复
    3 //非主流程节点之用户无回应（控制类）
    4 //非主流程节点之开放问题
    5 //非主流程节点之无意义回应
    6 //非主流程节点之打断节点（控制类）（此类型节点在一个场景中有且只有一个）
    7 //结束节点（控制类）（此类型节点在一个场景中有且只有一个）
    8 //超过轮次时候结束的节点（此类型节点在一个场景中有且只有一个）
    9 //系统异常时候，强制结束（此类型节点在一个场景中有且只有一个）
    10 //全局挽回节点，用于匹配中忽略节点时候使用（此类型节点在一个场景中有且只有一个）
    11 //全局噪音节点 用于忽略该次输入
    12 //跳转人工节点 用于跳转人工客服

    TDialogResourceInfo
    res_type
    1 //意图词
    2 //机器话术
    3 //白名单正则
    4 //黑名单正则
    """

    cache = Cache(maxsize=256, ttl=1 * 60, timer=time.time)

    resource_type_text_to_int: Dict[str, int] = {
        'similar_question': 1,
        'white_regex': 3,
        'black_regex': 4,
    }

    node_type_main = 'main'
    node_type_faq = 'faq'
    node_type_ctrl = 'ctrl'

    @staticmethod
    def demo1():
        area = 'hk'
        if area == 'gz':
            host = '10.20.251.13'
            password = 'wm%msjngbtmheh3TdqYbmgg3s@nxprd230417'
        elif area == 'hk':
            host = '10.52.66.41'
            password = 'SdruuKtzmjexpq%dj6mu9qryk@nxprd230413'
        elif area == 'mx':
            host = '172.16.1.149'
            password = 'Vstr2ajjlYeduvf7bu%@nxprd230417'
        else:
            raise AssertionError

        mysql_candidates = MysqlCandidatesIncludeIntentLib(
            product_id='callbot',
            scene_id='cdg26b89j98y',
            node_id='53ca5c4b-c28c-43f3-be22-49cd803a25de',
            mysql_connect=MySqlConnect(
                host=host,
                port=3306,
                user='nx_prd',
                password=password,
                database='callbot_ppe',
            ),
            resource_type_list=['similar_question', 'white_regex', 'black_regex'],
            priority_list=None
        )

        candidates = mysql_candidates.get()

        node_type_list = [candidate['node_type'] for candidate in candidates]
        node_id_list = [candidate['node_id'] for candidate in candidates]
        node_desc_list = [candidate['node_desc'] for candidate in candidates]
        print(set(node_type_list))
        print(node_type_list)
        print(node_id_list)
        # print(node_desc_list)
        return

    def __init__(self,
                 product_id: str,
                 scene_id: str,
                 node_id: str,
                 mysql_connect: MySqlConnect,
                 resource_type_list: List[str],
                 priority_list: List[str] = None,
                 ):
        super().__init__()
        if len(node_id) == 0:
            raise AssertionError('node_id should not be length 0')
        if len(resource_type_list) == 0:
            raise AssertionError('resource_type_list should not be empty')
        if any([t not in self.resource_type_text_to_int.keys() for t in resource_type_list]):
            raise AssertionError('some resource type {} not in {}'.format(
                resource_type_list, self.resource_type_choice))

        self.product_id = product_id
        self.scene_id = scene_id
        self.node_id = node_id
        self.mysql_connect = mysql_connect
        self.resource_type_list: List[int] = [self.resource_type_text_to_int[t] for t in resource_type_list]
        self.priority_list = priority_list or [self.node_type_main, self.node_type_faq, self.node_type_ctrl]

        # Table is singleton.
        self.t_dialog_node_info = TDialogNodeInfo(
            product_id=self.product_id,
            scene_id=self.scene_id,
            mysql_connect=self.mysql_connect
        )
        self.t_dialog_edge_info = TDialogEdgeInfo(
            product_id=self.product_id,
            scene_id=self.scene_id,
            mysql_connect=self.mysql_connect
        )
        self.t_dialog_resource_info = TDialogResourceInfo(
            product_id=self.product_id,
            scene_id=self.scene_id,
            mysql_connect=self.mysql_connect,
        )
        self.t_know_word_node_info = TKnowWordNodeInfo(
            product_id=self.product_id,
            scene_id=self.scene_id,
            mysql_connect=self.mysql_connect,
        )

        self.name_to_node_type_list: Dict[str, List[int]] = {
            self.node_type_main: [1],
            self.node_type_faq: [4],
            self.node_type_ctrl: [2, 5],
        }

    def get_candidates_by_node_type_list(self, node_type_list: List[int], target_node_id: str = None) -> List[dict]:
        key = '{func_name}_{product_id}_{scene_id}_{resource_type_list}_' \
              '{priority_list}_{node_type_list}_{target_node_id}'.format(
                func_name='self.get_candidates_by_node_type_list',
                product_id=self.product_id,
                scene_id=self.scene_id,
                resource_type_list=self.resource_type_list,
                priority_list=self.priority_list,
                node_type_list='node_type_list: {}'.format(node_type_list),
                target_node_id='target_node_id: {}'.format(target_node_id))

        value = self.cache.get(key)
        if value is not None:
            return value

        result = list()

        for i, row in self.t_dialog_node_info.data.iterrows():
            # (1)
            # resource in intent lib not belong to any scene,
            # therefore, the 'scene_id' is fill with intent lib identity 'lang_country'
            # (2)
            # 'word_name' is the intent name of resource.
            # I want to know what are the intent names ('word_name') for each 'node_id'.
            # List[Tuple[scene_id_or_lang_country, group_id, none_or_word_name]]
            group_id_list = list()

            product_id = row['product_id']
            scene_id = row['scene_id']
            node_id = row['node_id']
            node_desc = row['node_desc']
            node_type = row['node_type']
            dialog_node_info_group_id = row['intent_res_group_id']

            if product_id != self.product_id:
                continue
            if scene_id != self.scene_id:
                continue
            if node_type not in node_type_list:
                continue
            if target_node_id is not None and node_id != target_node_id:
                continue

            group_id_list.append((scene_id, dialog_node_info_group_id, None))

            # 意图库配置
            know_word_node_info_rows = self.t_know_word_node_info.get_rows_by_node_id(
                product_id=product_id,
                scene_id=scene_id,
                node_id=node_id,
            )
            for know_word_node_info_row in know_word_node_info_rows:
                basic_id = know_word_node_info_row['basic_id']
                word_id = know_word_node_info_row['word_id']

                t_know_word_info = TKnowWordInfo(
                    product_id=self.product_id,
                    basic_id=basic_id,
                    mysql_connect=self.mysql_connect,
                )
                know_word_info_rows = t_know_word_info.get_rows_by_word_id(
                    product_id=product_id,
                    basic_id=basic_id,
                    word_id=word_id,
                )

                t_know_basic_info = TKnowBasicInfo(
                    product_id=self.product_id,
                    basic_id=basic_id,
                    mysql_connect=self.mysql_connect,
                )
                know_basic_info_rows = t_know_basic_info.get_rows_by_basic_id(
                    product_id=product_id,
                    basic_id=basic_id,
                )
                if len(know_basic_info_rows) == 0:
                    continue
                lang_country = know_basic_info_rows[0]['lang_country']
                for know_word_info_row in know_word_info_rows:
                    word_name = know_word_info_row['word_name']
                    intent_res_group_id = know_word_info_row['intent_res_group_id']
                    regular_res_group_id = know_word_info_row['regular_res_group_id']
                    group_id_list.append((lang_country, intent_res_group_id, word_name))
                    group_id_list.append((lang_country, regular_res_group_id, word_name))

            # 场景中配置
            for scene_id_or_lang_country, group_id, none_or_word_name, in group_id_list:
                if pd.isna(group_id) and len(group_id) == 0:
                    continue

                dialog_resource_info_rows = self.t_dialog_resource_info.get_rows_by_group_id(
                    product_id=product_id,
                    scene_id=scene_id_or_lang_country,
                    group_id=group_id,
                )
                for dialog_resource_info_row in dialog_resource_info_rows:
                    text = dialog_resource_info_row['word']
                    if pd.isna(text) or len(text) == 0:
                        continue

                    resource_type = dialog_resource_info_row['res_type']
                    if resource_type not in self.resource_type_list:
                        continue

                    result.append({
                        'product_id': dialog_resource_info_row['product_id'],
                        'scene_id': dialog_resource_info_row['scene_id'],
                        'group_id': dialog_node_info_group_id,
                        'resource_id': dialog_resource_info_row['res_id'],
                        'resource_type': resource_type,
                        'node_id': node_id,
                        'node_desc': node_desc,
                        'node_type': node_type,
                        'text': dialog_resource_info_row['word'],
                        'word_name': none_or_word_name,
                    })

        self.cache.set(key, result)
        return result

    def get_main_candidates(self) -> List[dict]:
        # main candidates.
        rows = self.t_dialog_edge_info.get_rows_by_node_id(
            product_id=self.product_id,
            scene_id=self.scene_id,
            node_id=self.node_id,
        )
        dst_node_id_list = [row['dst_node_id'] for row in rows]

        main_candidates = list()
        for dst_node_id in dst_node_id_list:
            node_type_list = self.name_to_node_type_list['main']
            sub_candidates = self.get_candidates_by_node_type_list(
                node_type_list=node_type_list,
                target_node_id=dst_node_id
            )
            main_candidates.extend(sub_candidates)

        return main_candidates

    def get(self) -> List[dict]:
        candidates = list()
        word_set = set()

        # run
        for priority in self.priority_list:
            if priority == 'main':
                sub_candidates = self.get_main_candidates()
            else:
                sub_candidates = self.get_candidates_by_node_type_list(
                    node_type_list=self.name_to_node_type_list[priority],
                )
            for sub_candidate in sub_candidates:
                text = sub_candidate['text']
                if text in word_set:
                    continue
                candidates.append(sub_candidate)
                word_set.add(text)

        return candidates


def demo1():
    MysqlCandidatesIncludeIntentLib.demo1()
    return


if __name__ == '__main__':
    demo1()

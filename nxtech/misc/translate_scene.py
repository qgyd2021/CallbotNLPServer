#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
mysql -h 10.20.251.13 -P 3306 -u root -pNxcloudROOTAI2021! callbot_ppe < callbot_lupmormypq_business_questions.sql
mysql -h 10.20.251.13 -P 3306 -u root -pNxcloudROOTAI2021! callbot_ppe < callbot_lupmormypq_end.sql
mysql -h 10.20.251.13 -P 3306 -u root -pNxcloudROOTAI2021! callbot_ppe < callbot_lupmormypq_front.sql

https://docs.microsoft.com/zh-cn/azure/cognitive-services/speech-service/language-support

"""
from collections import defaultdict
import json
import os
import sys
import time
import random
from tqdm import tqdm
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
from pymysql.converters import escape_string

from nxtech.database.mysql_connect import MySqlConnect
from toolbox.translate.google_translate import GoogleTranslate


class GoogleTranslateScene(object):
    aibot_language2google_language = {
        'zh-CN': 'zh-cn',
        'zh-HK': 'zh-cn',
        'zh-TW': 'zh-cn',
        # 'zh-HK': 'zh-tw',
        # 'zh-TW': 'zh-tw',
        'id-ID': 'id',
        'en-PH': 'en',
        'en-US': 'en',
        'en-NG': 'en',
        'en-IN': 'en',
        'ja-JP': 'ja',
        'th-TH': 'th',
        'vi-VN': 'vi',
        'es-MX': 'es',
        'ms-MY': 'ms',
        'my-MM': 'my',
        'km-KH': 'km',
        'lo-LA': 'lo',
        'pt-BR': 'pt',
        'pt-PT': 'pt',

    }

    @staticmethod
    def google_translate(text: str, dest: str, src: str) -> str:
        text = text.strip()
        if len(text.strip()) != 0:
            print('google_translate: {}'.format(text))
            result = GoogleTranslate.translate(
                text=text,
                dest=dest,
                src=src,
            )
            result = result[0]
        else:
            result = text
        return result

    def __init__(self, mysql_connect: MySqlConnect, root_path: str):
        self.mysql_connect = mysql_connect
        self.root_path = root_path

        self._scene_words_format = '{product_id}_{scene_id}_words.xlsx'
        self._scene_words_translated_json_format = '{product_id}_{scene_id}_translated.json'
        self._scene_words_translated_xlsx_format = '{product_id}_{scene_id}_translated.xlsx'
        self._backward_sql_format = '{product_id}_{scene_id}_backend.sql'
        self._main_frontward_sql_format = '{product_id}_{scene_id}_main_frontend.sql'
        self._business_sql_format = '{product_id}_{scene_id}_business.sql'

    def get_src_language(self, product_id, scene_id, mysql_connect):
        sql = """
SELECT language 
FROM t_dialog_language_info 
WHERE product_id='{}' AND scene_id='{}';
""".format(product_id, scene_id)

        rows = mysql_connect.execute(
            sql=sql
        )
        src_language = rows[0][0]

        if src_language not in self.aibot_language2google_language:
            raise AssertionError('invalid source language: {}'.format(src_language))
        else:
            src_language = self.aibot_language2google_language[src_language]

        return src_language

    def collect_scene_words(self, product_id, scene_id, src_language: str, dest_language, to_filename: str):
        words = list()

        # 后端表数据 t_dialog_resource_info
        sql = """
SELECT product_id, scene_id, group_id, res_id, word 
FROM t_dialog_resource_info 
WHERE product_id='{}' AND scene_id='{}';
""".format(product_id, scene_id)
        rows = self.mysql_connect.execute(
            sql=sql
        )
        for row in tqdm(rows):
            product_id = row[0]
            scene_id = row[1]
            group_id = row[2]
            res_id = row[3]
            word = row[4].strip()

            if len(group_id) == 0:
                # group_id 为空的, 不管
                continue
            if len(word) == 0:
                continue
            words.append({
                'text': word,
                'src_language': src_language,
                'dest_language': dest_language,
            })

        # 前端表数据 t_cb_process_master
        sql = """
SELECT product_id, scene_id, process_graph
FROM t_cb_process_master 
WHERE product_id='{}' AND scene_id='{}';
""".format(product_id, scene_id)
        rows = self.mysql_connect.execute(
            sql=sql
        )
        process_graph = rows[0][2]
        process_graph = json.loads(process_graph)

        nodes = process_graph['nodes']
        connections = process_graph['connections']

        for node in tqdm(nodes):
            desc = node['desc']
            desc_again = node['descAgain']
            if len(desc.strip()) != 0:
                words.append({
                    'text': desc,
                    'src_language': src_language,
                    'dest_language': dest_language,
                })
            if len(desc_again.strip()) != 0:
                words.append({
                    'text': desc_again,
                    'src_language': src_language,
                    'dest_language': dest_language,
                })

        for connection in connections:
            keywords = connection['keywords']['keyWords']
            keywords = keywords.split('##')
            for keyword in tqdm(keywords):
                if len(keyword.strip()) != 0:
                    words.append({
                        'text': keyword,
                        'src_language': src_language,
                        'dest_language': dest_language,
                    })

        # 业务问答 t_cb_question
        sql = """
SELECT product_id, question_id, scene_id, mast_id, node_desc, bot_words, lead_words, key_words, node_id, intent_res_group_id, action_res_group_id, lead_res_group_id 
FROM t_cb_question 
WHERE product_id='{}' AND scene_id='{}';
""".format(product_id, scene_id)
        rows = self.mysql_connect.execute(
            sql=sql
        )
        for row in tqdm(rows):
            node_desc = row[4]
            bot_words = row[5]
            lead_words = row[6]
            key_words = row[7]
            key_words = json.loads(key_words)
            keywords = key_words['keyWords']
            keywords = keywords.split('##')
            for keyword in keywords:
                if len(keyword.strip()) != 0:
                    words.append({
                        'text': keyword,
                        'src_language': src_language,
                        'dest_language': dest_language,
                    })

        words = pd.DataFrame(words)
        words = words.drop_duplicates(subset=['text'])
        words.to_excel(to_filename, index=False, encoding='utf_8_sig')
        return to_filename

    def translate_scene_words(self, filename, to_filename_json, to_filename_xlsx):
        df = pd.read_excel(filename)

        with open(to_filename_json, 'a+', encoding='utf-8') as f:
            text_list = list()
            char_count = 0
            total = len(df)
            for idx, row in df.iterrows():
                text = row['text'].strip()
                src_language = row['src_language']
                dest_language = row['dest_language']

                text_list.append(text)
                char_count += len(text) + 5

                # 每次调用的句子长度, 随机.
                char_limit = 1000 + 2000 * random.random()
                if char_count > char_limit:
                    # 每次调用的间隔, 随机.
                    delay = 2 + 18 * random.random()
                    print('sleep: {}, char_count: {}, idx: {}, total: {}'.format(delay, char_count, idx, total))
                    time.sleep(delay)

                    translated_text_list: List[str] = GoogleTranslate.batch_translate(
                        text_list=text_list,
                        dest=dest_language,
                        src=src_language,
                    )
                    print(text_list)
                    print(translated_text_list)
                    for translated_text, text in zip(translated_text_list, text_list):
                        # print(text)
                        row = {
                            'src_text': text,
                            'dest_text': translated_text,
                            'src_language': src_language,
                            'dest_language': dest_language,
                        }
                        row = json.dumps(row, ensure_ascii=False)
                        # print(row)
                        f.write('{}\n'.format(row))
                    text_list = list()
                    char_count = 0

        result = list()
        with open(to_filename_json, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                result.append(line)
        result = pd.DataFrame(result)
        result.to_excel(to_filename_xlsx, index=False, encoding='utf_8_sig')
        return

    def make_backend_sql(self, filename, product_id, scene_id, src_language, dest_language, to_filename_sql):
        df = pd.read_excel(filename)
        src_text = df['src_text'].tolist()
        dest_text = df['dest_text'].tolist()
        src_text2dest_text = dict(zip(src_text, dest_text))

        # 更新 t_dialog_resource_info
        sql = """
SELECT product_id, scene_id, group_id, res_id, word 
FROM t_dialog_resource_info 
WHERE product_id='{}' AND scene_id='{}';
""".format(product_id, scene_id)
        rows = self.mysql_connect.execute(
            sql=sql
        )

        with open(to_filename_sql, 'w', encoding='utf-8') as f:
            for row in tqdm(rows):
                product_id = row[0]
                scene_id = row[1]
                group_id = row[2]
                res_id = row[3]
                word = row[4]

                if len(group_id) == 0:
                    # group_id 为空的, 不管
                    continue

                if word in src_text2dest_text:
                    word = src_text2dest_text[word]
                else:
                    word = self.google_translate(
                        text=word,
                        dest=dest_language,
                        src=src_language,
                    )

                sql = """UPDATE t_dialog_resource_info SET word='{}' WHERE product_id='{}' AND scene_id='{}' AND group_id='{}' AND res_id='{}';""".format(
                    escape_string(word), product_id, scene_id, group_id, res_id)
                # print('sql: {}'.format(sql))

                f.write('{}\n'.format(sql))
        return

    def make_main_frontend_sql(self, filename, product_id, scene_id, src_language, dest_language, to_filename_sql):
        df = pd.read_excel(filename)
        src_text = df['src_text'].tolist()
        dest_text = df['dest_text'].tolist()
        src_text2dest_text = dict(zip(src_text, dest_text))
        # print(src_text2dest_text)

        sql = """
    SELECT product_id, scene_id, process_graph
    FROM t_cb_process_master 
    WHERE product_id='{}' AND scene_id='{}';
    """.format(product_id, scene_id)
        rows = self.mysql_connect.execute(
            sql=sql
        )
        process_graph = rows[0][2]
        process_graph = json.loads(process_graph)

        nodes = process_graph['nodes']
        connections = process_graph['connections']

        result = defaultdict(list)
        print('translate the nodes')
        for node in tqdm(nodes):
            desc = node['desc'].strip()
            desc_again = node['descAgain'].strip()

            key = desc.replace('\'', '')
            if key in src_text2dest_text:
                desc = src_text2dest_text[key]
            else:
                desc = self.google_translate(
                    text=desc,
                    dest=dest_language,
                    src=src_language,
                )

            key = desc_again.replace('\'', '')
            if key in src_text2dest_text:
                desc_again = src_text2dest_text[key]
            else:
                desc_again = self.google_translate(
                    text=desc_again,
                    dest=dest_language,
                    src=src_language,
                )

            node['desc'] = desc
            node['descAgain'] = desc_again
            result['nodes'].append(node)

        print('translate the connections')
        for connection in connections:
            keywords = connection['keywords']['keyWords']
            keywords = keywords.split('##')
            translated_keywords = list()
            for keyword in tqdm(keywords):
                key = keyword.strip()
                # key = key.replace('\'', '')
                # key = key.replace('「', '')
                # key = key.replace('」', '')
                # key = key.replace('\"', '')

                if key in src_text2dest_text:
                    keyword = src_text2dest_text[key]
                else:
                    keyword = self.google_translate(
                        text=keyword,
                        dest=dest_language,
                        src=src_language,
                    )
                translated_keywords.append(keyword)

            translated_keywords = '##'.join(translated_keywords)
            connection['keywords']['keyWords'] = translated_keywords
            result['connections'].append(connection)

        process_graph = json.dumps(result, ensure_ascii=False)
        process_graph = escape_string(process_graph)

        sql = """
UPDATE t_cb_process_master SET process_graph='{}' WHERE product_id='{}' AND scene_id='{}';
""".format(process_graph, product_id, scene_id)

        with open(to_filename_sql, 'w', encoding='utf-8') as f:
            f.write('{}\n'.format(sql))
        return

    def make_business_sql(self, filename, product_id, scene_id, src_language, dest_language, to_filename_sql):
        df = pd.read_excel(filename)
        src_text = df['src_text'].tolist()
        dest_text = df['dest_text'].tolist()
        src_text2dest_text = dict(zip(src_text, dest_text))

        sql = """
SELECT product_id, question_id, scene_id, mast_id, node_desc, bot_words, lead_words, key_words, node_id, intent_res_group_id, action_res_group_id, lead_res_group_id 
FROM t_cb_question 
WHERE product_id='{}' AND scene_id='{}';
""".format(product_id, scene_id)
        rows = self.mysql_connect.execute(
            sql=sql
        )
        with open(to_filename_sql, 'w', encoding='utf-8') as f:
            for row in rows:
                product_id = row[0]
                question_id = row[1]
                scene_id = row[2]
                mast_id = row[3]
                node_desc = row[4]
                bot_words = row[5]
                lead_words = row[6]
                key_words = row[7]
                if len(key_words) == 0:
                    continue
                key_words = json.loads(key_words)
                keywords = key_words['keyWords']
                keywords = keywords.split('##')

                translated_keywords = list()
                for keyword in keywords:
                    print(keyword)
                    if keyword in src_text2dest_text:
                        translated_keyword = src_text2dest_text[keyword]
                    else:
                        translated_keyword = self.google_translate(
                            text=keyword,
                            dest=dest_language,
                            src=src_language,
                        )
                    translated_keywords.append(translated_keyword)
                translated_keywords = '##'.join(translated_keywords)
                translated_keywords = escape_string(translated_keywords)
                key_words['keyWords'] = translated_keywords
                key_words = json.dumps(key_words, ensure_ascii=False)

                sql = """UPDATE t_cb_question SET key_words='{}' WHERE product_id='{}' AND question_id='{}' AND scene_id='{}' AND mast_id='{}';""".format(
                    escape_string(key_words), product_id, question_id, scene_id, mast_id)
                f.write('{}\n'.format(sql))

                if bot_words in src_text2dest_text:
                    translated_bot_words = src_text2dest_text[bot_words]
                else:
                    translated_bot_words = self.google_translate(
                        text=bot_words,
                        dest=dest_language,
                        src=src_language,
                    )
                sql = """UPDATE t_cb_question SET bot_words='{}' WHERE product_id='{}' AND question_id='{}' AND scene_id='{}' AND mast_id='{}';""".format(
                    escape_string(translated_bot_words), product_id, question_id, scene_id, mast_id)
                f.write('{}\n'.format(sql))

                if lead_words in src_text2dest_text:
                    translated_lead_words = src_text2dest_text[lead_words]
                else:
                    translated_lead_words = self.google_translate(
                        text=lead_words,
                        dest=dest_language,
                        src=src_language,
                    )
                sql = """UPDATE t_cb_question SET lead_words='{}' WHERE product_id='{}' AND question_id='{}' AND scene_id='{}' AND mast_id='{}';""".format(
                    escape_string(translated_lead_words), product_id, question_id, scene_id, mast_id)
                f.write('{}\n'.format(sql))
        return

    def translate(self, product_id, scene_id, dest_language: str):
        src_language = self.get_src_language(
            product_id=product_id,
            scene_id=scene_id,
            mysql_connect=self.mysql_connect,
        )
        scene_words_filename = self._scene_words_format.format(product_id=product_id, scene_id=scene_id)
        scene_words_filename = os.path.join(self.root_path, scene_words_filename)
        # 收集需要翻译的句子
        scene_words_filename = self.collect_scene_words(
            product_id=product_id,
            scene_id=scene_id,
            src_language=src_language,
            dest_language=dest_language,
            to_filename=scene_words_filename,
        )

        translated_scene_words_json_filename = self._scene_words_translated_json_format.format(product_id=product_id, scene_id=scene_id)
        translated_scene_words_json_filename = os.path.join(self.root_path, translated_scene_words_json_filename)
        translated_scene_words_xlsx_filename = self._scene_words_translated_xlsx_format.format(product_id=product_id, scene_id=scene_id)
        translated_scene_words_xlsx_filename = os.path.join(self.root_path, translated_scene_words_xlsx_filename)
        # 翻译句子
        self.translate_scene_words(
            filename=scene_words_filename,
            to_filename_json=translated_scene_words_json_filename,
            to_filename_xlsx=translated_scene_words_xlsx_filename,
        )

        backend_sql_filename = self._backward_sql_format.format(product_id=product_id, scene_id=scene_id)
        # 创建后端 sql 文件
        self.make_backend_sql(
            filename=translated_scene_words_xlsx_filename,
            product_id=product_id,
            scene_id=scene_id,
            src_language=src_language,
            dest_language=dest_language,
            to_filename_sql=backend_sql_filename
        )

        main_frontend_sql_filename = self._main_frontward_sql_format.format(product_id=product_id, scene_id=scene_id)
        # 创建前端主流程 sql 文件
        self.make_main_frontend_sql(
            filename=translated_scene_words_xlsx_filename,
            product_id=product_id,
            scene_id=scene_id,
            src_language=src_language,
            dest_language=dest_language,
            to_filename_sql=main_frontend_sql_filename
        )

        business_sql_filename = self._business_sql_format.format(product_id=product_id, scene_id=scene_id)
        # 创建业务问答 sql 文件
        self.make_business_sql(
            filename=translated_scene_words_xlsx_filename,
            product_id=product_id,
            scene_id=scene_id,
            src_language=src_language,
            dest_language=dest_language,
            to_filename_sql=business_sql_filename
        )
        return


def demo1():
    mysql_connect = MySqlConnect(
        host='10.20.251.13',
        port=3306,
        user='callbot',
        password='NxcloudAI2021!',
        database='callbot_ppe',
        charset='utf8',
    )
    translate = GoogleTranslateScene(
        mysql_connect=mysql_connect,
        root_path=os.path.abspath(os.path.dirname(__file__))
    )

    translate.translate(
        product_id='callbot',
        scene_id='lupmormyps',
        dest_language='my',
    )
    return


if __name__ == '__main__':
    demo1()

import csv
import re
import numpy as np

small_train = "multi-to-uni/datasets/MyMINDsmall/train/news.tsv"
small_dev = "MINDsmall/MINDsmall_dev/news.tsv"
large_train = "MINDlarge/MINDlarge_train/news.tsv"
large_dev = "MINDlarge/MINDlarge_dev/news.tsv"

# MINDsmall_train 수정
title_to_id = {}
with open(large_train, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if len(row) > 3:
            title = row[3]
            news_id = row[0]
            title_to_id[title] = news_id

# 2. smallnews.tsv 읽고 id 바꿔서 저장
updated_rows = []
with open(small_train, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if len(row) > 3:
            title = row[3]
            if title in title_to_id:
                row[0] = title_to_id[title]  # id 교체
        updated_rows.append(row)

# 3. 결과를 새로운 파일로 저장
with open('small_train_updated.tsv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(updated_rows)

print("수정 완료: small_train_updated.tsv 파일 생성됨.")


# MINDsmall_dev 수정
title_to_id = {}
with open(large_dev, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if len(row) > 3:
            title = row[3]
            news_id = row[0]
            title_to_id[title] = news_id

# 2. smallnews.tsv 읽고 id 바꿔서 저장
updated_rows = []
with open(small_dev, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if len(row) > 3:
            title = row[3]
            if title in title_to_id:
                row[0] = title_to_id[title]  # id 교체
        updated_rows.append(row)

# 3. 결과를 새로운 파일로 저장
with open('small_dev_updated.tsv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(updated_rows)

print("수정 완료: small_dev_updated.tsv 파일 생성됨.")


# MINDsmall_behavior 수정
title_to_large_id = {}
with open(large_train, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if len(row) > 3:
            title_to_large_id[row[3]] = row[0]

# 2. small_id → large_id 매핑
small_to_large_id = {}
with open(small_train, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if len(row) > 3:
            small_id = row[0]
            title = row[3]
            large_id = title_to_large_id.get(title)
            if large_id and large_id != small_id:
                small_to_large_id[small_id] = large_id

# behaviors.tsv 수정
updated_rows = []
id_change_log = set()

with open('MINDsmall/MINDsmall_train/behaviors.tsv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if len(row) < 5:
            updated_rows.append(row)
            continue

        # ✅ 4번째 열 (index 3): 뉴스 ID 리스트
        news_ids = row[3].strip().split()
        new_news_ids = []
        for nid in news_ids:
            new_id = small_to_large_id.get(nid, nid)
            if new_id != nid:
                id_change_log.add((nid, new_id))
            new_news_ids.append(new_id)
        row[3] = ' '.join(new_news_ids)

        # ✅ 5번째 열 (index 4): impressions (뉴스ID-클릭여부)
        impressions = row[4].strip().split()
        new_impressions = []
        for impression in impressions:
            if '-' not in impression:
                new_impressions.append(impression)
                continue
            news_id, label = impression.rsplit('-', 1)
            new_id = small_to_large_id.get(news_id, news_id)
            if new_id != news_id:
                id_change_log.add((news_id, new_id))
            new_impressions.append(f"{new_id}-{label}")
        row[4] = ' '.join(new_impressions)

        updated_rows.append(row)

# 4. 저장
with open('mindsmall_train_behaviors.tsv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(updated_rows)

print("✅ 저장 완료: train_behaviors.tsv")


# dev behaviors 수정
updated_rows = []
id_change_log = set()

with open('MINDsmall/MINDsmall_dev/behaviors.tsv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if len(row) < 5:
            updated_rows.append(row)
            continue

        # ✅ 4번째 열 (index 3): 뉴스 ID 리스트
        news_ids = row[3].strip().split()
        new_news_ids = []
        for nid in news_ids:
            new_id = small_to_large_id.get(nid, nid)
            if new_id != nid:
                id_change_log.add((nid, new_id))
            new_news_ids.append(new_id)
        row[3] = ' '.join(new_news_ids)

        # ✅ 5번째 열 (index 4): impressions (뉴스ID-클릭여부)
        impressions = row[4].strip().split()
        new_impressions = []
        for impression in impressions:
            if '-' not in impression:
                new_impressions.append(impression)
                continue
            news_id, label = impression.rsplit('-', 1)
            new_id = small_to_large_id.get(news_id, news_id)
            if new_id != news_id:
                id_change_log.add((news_id, new_id))
            new_impressions.append(f"{new_id}-{label}")
        row[4] = ' '.join(new_impressions)

        updated_rows.append(row)

# 4. 저장
with open('mindsmall_dev_behaviors.tsv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(updated_rows)

print("✅ 저장 완료: dev_behaviors.tsv")
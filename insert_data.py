import json
import psycopg2

# PostgreSQLデータベースに接続
conn = psycopg2.connect(
    dbname='dan6pok9mj1oqi',
    user='udntpff7mutma6',
    password='pd4ed0d9ae39fa51034e8fbda0d127dfc26dce88dba163b6994d49905710d1bfd',
    host='ec2-34-194-230-91.compute-1.amazonaws.com',
    port='5432'
)
cursor = conn.cursor()

# a. plantnet300K_species_id_2_name.jsonを読み込む
with open('/Users/mame/plantnetvgg16_2/plantnetvgg16_2/data/plantnet300K_species_id_2_name.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# データをplantsテーブルに挿入（重複データをスキップ）
for species_id, name in data.items():
    cursor.execute("""
        INSERT INTO plants (species_id, name) 
        VALUES (%s, %s) 
        ON CONFLICT (species_id) DO NOTHING
    """, (species_id, name))

conn.commit()

# b. class_indices.jsonを読み込む
with open('/Users/mame/plantnetvgg16_2/plantnetvgg16_2/data/class_indices.json', 'r', encoding='utf-8') as file:
    class_data = json.load(file)

# データをclass_indicesテーブルに挿入
for class_id, class_name in class_data.items():
    cursor.execute("INSERT INTO class_indices (class_id, class_name) VALUES (%s, %s)", (class_id, class_name))

conn.commit()





# 接続を閉じる
cursor.close()
conn.close()

import numpy as np
import pandas as pd
import psycopg2
import os
import random
from tqdm import tqdm, trange
import docker
import pickle
from data_preparation.physical_db import DBConnection,TrueCardinalityEstimator
from sqlalchemy import create_engine
import time
from utils import generate_all_single_table_queries

conn = psycopg2.connect(database="e2e_autoce", host="localhost", port='30003', user='jintao', password='jintao2020')
cursor = conn.cursor()

client = docker.from_env()
container = client.containers.get('ce-benchmark')

# os.chdir('./pg13_data')
# os.system('docker cp ce-benchmark:/var/lib/pgsql/13.1/data/single_tbl_est_record.txt.txt ./pg13_data/')
# os.system(f'mv ./pg13_data/join_est_record_job.txt ./pg13_data/single_tbl_est_record.txt{i}.txt')

for i in range(15):
    imdb_sql_file = open(f'./data_preparation/benchmark/{i}_multitest.sql')
    queries = imdb_sql_file.readlines()
    imdb_sql_file.close()

    container.exec_run("rm /var/lib/pgsql/13.1/data/join_est_record_job.txt")
    # os.system('docker cp ./pg13_data/single_tbl_est_record.txt ce-benchmark:/var/lib/pgsql/13.1/data/')
    if os.path.exists(f"./pg13_data/join_est_record_job{i}.txt"):
        os.remove(f"./pg13_data/join_est_record_job{i}.txt")

    # cursor.execute('SET debug_card_est=true')
    cursor.execute('SET print_sub_queries=true')
    # cursor.execute('SET print_single_tbl_queries=true')

    for no, query in enumerate(queries):
        cursor.execute("EXPLAIN (FORMAT JSON)" + query.split(";,")[0] + ';')
        res = cursor.fetchall()
        cursor.execute("SET query_no=0")
        # print("%d-th query finished." % no)

    os.system('docker cp ce-benchmark:/var/lib/pgsql/13.1/data/join_est_record_job.txt ./pg13_data/')
    os.system(f'mv ./pg13_data/join_est_record_job.txt ./pg13_data/join_est_record_job{i}.txt')
    generate_all_single_table_queries(f'join_est_record_job{i}.txt', f'./data_preparation/benchmark/{i}_multitest.sql', f'sub_queries{i}.txt')  # sub_queries

cursor.close()
conn.close()

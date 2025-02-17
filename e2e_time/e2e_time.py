import numpy as np
import pandas as pd
import psycopg2
import os
import random
from tqdm import tqdm
import docker
import pickle
from data_preparation.physical_db import DBConnection,TrueCardinalityEstimator
from sqlalchemy import create_engine
import time
from gen_sub_queries_multi import generate_all_single_table_queries

# E2E time
# Interation
def interation(method,dataset_num1,dataset_num2,cursor):
    for i in range(dataset_num1,dataset_num2):
        os.system(f'chmod 777 ./pg13_data/{method}_multi{i}.txt')
        os.system(f'docker cp ./pg13_data/{method}_multi{i}.txt ce-benchmark:/var/lib/pgsql/13.1/data/{method}.txt')
        cursor.execute('SET ml_joinest_enabled=true;')
        cursor.execute('SET join_est_no=0;')
        cursor.execute(f"SET ml_joinest_fname='{method}.txt';")

        f=open(f"./data_preparation/benchmark/{i}_multitest.sql")
        queries = [line.split(";,")[0]+';' for line in f.readlines()]
        queries = [query+'\n' if query[-1]!='\n' else query for query in queries]

        for q in queries[0:-1]:
             cursor.execute(q)


# PG run time
conn = psycopg2.connect(database="autoce", host="localhost", port='30003', user='jintao', password=' ')
cursor = conn.cursor()

time_stpg=time.time()
interation('pg',0,15,cursor)
print('PG run time:', time.time()-time_stpg)

cursor.close()
conn.close()

import os
import sqlite3
import pickle


def get_conn(model_dir):
    try:
        conn = sqlite3.connect(model_dir)
    except sqlite3.OperationalError as e:
        print(e)
    return conn


def init_nasbench_macro_cifar10(model_dir):
    try:
        conn = sqlite3.connect(os.path.join(model_dir, 'models.db'))
        conn.execute("create table models (id text not null, hashkey text, modelpath text, train_acc real, val_acc real, "
                     "test_acc real)")
        conn.commit()
    except sqlite3.OperationalError as e:
        print(e)
    return conn



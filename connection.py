import pandas as pd
from datetime import datetime
import logging
import os
import dataset

def get_database():
    try:
        DATABASE_URL = os.environ["DATABASE_URL"]
    except KeyError:
        home = os.environ["HOME"]
        DATABASE_URL = f"sqlite:///{home}/ml_physics.db"
    return dataset.connect(DATABASE_URL)

DATABASE = get_database()

def get_table(table_name):
    if table_name not in DATABASE.tables:
        logging.error(f"{table_name} does not exist")
        return None
    table = DATABASE[table_name]
    records = [record for record in table.all()]
    return pd.DataFrame(records)


class TableResultHandler():
    def __init__(self, table_name, overwrite=False):
        table = DATABASE[table_name]
        if overwrite:
            logging.warning(f"Dropping table {table_name}")
            table.drop()
        else:
            logging.warning(f"Appending to table {table_name}")
        self.table = table

    def add_record(self, record):
        record["created_on"] = datetime.now()
        self.table.insert(record)

    def get_dataframe(self):
        records = [record for record in self.table.all()]
        return pd.DataFrame(records)

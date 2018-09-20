import pandas as pd
from datetime import datetime
import logging
import os
import dataset

def get_default_database_url():
    try:
        return os.environ["DATABASE_URL"]
    except KeyError:
        home = os.environ["HOME"]
        return f"sqlite:///{home}/ml_physics.db"

DEFAULT_DATABASE_URL = get_default_database_url()

def get_database(database_url = None):
    database_url = database_url or DEFAULT_DATABASE_URL
    return dataset.connect(database_url)

def get_table(table_name, database_url = None):
    database = get_database(database_url)
    if table_name not in database.tables:
        logging.error(f"{table_name} does not exist")
        return None
    table = database[table_name]
    records = [record for record in table.all()]
    return pd.DataFrame(records)


class TableResultHandler():
    def __init__(self, table_name, overwrite=False, database_url = None):
        database = get_database(database_url)
        table = database[table_name]
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

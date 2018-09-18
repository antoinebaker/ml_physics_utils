import pandas as pd
import logging


def gather(df, melted_columns, value_name="value", var_name="variable"):
    """Gather melted_columns."""
    id_vars = [column for column in df.columns if column not in melted_columns]
    melted = df.melt(id_vars=id_vars, value_name=value_name, var_name=var_name)
    return melted


def gather_match(df, value_name, var_name):
    """Gather all columns matching {value_name}_{var} pattern.
    The {var} value is stoded into {var_name} column.
    - Input dataframe: {value_name}_{var} = {value}
    - the {value_name}_{var} columns are melted
    - Output dataframe: {value_name} = {value} and {var_name} = {var}.
    """
    melted_columns = df.columns[df.columns.str.match(value_name + "_")]
    melted = gather(df, melted_columns, value_name, var_name)
    # sanity check : var_name values should all belong to melted_columns
    assert melted[var_name].isin(melted_columns).all()
    # sanity check : they should all match {value_name}_{var}
    assert melted[var_name].str.match(value_name + "_").all()
    # extract the {var} part
    melted[var_name] = melted[var_name].str.extract(
        value_name + "_(\w+)", expand=False
    )
    return melted


def count_records(df, keys):
    by_keys = df.groupby(keys).size().rename("count").reset_index()
    return by_keys


def check_for_duplicates(df, subset, keys=None):
    duplicates = df.duplicated(subset=subset, keep=False)
    if duplicates.any():
        logging.warning(f"{duplicates.sum()} duplicates")
        if keys:
            by_keys = count_records(df[duplicates], keys)
            print(by_keys)
    else:
        logging.info(f"No duplicates")

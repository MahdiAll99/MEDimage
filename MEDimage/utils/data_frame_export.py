import os.path
from isort import file
import pandas as pd

def export_table(file_name: file,
                 data: object):
    """Export table

    Args:
        file_name (file): name of the file
        data (object): the data
    
    Returns:
        None
    """

    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError(f"The exported data should be a pandas DataFrame or Series. Found: {type(data)}")

    # Find the extension
    ext = os.path.splitext(file_name)[1]

    # Set an index switch based on type of input
    if isinstance(data, pd.DataFrame):
        write_index = False
    else:
        write_index = True

    if ext == ".csv":
        data.to_csv(path_or_buf=file_name, sep=";", index=write_index)
    elif ext in [".xls", ".xlsx"]:
        data.to_excel(excel_writer=file_name, index=write_index)
    elif ext in [".tex"]:
        with open(file=file_name, mode="w") as f:
            data.to_latex(buf=f, index=write_index)
    elif ext in [".html"]:
        with open(file=file_name, mode="w") as f:
            data.to_html(buf=f, index=write_index)
    elif ext in [".json"]:
        data.to_json(path_or_buf=file_name)
    else:
        raise ValueError(f"File extension not supported for export of table data. Recognised extensions are: \".csv\", \".xls\", \".xlsx\", \".tex\", \".html\" and \".json\". Found: {ext}")

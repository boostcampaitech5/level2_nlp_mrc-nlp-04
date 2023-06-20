import streamlit as st
import pandas as pd
import os
import zipfile
from pathlib import Path
import json

st.markdown("# 📌 Share point")

with open('./conf.json') as json_file:
    json_config = json.load(json_file)

def zip_folder(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                if "cache" in file_path:
                    continue
                print(file_path)
                zipf.write(file_path, os.path.relpath(file_path, folder_path))

@st.cache_resource()
def zip_files():
    dict_files = []
    list_col = ["reader", "retrieval", "dataset"]

    for col in list_col:
        for i in json_config[col]:

            # 압축할 폴더 경로
            folder_path = i["path"]

            # 압축 파일 경로
            zip_path = "./zip/" + col + "_" + i["name"] + ".zip"
            # dict_files[col + "-" + i["name"]] = zip_path
            dict_files.append(zip_path)

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        print(file_path)
                        zipf.write(file_path, os.path.relpath(file_path, folder_path))

    return dict_files

dict_files = zip_files()

def get_config_df(parts, json_config):
    list_name = []
    list_desc = []
    list_version = []
    list_path = []

    for part in json_config[parts]:
        list_name.append(part["name"])
        list_desc.append(part["description"])
        list_version.append(part["version"])
        list_path.append(part["path"])

    return pd.DataFrame({"name": list_name, "description": list_desc, "version": list_version, "path": list_path})

st.markdown("## Reader")
df_reader = get_config_df("reader", json_config)

st.dataframe(
    df_reader
)

st.markdown("## Retrieval")
df_retrieval = get_config_df("retrieval", json_config)

st.dataframe(
    df_retrieval
)

st.markdown("## Dataset")
df_dataset= get_config_df("dataset", json_config)

st.dataframe(
    df_dataset
)

path_file = st.radio(
    "파일을 선택하세요. ",
    dict_files)

with open(path_file, "rb") as file:
    btn = st.download_button(
        label="Download File",
        data=file,
        file_name="file.zip",
        mime="application/zip"
    )
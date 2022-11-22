import json
import logging
import os
from typing import List

import requests
from .save import (
    _download_and_extract_data_to,
    get_linked_output_dir,
)
from .utils import get_model_data, LIGHTNING_CLOUD_URL, LIGHTNING_STORAGE_FILE, split_name, stage
from .utils import LIGHTNING_STORAGE_DIR


logging.basicConfig(level=logging.INFO)






def download_from_lightning_cloud(
    name: str,
    version: str = "latest",
    output_dir: str = "",
    progress_bar: bool = True,
):
    """
    Parameters
    ==========
    :param: name (str):
        The unique name of the model to be downloaded. Format: `<username>/<model_name>`.
    :param: version: (str, default="latest")
        The version of the model to be uploaded. If not provided, default will be latest (not overridden).
    :param: output_dir (str, default=""):
        The target directory, where the model and other data will be stored. If not passed,
            the data will be stored in `$HOME/.lightning/lightning_model_store/<username>/<model_name>/<version>`.
            (`version` defaults to `latest`)

    Returns
    =======
    None
    """
    version = version or "latest"
    username, model_name, version = split_name(name, version=version, l_stage=stage.DOWNLOAD)

    linked_output_dir = ""
    if not output_dir:
        output_dir = LIGHTNING_STORAGE_DIR
        output_dir = os.path.join(output_dir, username, model_name, version)
        linked_output_dir = get_linked_output_dir(output_dir)
    else:
        output_dir = os.path.abspath(output_dir)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    response = requests.get(f"{LIGHTNING_CLOUD_URL}/v1/models?name={username}/{model_name}&version={version}")
    assert response.status_code == 200, (
        f"Unable to download the model with name {name} and version {version}."
        " Maybe reach out to the model owner or check the arguments again?"
    )

    download_url_response = json.loads(response.content)
    download_url = download_url_response["downloadUrl"]
    meta_data = download_url_response["metadata"]

    logging.info(f"Downloading the model data for {name} to {output_dir} folder.")
    _download_and_extract_data_to(output_dir, download_url, progress_bar)

    if linked_output_dir:
        logging.info(f"Linking the downloaded folder from {output_dir} to {linked_output_dir} folder.")
        if os.path.islink(linked_output_dir):
            os.unlink(linked_output_dir)
        if os.path.exists(linked_output_dir):
            if os.path.isdir(linked_output_dir):
                os.rmdir(linked_output_dir)

        os.symlink(output_dir, linked_output_dir)

    if not os.path.isfile(LIGHTNING_STORAGE_FILE):
        with open(LIGHTNING_STORAGE_FILE, "w+") as storage_file:
            data = {}
            json.dump(data, storage_file)
    else:
        with open(LIGHTNING_STORAGE_FILE) as storage_file:
            data = json.load(storage_file)

    with open(LIGHTNING_STORAGE_FILE, "w+") as storage_file:
        storage = {
            username: {
                model_name: {
                    version: {
                        "output_dir": output_dir,
                        "linked_output_dir": str(linked_output_dir),
                        "metadata": meta_data,
                    },
                },
            },
        }
        if username in data:
            if model_name in data[username]:
                data[username][model_name][version] = storage[username][model_name][version]
            else:
                data[username][model_name] = storage[username][model_name]
        else:
            data[username] = storage[username]

        json.dump(data, storage_file, indent=4)

    logging.info("Downloading done...")
    logging.info(
        f"The source code for your model has been written to {output_dir} folder, and"
        f" linked to {linked_output_dir} folder."
    )

    logging.info(
        "Please make sure to add imports to the necessary classes needed for instantiation of"
        " your model before calling `load_from_lightning_cloud`."
    )


def _validate_output_dir(dir: str):
    if not os.path.exists(dir):
        raise ValueError(
            "The output directory doesn't exist... did you forget to call download_from_lightning_cloud(...)?"
        )


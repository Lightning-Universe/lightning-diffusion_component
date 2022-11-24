import logging
import os
import shutil
import tarfile
from pathlib import PurePath

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def _download_and_extract_data_to(
    output_dir: str, download_url: str, progress_bar: bool
):
    def _common_clean_up():
        data_file_path = f"{output_dir}/data.tar.gz"
        dir_file_path = f"{output_dir}/extracted"
        if os.path.exists(data_file_path):
            os.remove(data_file_path)
        shutil.rmtree(dir_file_path)

    try:
        with requests.get(download_url, stream=True) as req_stream:
            total_size_in_bytes = int(req_stream.headers.get("content-length", 0))
            block_size = 1024  # 1 Kibibyte

            download_progress_bar = None
            if progress_bar:
                download_progress_bar = tqdm(
                    total=total_size_in_bytes, unit="iB", unit_scale=True
                )
            with open(f"{output_dir}/data.tar.gz", "wb") as f:
                for chunk in req_stream.iter_content(chunk_size=block_size):
                    if download_progress_bar:
                        download_progress_bar.update(len(chunk))
                    f.write(chunk)
            if download_progress_bar:
                download_progress_bar.close()

        tar = tarfile.open(f"{output_dir}/data.tar.gz", "r:gz")
        tmpdir_name = tar.getnames()[0]
        tar.extractall(path=f"{output_dir}/extracted")
        tar.close()

        root = f"{output_dir}"
        for filename in os.listdir(os.path.join(root, "extracted", tmpdir_name)):
            abs_file_name = os.path.join(root, "extracted", tmpdir_name, filename)
            if os.path.isdir(abs_file_name):
                func = shutil.copytree
            else:
                func = shutil.copy

            dst_file_name = os.path.join(root, filename)
            if os.path.exists(dst_file_name):
                if os.path.isdir(dst_file_name):
                    shutil.rmtree(dst_file_name)
                else:
                    os.remove(dst_file_name)

            func(
                abs_file_name,
                os.path.join(root, filename),
            )

        assert os.path.isdir(f"{output_dir}"), (
            "Data downloading to the output"
            f" directory: {output_dir} failed. Maybe try again or contact the model owner?"
        )
    except Exception as e:
        _common_clean_up()
        raise e
    else:
        _common_clean_up()


def get_linked_output_dir(src_dir: str):
    # The last sub-folder will be our version
    version_folder_name = PurePath(src_dir).parts[-1]

    if version_folder_name == "latest":
        return str(PurePath(src_dir).parent.joinpath("version_latest"))
    else:
        replaced_ver = version_folder_name.replace(".", "_")
        return str(PurePath(src_dir).parent.joinpath(f"version_{replaced_ver}"))

import hashlib
import tarfile
import zipfile


def get_md5_checksum(file_path):
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
    return md5_hash.hexdigest()


def extract_files_from_archive(archive_file_path):
    if archive_file_path.endswith(".tar.gz"):
        with tarfile.open(archive_file_path, "r:gz") as tar:
            return tar.extractall()
    elif archive_file_path.endswith(".zip"):
        with zipfile.ZipFile(archive_file_path, "r") as zip_ref:
            return zip_ref.extractall()

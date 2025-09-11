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


def download_gcs_file(gcs_uri, destination_file_name):
    """Downloads a file from Google Cloud Storage."""
    try:
        from google.cloud import storage
    except ImportError:
        print(
            "The `google-cloud-storage` package is not installed. "
            "Please install it with `pip install google-cloud-storage` "
            "to download files from Google Cloud Storage."
        )

    storage_client = storage.Client.create_anonymous_client()
    bucket_name = gcs_uri.split("/")[2]
    source_blob_name = "/".join(gcs_uri.split("/")[3:])

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    # Download the file
    blob.download_to_filename(destination_file_name)

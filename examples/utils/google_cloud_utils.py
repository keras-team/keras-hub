# Copyright 2022 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from google.cloud import storage


def list_blobs_with_prefix(bucket_name, prefix, delimiter=None):
    """Lists all the blobs path in the bucket that begin with the prefix.

    This can be used to list all blobs in a "folder", e.g. "public/".

    The delimiter argument can be used to restrict the results to only the
    "files" in the given "folder". Without the delimiter, the entire tree under
    the prefix is returned. For example, given these blobs:

        a/1.txt
        a/b/2.txt

    If you specify prefix ='a/', without a delimiter, you'll get back:

        a/1.txt
        a/b/2.txt

    However, if you specify prefix='a/' and delimiter='/', you'll get back
    only the file directly under 'a/':

        a/1.txt

    Args:
        bucket_name: string, the cloud storage bucket name.
        prefix: string, the prefix of the file path to look for blobs.
        delimiter: string, the delimiter.

    Returns:
        A list of GCS urls in the format "gs://bucket-name/file-path".
    """

    storage_client = storage.Client()

    blobs = storage_client.list_blobs(
        bucket_name, prefix=prefix, delimiter=delimiter
    )
    file_prefix = "gs://" + bucket_name + "/"
    files = []
    for blob in blobs:
        files.append(file_prefix + blob.name)
    return files

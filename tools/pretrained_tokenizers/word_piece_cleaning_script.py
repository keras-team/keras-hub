# Copyright 2023 The KerasNLP Authors
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
import os

import tqdm

# Clean these folders and saved parsed version of them.
clean_folders = ["bnwiki", "arwiki", "ruwiki", "ptwiki", "idwiki"]

for i in range(len(clean_folders)):
    clean_folder = clean_folders[i]
    output_folder = clean_folders[i] + "_parsed"
    os.mkdir(output_folder)
    for folder in tqdm.tqdm(os.listdir(clean_folder)):
        path = os.path.join(clean_folder, folder)
        os.mkdir(os.path.join(output_folder, folder))
        for file in os.listdir(path):
            article = []
            with open(os.path.join(path, file)) as f:
                for line in f:
                    if line.startswith("</doc>") or line.startswith("<doc"):
                        continue
                    else:
                        article.append(line)
            with open(os.path.join(output_folder, folder, file), "w+") as f:
                for line in article:
                    f.write(line + "\n")

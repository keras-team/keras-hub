import os
import tqdm

# Clean these folders and saved parsed version of them.
clean_folders = ["bnwiki", "arwiki", "ruwiki", "ptwiki", "idwiki"]

for i in range(len(clean_folders)):
    clean_folder = clean_folders[i]
    output_folder = clean_folders[i]+"_parsed"
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
                    f.write(line+"\n")

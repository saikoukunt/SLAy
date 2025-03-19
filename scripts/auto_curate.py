import os

from cilantropy.curation import Curator

data_dir = "D:/SLAY_data/rat_insula"
ignore = []

for root, dirs, files in os.walk(data_dir):
    for dir_name in dirs:
        if dir_name[-4:-1] == "_ks":
            ks_dir = os.path.join(root, dir_name)
            if any(i in ks_dir for i in ignore):
                continue
            print(ks_dir)
            with Curator(ks_dir) as c:
                c.auto_curate({"overwrite": True})

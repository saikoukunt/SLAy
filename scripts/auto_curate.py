from cilantropy.curation import Curator

import os

data_dir = "D:/SLAY_data/midbrain_cullen"
# ignore = ["midbrain_cullen"]

for root, dirs, files in os.walk(data_dir):
    for dir_name in dirs:
        if dir_name[-4:-1] == "_ks":
            ks_dir = os.path.join(root, dir_name)
            # for i in ignore:
            #     if i in ks_dir:
            #         continue
            print(ks_dir)
            with Curator(ks_dir) as c:
                c.auto_curate({"overwrite": True})

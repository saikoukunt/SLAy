import os

from cilantropy.curation import Curator

data_dir = "D:/SLAY_data/bijan_monkey_motor"
ignore = []

for root, dirs, files in os.walk(data_dir):
    for dir_name in dirs:
        if "ks" in dir_name and "orig" not in dir_name:
            ks_dir = os.path.join(root, dir_name)
            if any(i in ks_dir for i in ignore):
                continue
            print(ks_dir)
            with Curator(
                ks_dir,
            ) as c:
                c.auto_curate(
                    {
                        "overwrite": True,
                    }
                )

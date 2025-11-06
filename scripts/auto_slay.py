import os

import slay

if __name__ == "__main__":
    merge_params = {
        "plot_merges": True,
        "max_spikes": 500,
        "auto_accept_merges": True,
    }  #

    data_dir = "D:/SLAY_data/bijan_monkey_motor"

    # Automatically finds all KS 5 folders in the data directory
    for root, dirs, files in os.walk(data_dir):
        for dir_name in dirs:
            # if "ks" in dir_name and "orig" not in dir_name:
            if "ks4_test" in dir_name:
                ks_dir = os.path.join(root, dir_name)
                print(ks_dir)
                slay.run_slay({"KS_folder": ks_dir, **merge_params})

import json
import os
from typing import Any

import slay
from slay.schemas import OutputParams, RunParams


def main(args: dict = None) -> None:
    args = slay.parse_kilosort_params(args)
    schema = RunParams()
    params = schema.load(args)
    vals, mst, xct, rpt, mt, tt, num_merge, oc = slay.run_merge(params)
    if params["auto_accept_merges"]:
        slay.stages.accept_all_merges(vals, params)

    output: dict[str, Any] = {
        "mean_time": mst,
        "xcorr_time": xct,
        "ref_pen_time": rpt,
        "merge_time": mt,
        "total_time": tt,
        "num_merges": num_merge,
        "orig_clust": oc,
    }
    outschema = OutputParams()
    outparams = outschema.load(output)

    # if input json default to input json name -output.json
    if params.get("output_json", None) is not None:
        output_json = params["output_json"]
    elif params.get("input_json", None) is not None:
        output_json = params["input_json"].replace(".json", "-output.json")
    else:
        output_json = os.path.join(params["KS_folder"], "automerge", "run-output.json")

    with open(output_json, "w") as f:
        json.dump(outparams, f)


if __name__ == "__main__":
    args = slay.parse_cmd_line_args()
    main(args)

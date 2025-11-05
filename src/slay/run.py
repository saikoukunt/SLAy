import json
import os
from typing import Any

from .algo import run_merge
from .plot import plot_merges
from .schemas import OutputParams, RunParams
from .stages import accept_all_merges
from .utils import parse_cmd_line_args, parse_kilosort_params


def run_slay(args: dict = None) -> None:
    args = parse_kilosort_params(args)
    schema = RunParams()
    params = schema.load(args)
    params["meta_path"] = params["data_filepath"].replace(".bin", ".meta")

    vals, mst, xct, rpt, mt, tt, num_merge, oc = run_merge(params)
    if params["auto_accept_merges"]:
        accept_all_merges(vals, params)
    else:
        os.makedirs(
            os.path.join(params["KS_folder"], "automerge", "merges"), exist_ok=True
        )
        data, cl_labels, mean_wf, n_spikes, spike_times, spike_clusters, times_multi = (
            vals
        )
        if params["plot_merges"]:
            plot_merges(
                data,
                times_multi,
                mean_wf,
                params,
            )

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
    args = parse_cmd_line_args()
    run_slay(args)

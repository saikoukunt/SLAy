import os

from marshmallow import INCLUDE, Schema
from marshmallow.fields import Boolean, Field, Float, Integer, List, String


class InputFile(Field):
    default_error_messages = {
        "invalid": "Not a valid filepath",
        "not_found": "File not found",
        "not_file": "Not a file",
    }

    def __init__(self, *args, check_exists=False, **kwargs):
        self.check_exists = check_exists
        super().__init__(*args, **kwargs)

    def _deserialize(self, value, attr, data, **kwargs):
        # Ensure value is a string
        if not isinstance(value, str):
            self.fail("invalid")
        # Normalize the path
        value = os.path.abspath(value)
        value = os.path.normpath(value)
        # Ensure the file exists
        if self.check_exists and self.required and value is not None:
            print(value)
            if not os.path.exists(value):
                self.fail("not_found")
            # Ensure the path is a file
            if not os.path.isfile(value):
                self.fail("not_file")
        return value

    def _serialize(self, value, attr, obj, **kwargs):
        return os.path.normpath(value) if value else None


class InputDir(Field):
    default_error_messages = {
        "invalid": "Not a valid filepath",
        "not_found": "Directory not found",
        "not_dir": "Not a directory",
    }

    def __init__(self, *args, check_exists=False, create=False, **kwargs):
        self.check_exists = check_exists
        self.create = create
        super().__init__(*args, **kwargs)

    def _deserialize(self, value, attr, data, **kwargs):
        # Ensure value is a string
        if not isinstance(value, str):
            self.fail("invalid")

        # Ensure the file exists
        if self.create:
            os.makedirs(value, exist_ok=True)

        if self.check_exists:
            if not os.path.exists(value):
                self.fail("not_found")
            # Ensure the path is a directory
            if not os.path.isdir(value):
                self.fail("not_dir")
        return value

    def _serialize(self, value, attr, obj, **kwargs):
        return os.path.normpath(value) if value else None


class KSParams(Schema):
    # Parameters from params.py
    data_filepath = InputFile(
        required=True,
        metadata={"description": "Filepath for recording binary"},
        check_exists=True,
    )
    KS_folder = InputDir(
        required=True,
        metadata={"description": "Kilosort output directory"},
        check_exists=True,
    )
    dtype = String(
        required=False,
        load_default="int16",
        metadata={"description": "Datatype of words in recording binary"},
    )
    sample_rate = Float(
        required=True, metadata={"description": "Sampling frequency of the recording"}
    )
    n_chan = Integer(
        required=True, metadata={"description": "Number of channels in the recording"}
    )
    offset = Integer(
        required=False, metadata={"description": "Offset of the recording"}
    )
    hp_filtered = Boolean(
        required=False,
        metadata={"description": "True if recording is high-pass filtered"},
    )


class WaveformParams(Schema):
    # Parameters for waveform extraction
    pre_samples = Integer(
        required=False,
        load_default=20,
        metadata={
            "description": "Number of samples to extract before the peak of the spike"
        },
    )
    post_samples = Integer(
        required=False,
        load_default=62,
        metadata={
            "description": "Number of samples to extract after the peak of the spike"
        },
    )
    min_spikes = Integer(
        required=False,
        load_default=100,
        metadata={
            "description": "Number of spikes threshold for a cluster to undergo further stages."
        },
    )
    max_spikes = Integer(
        required=False,
        load_default=500,
        metadata={
            "description": "Maximum number of spikes per cluster used to calculate mean waveforms and train the autoencoder(-1 uses all spikes)"
        },
    )
    good_lbls = List(
        String,
        required=False,
        load_default=["good"],
        metadata={"description": "Cluster labels that denote non-noise clusters."},
    )


class CorrelogramParams(Schema):
    # Cross-correlogram parameters
    window_size = Float(
        required=False,
        load_default=0.1,
        metadata={
            "description": "The width in seconds of the cross correlogram window."
        },
    )
    xcorr_bin_width = Float(
        required=False,
        load_default=0.0005,
        metadata={
            "description": "The width in seconds of bins for cross correlogram calculation"
        },
    )
    overlap_tol = Float(
        required=False,
        load_default=5 / 30000,
        metadata={
            "description": "Overlap tolerance in seconds. Spikes within the tolerance of the reference spike time will not be counted for cross correlogram calculation"
        },
    )
    min_xcorr_rate = Float(
        required=False,
        load_default=800,
        metadata={
            "description": "Spike count threshold (per second) for cross correlograms. Cluster pairs whose cross correlogram spike rate is lower than the threshold will have a penalty applied to their cross correlation metric"
        },
    )
    xcorr_coeff = Float(
        required=False,
        load_default=0.25,
        metadata={
            "description": "Coefficient applied to cross correlation metric during final metric calculation"
        },
    )


class RefractoryParams(Schema):
    ref_pen_bin_width = Float(
        required=False,
        load_default=1,
        metadata={
            "description": "For refractory period penalty, minimum bin width in milliseconds of cross correlogram"
        },
    )
    max_viol = Float(
        required=False,
        load_default=0.15,
        metadata={
            "description": "For refractory period penalty, maximum acceptable proportion (w.r.t baseline ccg) of refractory period collisions"
        },
    )
    ref_pen_coeff = Float(
        required=False,
        load_default=1,
        metadata={"description": "Coefficient applied to refractory period penalty"},
    )


class SimilarityParams(Schema):
    # Similarity: Autoencoder parameters
    spikes_path = InputDir(
        required=False,
        load_default=None,
        metadata={"description": "Path to pre-extracted spikes folder"},
        create=True,
        check_exists=True,
        allow_none=True,
    )
    model_path = InputFile(
        required=False,
        load_default=None,
        metadata={"description": "Path to pre-trained model"},
        check_exists=True,
        allow_none=True,
    )
    sim_thresh = Float(
        required=False,
        load_default=0.4,
        metadata={
            "description": "Similarity threshold for a cluster pair to undergo further stages"
        },
    )
    ae_pre = Integer(
        required=False,
        load_default=10,
        metadata={
            "description": "For autoencoder training snippet, number of samples to extract before peak of the spike"
        },
    )
    ae_post = Integer(
        required=False,
        load_default=30,
        metadata={
            "description": "For autoencoder training snippet, number of samples to extract after peak of the spike"
        },
    )
    ae_chan = Integer(
        required=False,
        load_default=8,
        metadata={
            "description": "For autoencoder training snippet, number of channels to include"
        },
    )
    ae_noise = Boolean(
        required=False,
        load_default=False,
        metadata={
            "description": "For autoencoder training, True if autoencoder should explicitly be trained on noise snippets"
        },
    )
    ae_shft = Boolean(
        required=False,
        load_default=False,
        metadata={
            "description": "For autoencoder training, True if autoencoder should be trained on time-shifted snippets"
        },
    )
    ae_epochs = Integer(
        required=False,
        load_default=25,
        metadata={"description": "Number of epochs to train autoencoder for"},
    )


class CustomMetricsParams(KSParams, WaveformParams):
    class Meta:
        unknown = INCLUDE

    pass


class RunParams(
    KSParams,
    WaveformParams,
    CorrelogramParams,
    RefractoryParams,
    SimilarityParams,
):
    class Meta:
        unknown = INCLUDE

    output_json = InputFile(
        required=False,
        load_default=None,
        check_exists=False,
        metadata={"description": "Output JSON file for run parameters"},
    )
    final_thresh = Float(
        required=False,
        load_default=0.5,
        metadata={"description": "Final metric threshold for merge decisions"},
    )
    max_dist = Integer(
        required=False,
        load_default=10,
        metadata={
            "description": "Maximum distance between peak channels for a merge to be valid"
        },
    )
    auto_accept_merges = Boolean(
        required=False,
        load_default=False,
        metadata={"description": "True if merges should be accepted"},
    )
    plot_merges = Boolean(
        required=False,
        load_default=True,
        metadata={"description": "Whether or not to plot merges"},
    )


class OutputParams(Schema):
    mean_time = String()
    xcorr_time = String()
    ref_pen_time = String()
    merge_time = String()
    total_time = String()
    num_merges = Integer()
    orig_clust = Integer()

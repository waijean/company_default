from pandas_profiling import ProfileReport

from model_pipeline_io import read_train_file

from conf.tables import TRAIN_COMBINED_SET

if __name__ == "__main__":
    df = read_train_file(TRAIN_COMBINED_SET)

    # Only disable interactions between features (this is the main cause of huge file size in the full mode
    # since it generates 2^64 scatter plots)
    profile = ProfileReport(
        df,
        interactions={"targets": ["bankruptcy_label"]},
        plot={"histogram": {"bayesian_blocks_bins": True}},
        missing_diagrams=None,
    )

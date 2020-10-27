import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from conf.tables import TRAIN_COMBINED_SET
from modeling.utils import model_pipeline_io

RANDOM_STATE = 0


def run(train_set_name: list):
    train_data, target = model_pipeline_io.get_training_set(train_set_name)

    X_train, X_test, y_train, y_test = train_test_split(
        train_data, target, test_size=0.2, random_state=RANDOM_STATE
    )

    for n_clusters in range(2, 5):
        # fit and predict cluster label
        pipeline = make_pipeline(
            QuantileTransformer(),
            KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE),
        )
        pipeline.fit(X_train)
        X_labels = pipeline.predict(X_train)

        # compare the target label proportion in each cluster
        df = pd.DataFrame({"cluster_labels": X_labels, "target": y_train})
        cluster_df = df.groupby(["cluster_labels", "target"]).size()

        cluster_pct_df = cluster_df.groupby(level=0).apply(
            lambda x: 100 * x / float(x.sum())
        )
        combined_df = pd.concat([cluster_df, cluster_pct_df], axis=1)
        combined_df.columns = ["count", "percentage"]
        print(
            f"Clustering result for {n_clusters} clusters: \n{combined_df.to_string()}\n"
        )


if __name__ == "__main__":
    train_set_name = [TRAIN_COMBINED_SET]
    print(f"Experimeting clustering on {train_set_name} dataset...")
    run(train_set_name)

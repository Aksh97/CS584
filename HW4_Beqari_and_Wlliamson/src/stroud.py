import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance as dist
from sklearn import model_selection
from sklearn import metrics
from sklearn import decomposition
import glob
import time
import uuid
import warnings

# np.__config__.show()

# ignore numpy warnings for empty files
warnings.filterwarnings('ignore')

# LOF(k) ~ 1 means Similar density as neighbors,
# LOF(k) < 1 means Higher density than neighbors (Inlier),
# LOF(k) > 1 means Lower density than neighbors (Outlier)

class StdOUDClassifier():
    def __init__(self, k=5):
        self.k = k
        self.x_train = None
        self.x_test  = None
        self._id = uuid.uuid1()

    @staticmethod
    def read_files(parent_dir="", dir_list=[], key_func=None):
        points_df = pd.DataFrame(columns=["points", "label"])
        for dir in dir_list:
            for filename in sorted(glob.glob('./' + parent_dir + "/" + dir +
                                             '/*.txt'),
                                   key=key_func):
                points = np.loadtxt(filename)
                if points.size != 0:
                    points_df = points_df.append(
                        {
                            "points": points,
                            "label": dir
                        }, ignore_index=True)
        return points_df

    @staticmethod
    def fft_real(data):
        return np.fft.fft(data).real

    @staticmethod
    def calc_distance(point, points):
        return [[i, dist.euclidean(point, p)] for i, p in enumerate(points)]

    @staticmethod
    def reach_distance(distances, k):
        sorted_distances = sorted(distances, key=lambda x: x[1],
                                  reverse=False)[:k]
        max_d = sorted_distances[len(sorted_distances) - 1]
        return [max(max_d, d) for d in distances]

    @staticmethod
    def calc_lrd(minPts, reach_dist):
        return minPts / sum([d for i, d in reach_dist])

    @staticmethod
    def calc_lof(minPts, lrd, lrds, training):
        if training is True:
            return (1 / minPts) * (np.sum([(lrd / lrd_p) for lrd_p in lrds]) - 1.0)
        return (1 / minPts) * np.sum([(lrd_p / lrd) for lrd_p in lrds])

    @staticmethod
    def fft(points_df):
        points_df = points_df.copy(deep=True)
        points_df["fft"] = points_df["points"].apply(
            lambda row: StdOUDClassifier.fft_real(row))
        return points_df

    @staticmethod
    def distance_from_fft(points_df, relative_df=None):
        points_df = points_df.copy(deep=True)
        if relative_df is None: relative_df = points_df
        points_df["distance"] = points_df["fft"].apply(
            lambda row: StdOUDClassifier.calc_distance(
                row, relative_df["fft"].tolist()))
        return points_df

    @staticmethod
    def reachability_distance(points_df, k):
        points_df = points_df.copy(deep=True)
        points_df["reach_dist"] = points_df["distance"].apply(
            lambda row: StdOUDClassifier.reach_distance(row, k))
        return points_df

    @staticmethod
    def local_reachability_density(points_df):
        points_df = points_df.copy(deep=True)
        points_df["lrd"] = points_df["reach_dist"].apply(
            lambda row: StdOUDClassifier.calc_lrd(len(row), row))
        return points_df

    @staticmethod
    def local_outlier_factor(points_df, relative_df=None, training=False):
        points_df = points_df.copy(deep=True)
        if relative_df is None: relative_df = points_df
        points_df["lof"] = points_df[["reach_dist", "lrd"]].apply(
            lambda row: StdOUDClassifier.calc_lof(
                len(row[0]),
                row[1],
                relative_df["lrd"].take([i for i, lrd in row[0]]),
                training=training),
            axis=1)
        return points_df

    @staticmethod
    def lof_train_step(train_df, k):
        points_df = StdOUDClassifier.fft(train_df)
        points_df = StdOUDClassifier.distance_from_fft(points_df,
                                                       relative_df=None)
        points_df = StdOUDClassifier.reachability_distance(points_df, k=k)
        points_df = StdOUDClassifier.local_reachability_density(points_df)
        points_df = StdOUDClassifier.local_outlier_factor(points_df,
                                                          relative_df=None,
                                                          training=True)
        points_df = points_df.sort_values(by=['lof'], ascending=True)
        return points_df

    @staticmethod
    def lof_test_step(test_df, train_df, k):
        points_df = StdOUDClassifier.fft(test_df)
        points_df = StdOUDClassifier.distance_from_fft(points_df,
                                                       relative_df=train_df)
        points_df = StdOUDClassifier.reachability_distance(points_df, k=k)
        points_df = StdOUDClassifier.local_reachability_density(points_df)
        points_df = StdOUDClassifier.local_outlier_factor(points_df,
                                                          relative_df=train_df,
                                                          training=False)
        return points_df

    @staticmethod
    def calc_b_value(lof, lofs):
        ind_grt_lof = [i for i, lof_p in enumerate(lofs) if lof_p >= lof]
        if len(ind_grt_lof) == 0:
            return 0
        return ind_grt_lof[-1]

    @staticmethod
    def b_value(test_df, train_df):
        points_df = test_df.copy(deep=True)
        points_df["b"] = points_df["lof"].apply(
            lambda row: StdOUDClassifier.calc_b_value(row, train_df['lof'].tolist()))
        return points_df

    @staticmethod
    def p_value(test_df, train_df):
        points_df = test_df.copy(deep=True)
        points_df["p"] = points_df["b"].apply(
            lambda row: float(row) / float((len(train_df['lof'].tolist()) + 1)))
        return points_df

    @staticmethod
    def is_normal(p_value, conf_level):
        if p_value < 1 - conf_level:
            return False
        else:
            return True

    @staticmethod
    def check_normality(test_df, conf_level):
        points_df = test_df.copy(deep=True)
        points_df["y_pred"] = points_df["p"].apply(
            lambda row: StdOUDClassifier.is_normal(row, conf_level))
        return points_df

    @staticmethod
    def is_ModeM(label):
        if label == "ModeM":
            return False
        else:
            return True

    @staticmethod
    def create_eval_labels(test_df):
        points_df = test_df.copy(deep=True)
        points_df["y_true"] = points_df["label"].apply(
            lambda row: StdOUDClassifier.is_ModeM(row))
        return points_df

    def fit(self, x, y=None):
        x = x[x['label'] != "ModeM"]
        x = StdOUDClassifier.lof_train_step(x, k=self.k)
        self.x_train = x
        return self

    def predict(self, x):
        x = StdOUDClassifier.lof_test_step(x, self.x_train, k=self.k)
        x = StdOUDClassifier.b_value(x, self.x_train)
        x = StdOUDClassifier.p_value(x, self.x_train)

        conf_level = 0.99

        x = StdOUDClassifier.check_normality(x, conf_level)
        self.x_test = x
        return self.x_test

    def roc_curve(self, x):
        x = self.predict(x)
        x = StdOUDClassifier.create_eval_labels(x)

        y_true = np.array(x["y_true"].tolist())
        y_pred = np.array(x["y_pred"].tolist())
        fpr, tpr, thresholds = metrics.roc_curve(y_true,
                                                 y_pred,
                                                 pos_label=True)
        accuracy = metrics.auc(fpr, tpr)
        return fpr, tpr, accuracy

    def score(self, x, y=None):
        fpr, tpr, accuracy = self.roc_curve(x)
        return accuracy

    def get_params(self, deep=True):
        return {"k": self.k}

    def set_params(self, k):
        self.k = k
        return self

# source: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html
def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def key_func_train(x):
    str = x.split("File")[1]
    num = int(str.split(".txt")[0])
    return num

def key_func_test(x):
    str = x.split("Data")[1]
    num = int(str.split(".txt")[0])
    return num

def blue_or_range(i, k):
    colors = ["blue", "orange"]
    if i == k:
        return colors[1]
    else:
        return colors[0]


def main():

    dirs = ["ModeA", "ModeB", "ModeC", "ModeD", "ModeM"]
    points_df = StdOUDClassifier.read_files(parent_dir="train", dir_list=dirs, key_func=key_func_train)

    n_splits = 5

    # best k: 33, best auc score: 0.962
    accuracy_tracker = 0
    k_tracker = 0

    start_k = 30
    end_k = 41
    step_k = 1

    k_history = []
    score_history = []


    # 1. cross-validation
    for k in range(start_k, end_k, step_k):
        k_history.append(k)
        start = time.time()
        cv = model_selection.StratifiedKFold(n_splits=n_splits)
        model = StdOUDClassifier(k=k)
        results_skfold = model_selection.cross_val_score(model,
                                                         X=points_df,
                                                         y=points_df['label'],
                                                         cv=cv,
                                                         n_jobs=-1)
        accuracy = results_skfold.mean()
        score_history.append(accuracy)
        end = time.time()

        if accuracy > accuracy_tracker:
            accuracy_tracker = accuracy
            k_tracker = k

        print(
            "current k: {}, best k: {}, best auc score: {:.3f}, elapsed time: {}"
            .format(k, k_tracker, accuracy_tracker, round(end - start, 3)))


    # 2. cross-validation group bar plot
    # source: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html

    labels = k_history
    scores = [round(score, 2) for score in score_history]

    x = np.arange(len(labels))  # the label locations
    width = 0.8   # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x,
                    scores,
                    width,
                    color=[
                        blue_or_range(i, k_tracker)
                        for i in range(start_k, end_k, step_k)
                    ])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Score')
    ax.set_title('AUC-ROC Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('K-value')
    ax.legend()

    autolabel(rects1, ax)
    fig.tight_layout()
    plt.ylim(0.5, 1)
    plt.show()


    # 3. roc plot for best k
    # source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    cv = model_selection.StratifiedKFold(n_splits=n_splits)
    best_model = StdOUDClassifier(k=k_tracker)

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(points_df, points_df["label"])):
        best_model.fit(points_df.iloc[train])
        fpr, tpr, roc_auc = best_model.roc_curve(points_df.iloc[test])

        viz = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot(
            ax=ax, name='ROC fold {}'.format(i))

        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1],
            linestyle='--',
            lw=2,
            color='r',
            label='Chance',
            alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr,
            mean_tpr,
            color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2,
            alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr,
                    tprs_lower,
                    tprs_upper,
                    color='grey',
                    alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title='Receiver Operating Characteristic (Best k={})'.format(k_tracker))
    ax.legend(loc="lower right")
    plt.show()


    # 4. predict the real test data

    dir = ["TestWT"]
    points_test_df = StdOUDClassifier.read_files(parent_dir="test", dir_list=dir, key_func=key_func_test)

    pred_model = StdOUDClassifier(k=k_tracker)
    pred_model.fit(points_df)
    points_test_df = pred_model.predict(points_test_df)
    
    pred_model.x_test[["p"]].to_csv(r'anomaly.txt', header=None, index=None, sep=' ', mode='w')


    # 5. visualize the predictions with PCA
    # source: https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_outlier_detection.html

    x_fft = np.array(points_test_df["points"].tolist())
    pca = decomposition.PCA(n_components=2)
    pca.fit(x_fft)
    pca_components = pca.fit_transform(x_fft)

    pca_df = pd.DataFrame(data = pca_components, columns = ['pca_1', 'pca_2'])
    pca_df = pd.concat([pca_df, points_test_df[['p']]], axis = 1)
    pca_df['radius'] = pca_df['p'].apply(lambda row: 1-row)
    radius = np.array(pca_df['radius'].tolist())

    plt.title("Local Outlier Factor (LOF)")
    plt.scatter(pca_df['pca_1'], pca_df['pca_2'], color='k', s=3., label='Data Points')
    plt.scatter(pca_df['pca_1'], pca_df['pca_2'], s=1000 * radius, edgecolors='r', facecolors='none', label='Outlier Scores')
    plt.axis('tight')
    plt.xlabel("PCA C1")
    plt.ylabel("PCA C2")
    legend = plt.legend(loc='upper left')
    legend.legendHandles[0]._sizes = [10]
    legend.legendHandles[1]._sizes = [20]
    plt.show()

if __name__ == '__main__':
    main()

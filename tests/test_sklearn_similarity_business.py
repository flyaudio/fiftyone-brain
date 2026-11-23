"""
``SklearnSimilarityIndex`` 业务场景测试（纯内存、合成向量、不下载模型）。

| # | 业务场景                              | 被测 API（基类 / sklearn 后端）                                   |
|---|---------------------------------------|------------------------------------------------------------------|
| 1 | 把一批样本特征入库，核对数量/维度/ID  | ``fob.compute_similarity(..., backend="sklearn")``、``total_index_size``、``get_embeddings`` |
| 2 | “找和这张板最像的其他板”（按样本 ID） | ``index.sort_by_similarity(query=<sample_id>, k=...)``           |
| 3 | 新图提特征后，用裸向量找近邻          | ``index.sort_by_similarity(query=<np.ndarray>, k=...)``          |
| 4 | 在视图/流水线里按相似度排序浏览       | ``Dataset.sort_by_similarity(..., brain_key=...)`` 返回 ``DatasetView`` |
| 5 | 目标级检索：一张图多个缺陷框，按框查  | ``patches_field=``、``label_ids``、``get_embeddings(label_ids=)`` |
| 6 | 多张参考图联合检索，距离聚合          | ``query=[id, id]`` + ``aggregation="post-mean" / "post-min"``    |
| 7 | 清理近乎重复的样本                    | ``index.find_duplicates(...)``、``index.duplicates_view``        |
| 8 | 从大样本集挑代表性子集送去标注/复盘   | ``index.find_unique(count=...)``、``index.unique_view``          |
| 9 | 索引持久化，跨进程/会话复用           | ``dataset.load_brain_results(brain_key)`` 重载后再次检索         |

被测后端：``fiftyone.brain.internal.core.sklearn.SklearnSimilarityIndex``
（默认、纯内存、无外部向量库依赖）。

------------------------------------------------------------------------
运行前提（本机当前未安装 fiftyone，需先在目标环境准备好）::

    pip install fiftyone fiftyone-brain scikit-learn numpy pytest

运行方式（在 fiftyone-brain 仓库根目录）::

    pytest tests/test_sklearn_similarity_business.py -v

说明：
- 所有向量都是手工构造的确定性合成数据，不触发 Model Zoo 下载。
- 距离度量默认 ``cosine``；簇内向量方向接近，簇间方向正交，因此
  “同簇样本排在前面”这一结论在余弦距离下稳定成立。
- 每个用例自建数据集并在 ``finally`` 中 ``delete()``，互不污染。
"""

import numpy as np
import pytest

import fiftyone as fo
import fiftyone.brain as fob


# ---------------------------------------------------------------------------
# 合成数据：2 维向量，便于人脑核对“谁和谁像”
#
# 簇 A（方向 ≈ x 轴，模拟“药片缺粒”一类缺陷）
# 簇 B（方向 ≈ y 轴，模拟“药片异色”一类缺陷）
# O   （方向 ≈ (1,-1)，模拟离群/罕见样本）
# ---------------------------------------------------------------------------
A_VECTORS = {
    "A0": [1.00, 0.00],
    "A1": [1.00, 0.05],
    "A2": [0.98, 0.04],
}
B_VECTORS = {
    "B0": [0.00, 1.00],
    "B1": [0.05, 1.00],
    "B2": [0.04, 0.98],
}
OUTLIER_VECTOR = {"O": [1.00, -1.00]}

INTACT_LABELS = list(A_VECTORS) + list(B_VECTORS) + list(OUTLIER_VECTOR)
ALL_VECTORS = {**A_VECTORS, **B_VECTORS, **OUTLIER_VECTOR}


def _build_intact_dataset():
    """整图模式：7 个样本，每个样本 1 条整图嵌入。

    Returns:
        (dataset, embeddings, label_to_id)
        - dataset: fo.Dataset
        - embeddings: ``num_samples x num_dims`` 的 ndarray，与样本顺序对齐
        - label_to_id: 业务标签（A0/B0/...）-> 真实样本 ID 的映射
    """
    dataset = fo.Dataset()
    rr = []
    for label in INTACT_LABELS:
        rr.append("/tmp/%s.png" % label)

    dataset.add_samples(
        [fo.Sample(filepath="/tmp/%s.png" % label) for label in INTACT_LABELS]
    )

    ids = dataset.values("id")
    label_to_id = dict(zip(INTACT_LABELS, ids)) #{image_1: uuid,}
    embeddings = np.array([ALL_VECTORS[label] for label in INTACT_LABELS])
    # print("id=", dataset.values("id"))
    # print("filepath=", dataset.values("filepath"))
    # print("tags=", dataset.values("tags"))
    # print("metadata=", dataset.values("metadata"))
    # print("created_at=", dataset.values("created_at"))
    # print("last_modified_at=", dataset.values("last_modified_at"))
    return dataset, embeddings, label_to_id


def _build_intact_index(dataset, embeddings, brain_key="skl_intact"):
    """用预计算向量建一个纯内存 sklearn 相似度索引（整图模式）。"""
    return fob.compute_similarity(
        dataset,
        embeddings=embeddings,
        brain_key=brain_key,
        backend="sklearn",
        metric="cosine",
    )


def _build_patch_dataset():
    """补丁模式：3 张图，每张图 2 个检测框，共 6 条目标级嵌入。

    每张图的第 1 个框靠近簇 A，第 2 个框靠近簇 B，便于演示“同一幅
    图里不同目标可以被分别检索”。
    """
    dataset = fo.Dataset()
    samples = []
    for i in range(3):
        detections = [
            fo.Detection(label="a%d" % i, bounding_box=[0.1, 0.1, 0.2, 0.2]),
            fo.Detection(label="b%d" % i, bounding_box=[0.4, 0.4, 0.2, 0.2]),
        ]
        samples.append(
            fo.Sample(
                filepath="/tmp/patch%d.png" % i,
                ground_truth=fo.Detections(detections=detections),
            )
        )
    dataset.add_samples(samples)

    sample_ids = dataset.values("id")
    assert len(sample_ids) == 3

    # patches 模式：dict 映射 sample_id -> (num_patches x num_dims)
    embeddings = {}
    for sid in sample_ids:
        embeddings[sid] = np.array(
            [
                [1.00, 0.02],  # 框 1：靠近簇 A
                [0.02, 1.00],  # 框 2：靠近簇 B
            ]
        )
    assert isinstance(embeddings, dict)
    assert len(embeddings) == 3
    return dataset, embeddings


# ===========================================================================
# 场景 1：建库后核对索引规模、ID 与向量
# ===========================================================================
def test_01_build_index_and_inspect():
    """业务：一批图像特征入库后，先核对“入了几条、维度多少、ID 是否对齐”。"""
    dataset, embeddings, label_to_id = _build_intact_dataset()
    try:
        index = _build_intact_index(dataset, embeddings)
        print(index)
        assert isinstance(index, fob.internal.core.sklearn.SklearnSimilarityIndex)

        # 索引规模 = 样本数；整图模式下每条向量对应一个样本
        assert index.total_index_size == len(INTACT_LABELS)

        # sample_ids 与数据集 ID 一一对应；整图模式 label_ids 为 None
        assert list(index.sample_ids) == dataset.values("id")
        assert index.label_ids is None

        # 不传 ID 时 get_embeddings() 返回整个索引的三元组
        idx_emb, idx_sample_ids, idx_label_ids = index.get_embeddings()
        assert idx_emb.shape == embeddings.shape
        assert list(idx_sample_ids) == list(index.sample_ids)
        assert idx_label_ids is None
    finally:
        dataset.delete()


# ===========================================================================
# 场景 2：按样本 ID 找最相似的 k 个样本
# ===========================================================================
def test_02_query_nearest_by_sample_id():
    """业务：给定一张缺陷板，找出库里和它最像的其他板（Top-K 相似检索）。"""
    dataset, embeddings, label_to_id = _build_intact_dataset()
    try:
        index = _build_intact_index(dataset, embeddings)

        query_id = label_to_id["A0"]
        # 取最相似的 3 条；返回的是按相似度排好序的 DatasetView
        view = index.sort_by_similarity(query_id, k=3)
        result_ids = view.values("id")

        # 自身距离为 0 必然排第一
        assert result_ids[0] == query_id

        # 余弦距离下，A1/A2 与 A0 同方向，应排在 B 簇与离群点之前
        result_labels = {
            label for label, _id in label_to_id.items() if _id in result_ids
        }
        assert result_labels == {"A0", "A1", "A2"}
    finally:
        dataset.delete()


# ===========================================================================
# 场景 3：用裸查询向量（而非库内已有样本）找近邻
# ===========================================================================
def test_03_query_by_raw_vector():
    """业务：新来一张图，模型已经提好特征，直接拿向量去库里找近邻。"""
    dataset, embeddings, label_to_id = _build_intact_dataset()
    try:
        index = _build_intact_index(dataset, embeddings)

        # 一个靠近簇 A 方向、但库里不存在的新向量
        query_vector = np.array([1.0, 0.03])

        view = index.sort_by_similarity(query_vector, k=3)
        result_labels = {
            label for label, _id in label_to_id.items() if _id in view.values("id")
        }

        # 新向量落在 A 簇附近，前 3 名应全部来自 A 簇
        assert result_labels == {"A0", "A1", "A2"}
    finally:
        dataset.delete()


# ===========================================================================
# 场景 4：在数据集视图上按相似度排序（供 App / 流水线浏览）
# ===========================================================================
def test_04_sort_dataset_view_by_similarity():
    """业务：把“按相似度排序”后的视图交给标注界面或后续处理流水线。"""
    dataset, embeddings, label_to_id = _build_intact_dataset()
    try:
        index = _build_intact_index(dataset, embeddings, brain_key="skl_view")

        # 数据集级别的入口：内部按 brain_key 找回索引，返回排序后的 DatasetView
        view = dataset.sort_by_similarity(
            query=label_to_id["B0"], k=3, brain_key="skl_view"
        )

        # 返回的是一个可继续取数/过滤的数据集视图
        assert len(view) == 3

        result_labels = {
            label for label, _id in label_to_id.items() if _id in view.values("id")
        }
        assert result_labels == {"B0", "B1", "B2"}
    finally:
        dataset.delete()


# ===========================================================================
# 场景 5：补丁（目标/框）级索引 —— 一张图多个缺陷框，按框检索
# ===========================================================================

def test_05_patch_level_index_and_query_by_label_id():
    """业务：缺陷框级相似检索 —— 找到和某个药片/缺陷框最像的其他框。"""
    dataset, embeddings = _build_patch_dataset()
    try:
        index = fob.compute_similarity(
            dataset,
            patches_field="ground_truth",
            embeddings=embeddings,
            brain_key="skl_patch",
            backend="sklearn",
            metric="cosine",
        )

        # 补丁模式：索引按“框”建，label_ids 有值；sample_ids 不再作为主键
        assert index.label_ids is not None
        assert len(index.label_ids) == 6
        assert index.total_index_size == 6

        # 任取一个框的 label_id，按目标 ID 取回它的 embedding 向量
        target_label_id = index.label_ids[0]
        patch_emb, patch_sample_ids, patch_label_ids = index.get_embeddings(
            label_ids=[target_label_id]
        )
        assert patch_emb.shape == (1, 2)

        # 数据集第 0 个样本、它的第 0 个框（索引顺序与 label_ids 一致）
        sample0_id = dataset.values("id")[0]
        det0_id = dataset.values("ground_truth")[0].detections[0].id

        # 关键区分：补丁模式下返回的两类 ID 不是一回事
        #   patch_label_ids[0]  —— 框自身的标签 ID（= detections[0].id）
        #   patch_sample_ids[0] —— 该框所属样本的样本 ID（= 样本的 id）
        assert str(patch_label_ids[0]) == str(det0_id)
        assert str(patch_sample_ids[0]) == str(sample0_id)

        # 按该框做 Top-K 检索；普通数据集上补丁检索返回的是“含近邻框”的
        # 样本视图，这里只验证检索能正常返回非空结果。
        view = index.sort_by_similarity(target_label_id, k=3)
        assert len(view) >= 1
    finally:
        dataset.delete()


# ===========================================================================
# 场景 6：多查询联合检索 + 距离聚合
# ===========================================================================
@pytest.mark.parametrize(
    "aggregation", ["mean", "post-mean", "post-min", "post-max"]
)
def test_06_multi_query_aggregation(aggregation):
    """业务：拿多张参考图（如 A 类、B 类各一张）联合找“都比较像”的样本。

    多查询时用 ``aggregation`` 决定如何把多条距离合并成一个排序依据：
    - mean / post-mean：按平均相似度
    - post-min：        取到任一查询的最近距离（偏“至少像一张”）
    - post-max：        取到任一查询的最远距离（偏“每张都得像”）
    """
    dataset, embeddings, label_to_id = _build_intact_dataset()
    try:
        index = _build_intact_index(dataset, embeddings)

        query_ids = [label_to_id["A0"], label_to_id["B0"]]
        view = index.sort_by_similarity(
            query_ids, k=3, aggregation=aggregation
        )

        # 聚合后返回的是单条近邻列表（扁平 ID），长度 = k
        assert len(view) == 3
    finally:
        dataset.delete()


# ===========================================================================
# 场景 7：近重复检测 —— 清理重复/近乎重复的样本
# ===========================================================================
def test_07_find_near_duplicates():
    """业务：数据采集常有重复抓拍，入库后找出近乎重复的样本做去重。"""
    # 构造 4 个彼此几乎同向的向量（近重复簇）+ 1 个明显不同的离群点
    near_dup = np.array(
        [
            [1.00, 0.00],
            [1.00, 0.01],
            [0.99, 0.01],
            [1.00, 0.02],
            [0.00, 1.00],  # 离群，方向正交
        ]
    )
    dataset = fo.Dataset()
    dataset.add_samples(
        [fo.Sample(filepath="/tmp/dup%d.png" % i) for i in range(len(near_dup))]
    )
    try:
        index = fob.compute_similarity(
            dataset,
            embeddings=near_dup,
            brain_key="skl_dup",
            backend="sklearn",
            metric="cosine",
        )

        # fraction=0.5：自动调阈值，把约一半样本标记为重复
        index.find_duplicates(fraction=0.5)

        # 调用后填充 unique_ids / duplicate_ids；二者并集覆盖索引全部样本
        all_index_ids = set(index.sample_ids)
        assert set(index.unique_ids) | set(index.duplicate_ids) == all_index_ids
        # 近重复簇里应能挑出至少 1 个被判定为重复
        assert len(index.duplicate_ids) >= 1

        # duplicates_view 是“被判定为重复”的样本视图，可直接在 App 里复盘。
        # 注意它是方法而非属性；视图里还会带上每个重复项最近的非重复样本，
        # 因此只断言所有重复 ID 都被包含，不要求视图长度恰好等于重复数。
        dup_view = index.duplicates_view()
        assert set(index.duplicate_ids) <= set(dup_view.values("id"))
    finally:
        dataset.delete()


# ===========================================================================
# 场景 8：挑选代表性子集 —— 送标注/复盘时去掉冗余
# ===========================================================================
def test_08_find_unique_representatives():
    """业务：从大批量样本里挑 count 个彼此最不相似的代表样本，优先标注。"""
    dataset, embeddings, label_to_id = _build_intact_dataset()
    try:
        index = _build_intact_index(dataset, embeddings)

        # 挑 2 个最具代表性的样本：彼此差异尽量大的两个代表点
        index.find_unique(count=2)

        assert len(index.unique_ids) == 2
        assert len(index.duplicate_ids) == len(INTACT_LABELS) - 2

        unique_ids = set(index.unique_ids)
        all_index_ids = set(index.sample_ids)
        # 两个代表点互不相同，且都来自当前索引
        assert len(unique_ids) == 2
        assert unique_ids <= all_index_ids

        # unique_view 即挑出的代表性子集视图（方法调用）
        assert len(index.unique_view()) == 2
    finally:
        dataset.delete()


# ===========================================================================
# 场景 9：索引持久化与重载
# ===========================================================================
def test_09_persist_and_reload():
    """业务：索引建好后落库，新的 Python 会话直接按 brain_key 取回复用。"""
    dataset, embeddings, label_to_id = _build_intact_dataset()
    try:
        _build_intact_index(dataset, embeddings, brain_key="skl_persist")

        # 模拟“新会话”：不重新计算，按 brain_key 从数据集上重载索引
        reloaded = dataset.load_brain_results("skl_persist")

        # 重载后规模一致，且仍可按样本 ID 正常检索
        assert reloaded.total_index_size == len(INTACT_LABELS)
        view = reloaded.sort_by_similarity(label_to_id["A0"], k=2)
        result_labels = {
            label for label, _id in label_to_id.items() if _id in view.values("id")
        }
        assert "A0" in result_labels
    finally:
        dataset.delete()


if __name__ == "__main__":
    # test_01_build_index_and_inspect()
    # test_02_query_nearest_by_sample_id()
    # test_03_query_by_raw_vector()
    test_05_patch_level_index_and_query_by_label_id()
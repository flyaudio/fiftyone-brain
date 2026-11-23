"""
get_embeddings_from_index() 行为测试（双测，不连真实索引/模型）。

被测: fiftyone.brain.internal.core.utils.get_embeddings_from_index
"""
import numpy as np
import fiftyone.brain.internal.core.utils
import fiftyone.brain.internal.core.utils as fbu


# ---------- 测试替身 ----------

class FakeIndex:
    """记录 get_embeddings 入参，并返回预设的嵌入三元组。"""

    def __init__(self, embeddings=None, sample_ids=None, label_ids=None):
        self._embeddings = embeddings if embeddings is not None else np.empty((0, 0))
        self._sample_ids = sample_ids
        self._label_ids = label_ids
        self.calls = []  # 记录每次调用的 kwargs

    def get_embeddings(self, sample_ids=None, label_ids=None,
                       allow_missing=True, warn_missing=False):
        self.calls.append({
            "sample_ids": sample_ids,
            "label_ids": label_ids,
            "allow_missing": allow_missing,
            "warn_missing": warn_missing,
        })
        # 真实返回: (embeddings, sample_ids, label_ids)
        return self._embeddings, self._sample_ids, self._label_ids


class FakeSamples:
    """模拟 SampleCollection 被测到的最小接口。"""

    def __init__(self, is_patches=False, values_map=None,
                 label_id_path="ground_truth.detections.id"):
        self._is_patches = is_patches
        self._values_map = values_map or {}
        self._label_id_path = label_id_path

    def values(self, path, unwind=False):
        # 真实实现按路径取值；unwind=True 时把列表展开成一维
        return list(self._values_map.get(path, []))

    def _get_label_field_path(self, patches_field, attr):
        # 真实返回 (标签字段路径, 属性路径)，这里只取第二元素
        return patches_field, self._label_id_path


# ---------- 整图模式（patches_field=None） ----------

def test_sample_mode_regular_view():
    """普通样本视图：样本 ID 走 "id" 路径，label_ids 恒为 None。"""
    samples = FakeSamples(
        is_patches=False,
        values_map={"id": ["s1", "s2", "s3"]},
    )
    index = FakeIndex(sample_ids=["s1", "s2", "s3"])
    print("samples:\n", samples.values("id"))
    print(index)
    emb, sids, lids = fiftyone.brain.internal.core.utils.get_embeddings_from_index(samples, index)

    call = index.calls[0]
    assert call["sample_ids"] == ["s1", "s2", "s3"]   # values("id")
    assert call["label_ids"] is None                  # 整图模式无标签维度
    assert sids == ["s1", "s2", "s3"]
    assert lids is None


def test_sample_mode_patches_view():
    """本身就是补丁视图（_is_patches=True）：样本 ID 改走 "sample_id" 路径。"""
    samples = FakeSamples(
        is_patches=True,
        values_map={"sample_id": ["s1", "s1", "s2"]},
    )
    index = FakeIndex()

    fbu.get_embeddings_from_index(samples, index)

    call = index.calls[0]
    assert call["sample_ids"] == ["s1", "s1", "s2"]   # values("sample_id")
    assert call["label_ids"] is None


# ---------- 补丁模式（patches_field 非空） ----------

def test_patch_mode_regular_view():
    """普通视图：标签 ID 由 _get_label_field_path 解析出完整路径，不传 sample_ids。"""
    samples = FakeSamples(
        is_patches=False,
        values_map={"ground_truth.detections.id": ["l1", "l2", "l3"]},
        label_id_path="ground_truth.detections.id",
    )
    index = FakeIndex(label_ids=["l1", "l2", "l3"])

    emb, sids, lids = fbu.get_embeddings_from_index(
        samples, index, patches_field="ground_truth"
    )

    call = index.calls[0]
    assert call["sample_ids"] is None                  # 补丁模式不传样本 ID
    assert call["label_ids"] == ["l1", "l2", "l3"]     # values(id 路径, unwind=True)
    assert sids is None
    assert lids == ["l1", "l2", "l3"]


def test_patch_mode_patches_view():
    """补丁视图本身：标签 ID 直接用 "id" 路径，无需字段前缀解析。"""
    samples = FakeSamples(
        is_patches=True,
        values_map={"id": ["l1", "l2"]},
    )
    index = FakeIndex()

    fbu.get_embeddings_from_index(
        samples, index, patches_field="ground_truth"
    )

    call = index.calls[0]
    assert call["sample_ids"] is None
    assert call["label_ids"] == ["l1", "l2"]            # values("id", unwind=True)


# ---------- 透传参数 ----------

def test_missing_flags_passthrough():
    """allow_missing / warn_missing 原样传给索引，不做改动。"""
    samples = FakeSamples(values_map={"id": ["s1"]})
    index = FakeIndex()

    fbu.get_embeddings_from_index(
        samples, index, allow_missing=False, warn_missing=True
    )

    call = index.calls[0]
    assert call["allow_missing"] is False
    assert call["warn_missing"] is True



if __name__ == "__main__":
    test_sample_mode_regular_view()
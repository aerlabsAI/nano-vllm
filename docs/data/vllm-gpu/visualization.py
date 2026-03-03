import os
from collections import defaultdict
from glob import glob

import zerohertzLib as zz
from loguru import logger

YLAB = {
    "mean_e2el_ms": "Mean E2E Latency [ms]",
    "mean_ttft_ms": "Mean TTFT [ms]",
    "mean_tpot_ms": "Mean TPOT [ms]",
    "output_throughput": "Output Throughput [tok/s]",
    "total_token_throughput": "Total Token Throughput [tok/s]",
    "request_throughput": "Request Throughput [req/s]",
    "ttfts": "Time To First Token [sec]",
    "itls": "Inter-Token Latency [sec]",
}
MODELS = {
    "Qwen3-0.6B": "qwen3-0.6b",
    "Qwen2.5-3B": "qwen2.5-3b",
}
CASE_LABELS = [
    "Chunked Prefill\n(tokens=512, seqs=16)",
    "Chunked Prefill\n(tokens=1024, seqs=16)",
    "Chunked Prefill\n(tokens=2048, seqs=16)",
    "Cont. Batching\n(seqs=8, tokens=1024)",
    "Cont. Batching\n(seqs=16, tokens=1024)",
    "Cont. Batching\n(seqs=24, tokens=1024)",
]


def load_data():
    data = defaultdict(list)
    model_paths = {name: sorted(glob(f"{d}/*.json")) for name, d in MODELS.items()}
    for name, paths in model_paths.items():
        logger.info(paths)
        for path in paths:
            data[name].append(zz.util.Json(path))
    return data


def barv(data, key, mt, xticks, xlabs, colors, xmul, xgap, prefix="", show_avg=True):
    logger.info(f"Start: {key=}")
    _data = {}
    for k in data.keys():
        _data[k] = {}
    for i, (k, v) in enumerate(data.items()):
        for xtick, _v in zip(xticks, v):
            _data[k][xtick * xmul + (i - (len(data.keys()) - 1) / 2) * xgap] = _v[key]
        if show_avg:
            _data[k][xticks[-1] * xmul + (i - (len(data.keys()) - 1) / 2) * xgap] = (
                sum(_data[k].values()) / mt
            )
    fig = zz.plot.figure((25, 12))
    ax = fig.gca()
    dim = ""
    if key == "mean_tpot_ms":
        dim = "ms/tok"
    elif key == "request_throughput":
        dim = "req/s"
    elif key.endswith("ms"):
        dim = "ms"
    else:
        dim = "tok/s"
    ylim = max(max(v.values()) for v in _data.values())
    ylim *= 1.05
    n_bars = mt + 1 if show_avg else mt
    for _value, _color in zip(_data.values(), colors):
        zz.plot.barv(
            _value,
            ylim=[0, ylim],
            title="",
            colors=[_color] * n_bars,
            xlab="",
            ylab=YLAB[key],
            rot=0,
            dim=dim,
            dimsize=12,
            sign=2,
        )
    ax.set_xticks([xtick * xmul for xtick in xticks], xlabs)
    if 1 < len(_data.keys()):
        ax.legend(_data.keys(), bbox_to_anchor=(1, 1), loc="upper left")
    filename = f"{prefix}_{key}" if prefix else key
    path = zz.plot.savefig(filename)
    logger.info(f"End: {key=}")


def main():
    data = load_data()
    mt = min([len(v) for v in data.values()])
    logger.info(f"Max trial: {mt}")
    for k, v in data.items():
        if mt < len(v):
            logger.warning(
                f"다른 실험 케이스의 벤치마크 결과가 부족하여 {k}의 실험 결과들이 무시될 수 있습니다."
            )
        for _v in v:
            if _v.get("errors") is None:
                continue
            for e in _v["errors"]:
                if not e:
                    continue
                logger.error(f"[{k=}] {e=}")
    xticks = [i for i in range(mt + 1)]
    palette = os.getenv("PALETTE", "husl")
    logger.info(
        f"Color scheme: {palette=} (변경을 원할 시 PALETTE 환경 변수 설정, https://seaborn.pydata.org/tutorial/color_palettes.html)"
    )
    colors = ["red", "blue"]
    xmul = float(os.getenv("XMUL", 2.5))
    xgap = float(os.getenv("XGAP", 0.8))
    metrics = [
        "mean_e2el_ms",
        "mean_ttft_ms",
        "mean_tpot_ms",
        "output_throughput",
        "total_token_throughput",
        "request_throughput",
    ]
    if set(data.keys()) == set(MODELS.keys()):
        xticks = list(range(mt))
        xlabs = CASE_LABELS
        for metric in metrics:
            barv(data, metric, mt, xticks, xlabs, colors, xmul, xgap, show_avg=False)
    else:
        xlabs = [f"Trial #{i}" for i in range(1, mt + 1)] + ["Average"]
        for metric in metrics:
            barv(data, metric, mt, xticks, xlabs, colors, xmul, xgap)


if __name__ == "__main__":
    main()

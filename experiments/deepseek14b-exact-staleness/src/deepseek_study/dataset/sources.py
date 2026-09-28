from dataclasses import asdict, dataclass

from deepseek_study import DATASET_ID, DATASET_REVISION, DATASET_ROWS, DATASET_SHA256


@dataclass(frozen=True)
class DatasetSource:
    repo_id: str
    revision: str
    filename: str
    sha256: str
    rows: int
    storage_format: str
    question_id_field: str

    def identity(self):
        return asdict(self)


DEEPSCALER = DatasetSource(
    repo_id=DATASET_ID,
    revision=DATASET_REVISION,
    filename="data/train.parquet",
    sha256=DATASET_SHA256,
    rows=DATASET_ROWS,
    storage_format="parquet",
    question_id_field="id",
)
DAPO_17K = DatasetSource(
    repo_id="zbeeb/Staleness-GRPO-DAPO-Math-17k",
    revision="53064564abf94eac096877a61d63e92ac4217433",
    filename="data/train.jsonl",
    sha256="285a7b92a3b5a8efee80bda7506764f7ffd42a195303bc9cb77c399927a2b2c3",
    rows=17005,
    storage_format="jsonl",
    question_id_field="source_id",
)
SOURCES = {source.sha256: source for source in (DEEPSCALER, DAPO_17K)}


def source_for_sha256(value):
    try:
        return SOURCES[value]
    except KeyError as error:
        raise ValueError("Dataset differs from every locked training release") from error

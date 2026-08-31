#!/usr/bin/env python3
"""로컬 임베딩 기반 Pinecone 인덱스 생성·동기화·검색 도구.

작성자: 김진우

사용 예:
    python vdb/script/manage_pinecone.py create-index
    python vdb/script/manage_pinecone.py ingest --collection health_checkup_info
    python vdb/script/manage_pinecone.py ingest --collection health_checkup_info --delete-stale
    python vdb/script/manage_pinecone.py ingest-precomputed --source data/eyak/eyak --collection medication_info
    python vdb/script/manage_pinecone.py search --collection health_checkup_info --query "정상B가 뭐야?"
    python vdb/script/manage_pinecone.py stats
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import unicodedata
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
CHUNK_ROOT = ROOT / "vdb" / "chunk"
MANIFEST_ROOT = ROOT / "vdb" / "manifest" / "pinecone"

DEFAULT_INDEX_NAME = "heapy-rag"
EMBED_MODEL = "jhgan/ko-sroberta-multitask"
EMBED_DIMENSION = 768
METRIC = "cosine"
CLOUD = "aws"
REGION = "us-east-1"
DEFAULT_BATCH_SIZE = 64
DEFAULT_TOP_K = 3
MAX_METADATA_BYTES = 40 * 1024
MAX_RECORD_ID_LENGTH = 512
MAX_ATTEMPTS = 5

DATA_URI_PATTERN = re.compile(
    r"data:[A-Za-z0-9.+-]+/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=]+"
)
SAFE_PATH_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
COLLECTION_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")


def _validate_collection_name(collection: str) -> str:
    """collection을 안전한 로컬 폴더명과 Pinecone namespace로 검증한다."""
    normalized = collection.strip()
    if not COLLECTION_NAME_PATTERN.fullmatch(normalized):
        raise ValueError(
            "collection 이름은 1~63자의 소문자 영문, 숫자, '_', '-'만 "
            "사용하고 영문 또는 숫자로 시작해야 합니다."
        )
    return normalized


def _collection_argument(value: str) -> str:
    """명령행에서 전달받은 collection 이름을 검증한다."""
    try:
        return _validate_collection_name(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _collection_dir(collection: str) -> Path:
    """collection과 같은 이름의 청크 폴더를 반환한다."""
    normalized = _validate_collection_name(collection)
    chunk_dir = CHUNK_ROOT / normalized
    if not chunk_dir.is_dir():
        raise FileNotFoundError(
            f"collection 청크 폴더를 찾을 수 없습니다: {chunk_dir}"
        )
    return chunk_dir


def _local_collection_names() -> set[str]:
    """청크 루트에서 사용 가능한 collection 폴더명을 찾는다."""
    if not CHUNK_ROOT.is_dir():
        return set()
    return {
        path.name
        for path in CHUNK_ROOT.iterdir()
        if path.is_dir() and COLLECTION_NAME_PATTERN.fullmatch(path.name)
    }


def _load_environment(index_name_override: str | None = None) -> tuple[str, str]:
    """프로젝트 환경변수에서 Pinecone 연결 정보를 읽는다."""
    load_dotenv(ROOT / ".env")

    api_key = os.environ.get("PINECONE_API_KEY", "").strip()
    configured_name = os.environ.get(
        "PINECONE_INDEX_NAME",
        DEFAULT_INDEX_NAME,
    ).strip()
    index_name = (index_name_override or configured_name).strip()

    if not api_key:
        raise RuntimeError(
            "PINECONE_API_KEY가 설정되지 않았습니다. "
            "프로젝트 루트의 .env 파일을 확인하세요."
        )
    if not index_name:
        raise RuntimeError("PINECONE_INDEX_NAME은 빈 문자열일 수 없습니다.")
    return api_key, index_name


def _create_client(
    index_name_override: str | None = None,
) -> tuple[Pinecone, str]:
    """Pinecone 클라이언트와 사용할 인덱스 이름을 반환한다."""
    api_key, index_name = _load_environment(index_name_override)
    return Pinecone(api_key=api_key), index_name


def _read_value(value: Any, key: str, default: Any = None) -> Any:
    """SDK 응답의 객체·딕셔너리 표현을 모두 읽는다."""
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _wait_until_ready(
    client: Pinecone,
    index_name: str,
    timeout_seconds: int,
) -> Any:
    """인덱스가 준비될 때까지 제한 시간 동안 기다린다."""
    deadline = time.monotonic() + timeout_seconds

    while True:
        description = client.describe_index(index_name)
        status = _read_value(description, "status", {})
        if bool(_read_value(status, "ready", False)):
            return description

        state = _read_value(status, "state", "알 수 없음")
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"인덱스 준비 대기 시간이 초과되었습니다: {index_name} "
                f"(현재 상태: {state})"
            )
        print(f"인덱스 준비 중: {state}", flush=True)
        time.sleep(2)


def _validate_index(description: Any, index_name: str) -> None:
    """인덱스가 로컬 768차원 임베딩과 호환되는지 확인한다."""
    dimension = int(_read_value(description, "dimension", 0) or 0)
    metric = str(_read_value(description, "metric", "")).lower()
    vector_type = str(_read_value(description, "vector_type", "dense")).lower()

    if dimension != EMBED_DIMENSION or metric != METRIC or vector_type != "dense":
        raise RuntimeError(
            f"인덱스 설정이 로컬 임베딩과 호환되지 않습니다: {index_name} "
            f"(현재 dimension={dimension}, metric={metric}, type={vector_type}; "
            f"필요 dimension={EMBED_DIMENSION}, metric={METRIC}, type=dense). "
            "기존 통합 임베딩 인덱스가 아닌 새 인덱스 이름을 사용하세요."
        )


def _get_ready_index(
    client: Pinecone,
    index_name: str,
    timeout_seconds: int,
):
    """호환되는 준비 상태 인덱스에 host로 연결한다."""
    if not client.has_index(index_name):
        raise RuntimeError(
            f"Pinecone 인덱스가 없습니다: {index_name}. "
            "먼저 create-index 명령을 실행하세요."
        )

    description = _wait_until_ready(client, index_name, timeout_seconds)
    _validate_index(description, index_name)
    host = _read_value(description, "host")
    if not host:
        raise RuntimeError(f"인덱스 host를 확인할 수 없습니다: {index_name}")
    return client.Index(host=host)


def create_index(
    timeout_seconds: int,
    index_name_override: str | None,
) -> None:
    """로컬 임베딩을 저장할 768차원 dense 인덱스를 생성한다."""
    client, index_name = _create_client(index_name_override)

    if client.has_index(index_name):
        description = client.describe_index(index_name)
        _validate_index(description, index_name)
        print(f"호환되는 기존 인덱스를 사용합니다: {index_name}")
    else:
        print(
            f"인덱스를 생성합니다: {index_name} "
            f"(dimension={EMBED_DIMENSION}, metric={METRIC}, "
            f"cloud={CLOUD}, region={REGION})"
        )
        client.create_index(
            name=index_name,
            vector_type="dense",
            dimension=EMBED_DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
            deletion_protection="disabled",
            tags={"embedding_model": EMBED_MODEL, "project": "heapy-ai-health"},
        )

    description = _wait_until_ready(client, index_name, timeout_seconds)
    _validate_index(description, index_name)
    print("인덱스 준비 완료")
    print(f"  이름: {index_name}")
    print(f"  차원: {EMBED_DIMENSION}")
    print(f"  거리 측정: {METRIC}")
    print(f"  임베딩 모델: {EMBED_MODEL} (로컬)")


def _normalize_metadata_value(
    value: Any,
) -> str | int | float | bool | list[str] | None:
    """Pinecone이 지원하는 메타데이터 값으로 변환한다."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int, float)):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _build_record(
    source: Mapping[str, Any],
    source_file: Path,
    line_number: int,
    collection: str,
) -> dict[str, Any] | None:
    """JSONL 한 줄을 로컬 임베딩용 레코드로 변환한다."""
    raw_text = str(source.get("text", "")).strip()
    if not raw_text:
        print(
            f"  ! 빈 text 제외: {source_file.name}:{line_number}",
            file=sys.stderr,
        )
        return None
    if DATA_URI_PATTERN.search(raw_text):
        print(
            f"  ! Base64 이미지 포함 청크 제외: {source_file.name}:{line_number}",
            file=sys.stderr,
        )
        return None

    text = unicodedata.normalize("NFC", raw_text)
    raw_metadata = source.get("metadata", {})
    if not isinstance(raw_metadata, Mapping):
        raise ValueError(
            f"metadata는 JSON 객체여야 합니다: {source_file.name}:{line_number}"
        )

    raw_id = source.get("id") or raw_metadata.get("canonical_key")
    if not raw_id:
        raw_id = f"{source_file.stem}-{line_number - 1}"
    record_id = unicodedata.normalize("NFC", str(raw_id).strip())
    if not record_id:
        raise ValueError(f"빈 레코드 ID입니다: {source_file.name}:{line_number}")
    if len(record_id) > MAX_RECORD_ID_LENGTH:
        raise ValueError(
            f"레코드 ID가 {MAX_RECORD_ID_LENGTH}자를 초과합니다: {record_id[:80]}"
        )

    metadata: dict[str, Any] = {}
    for key, value in raw_metadata.items():
        normalized = _normalize_metadata_value(value)
        if normalized is not None:
            metadata[str(key)] = normalized

    content_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    record_payload = json.dumps(
        {"text": text, "metadata": metadata},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    record_sha256 = hashlib.sha256(record_payload.encode("utf-8")).hexdigest()
    metadata.update(
        {
            "chunk_text": text,
            "collection": collection,
            "source_file": source_file.name,
            "content_sha256": content_sha256,
            "record_sha256": record_sha256,
            "embedding_model": EMBED_MODEL,
        }
    )
    metadata_bytes = len(
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    if metadata_bytes > MAX_METADATA_BYTES:
        raise ValueError(
            f"메타데이터가 40KB를 초과합니다: {record_id} ({metadata_bytes} bytes)"
        )

    return {
        "id": record_id,
        "text": text,
        "metadata": metadata,
        "content_sha256": content_sha256,
        "record_sha256": record_sha256,
    }


def load_records(
    collection: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """컬렉션 JSONL을 읽고 정규화된 ID 기준으로 중복 제거한다."""
    chunk_dir = _collection_dir(collection)
    files = sorted(chunk_dir.rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"청크 JSONL을 찾을 수 없습니다: {chunk_dir}")

    unique_records: dict[str, dict[str, Any]] = {}
    total_rows = 0
    duplicate_rows = 0
    skipped_rows = 0

    for jsonl_file in files:
        with jsonl_file.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                total_rows += 1
                source = json.loads(line)
                if not isinstance(source, Mapping):
                    raise ValueError(
                        f"JSONL 레코드는 JSON 객체여야 합니다: "
                        f"{jsonl_file.name}:{line_number}"
                    )

                record = _build_record(
                    source=source,
                    source_file=jsonl_file,
                    line_number=line_number,
                    collection=collection,
                )
                if record is None:
                    skipped_rows += 1
                    continue

                record_id = record["id"]
                previous = unique_records.get(record_id)
                if previous is not None:
                    duplicate_rows += 1
                    if previous["record_sha256"] != record["record_sha256"]:
                        raise ValueError(
                            f"동일 ID의 본문이 서로 다릅니다: {record_id} "
                            f"({jsonl_file.name}:{line_number})"
                        )
                    continue

                unique_records[record_id] = record
                if limit is not None and len(unique_records) >= limit:
                    break
        if limit is not None and len(unique_records) >= limit:
            break

    records = list(unique_records.values())
    print(f"[{collection}] JSONL 행 {total_rows}건")
    print(f"[{collection}] 중복 제거 {duplicate_rows}건")
    print(f"[{collection}] 제외 {skipped_rows}건")
    print(f"[{collection}] 적재 대상 {len(records)}건")
    return records


def _safe_path_part(value: str) -> str:
    """인덱스·namespace 이름을 안전한 manifest 경로 조각으로 만든다."""
    return SAFE_PATH_PATTERN.sub("_", value).strip("._") or "default"


def _manifest_path(index_name: str, collection: str) -> Path:
    return (
        MANIFEST_ROOT
        / _safe_path_part(index_name)
        / f"{_safe_path_part(collection)}.json"
    )


def _precomputed_checkpoint_path(index_name: str, collection: str) -> Path:
    """사전 계산 벡터 적재의 재시작 위치를 저장할 경로를 반환한다."""
    return (
        MANIFEST_ROOT
        / _safe_path_part(index_name)
        / f"{_safe_path_part(collection)}.precomputed.json"
    )


def _load_manifest(index_name: str, collection: str) -> dict[str, str]:
    """마지막 성공 적재 상태의 ID별 본문 해시를 읽는다."""
    path = _manifest_path(index_name, collection)
    if not path.exists():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("embedding_model") != EMBED_MODEL:
        raise RuntimeError(
            f"manifest 임베딩 모델이 다릅니다: {path} "
            f"({payload.get('embedding_model')} != {EMBED_MODEL})"
        )
    records = payload.get("records", {})
    if not isinstance(records, Mapping):
        raise ValueError(f"manifest records 형식이 올바르지 않습니다: {path}")
    return {str(key): str(value) for key, value in records.items()}


def _save_manifest(
    index_name: str,
    collection: str,
    records: Mapping[str, str],
) -> None:
    """성공한 적재 상태를 원자적으로 저장해 재실행 체크포인트로 사용한다."""
    path = _manifest_path(index_name, collection)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    payload = {
        "index_name": index_name,
        "namespace": collection,
        "embedding_model": EMBED_MODEL,
        "dimension": EMBED_DIMENSION,
        "records": dict(sorted(records.items())),
    }
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == MAX_ATTEMPTS:
                    raise
                time.sleep(0.1 * attempt)
    finally:
        temporary.unlink(missing_ok=True)


def _batches(items: Iterable[Any], size: int) -> Iterator[list[Any]]:
    """항목을 지정한 크기의 배치로 나눈다."""
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _load_embeddings():
    """기존 한국어 SentenceTransformer 임베딩 모델을 로딩한다."""
    from langchain_huggingface import HuggingFaceEmbeddings

    print(f"임베딩 모델 로드 중: {EMBED_MODEL}", flush=True)
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def _retry(operation_name: str, operation) -> Any:
    """일시적인 API 오류를 지수 백오프로 재시도한다."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return operation()
        except Exception as error:
            status = getattr(error, "status", None) or getattr(
                error,
                "status_code",
                None,
            )
            retryable = status in {408, 429, 500, 502, 503, 504}
            if not retryable or attempt == MAX_ATTEMPTS:
                raise

            delay = 2 ** (attempt - 1)
            print(
                f"  ! {operation_name} 일시적 오류({status}), {delay}초 후 재시도 "
                f"({attempt}/{MAX_ATTEMPTS})",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)


def _load_precomputed_package(source_dir: Path) -> tuple[Any, dict[str, Any], str]:
    """검증된 사전 계산 임베딩 패키지의 로더와 manifest를 읽는다."""
    source_dir = source_dir.resolve()
    loader_path = source_dir / "load_records.py"
    manifest_path = source_dir / "manifest.json"
    if not loader_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "사전 계산 패키지에는 load_records.py와 manifest.json이 필요합니다: "
            f"{source_dir}"
        )

    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    embedding = manifest.get("embedding", {})
    if (
        embedding.get("model") != EMBED_MODEL
        or int(embedding.get("dimension", 0)) != EMBED_DIMENSION
        or str(embedding.get("distance_metric", "")).lower() != METRIC
        or str(embedding.get("dtype", "")).lower() != "float32"
        or embedding.get("normalized") is not True
    ):
        raise ValueError(
            "사전 계산 패키지가 현재 인덱스 규격과 호환되지 않습니다: "
            f"model={embedding.get('model')}, "
            f"dimension={embedding.get('dimension')}, "
            f"metric={embedding.get('distance_metric')}, "
            f"dtype={embedding.get('dtype')}, "
            f"normalized={embedding.get('normalized')}"
        )

    spec = importlib.util.spec_from_file_location(
        "heapy_precomputed_record_loader",
        loader_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"사전 계산 패키지 로더를 불러올 수 없습니다: {loader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "validate_package") or not hasattr(
        module,
        "iter_matrix_records",
    ):
        raise ValueError(
            "load_records.py에 validate_package와 iter_matrix_records가 필요합니다."
        )

    validation = module.validate_package(check_checksums=True)
    expected_count = int(manifest.get("records", {}).get("document_count", 0))
    if int(validation.get("documents", 0)) != expected_count:
        raise ValueError(
            "패키지 검증 문서 수가 manifest와 다릅니다: "
            f"{validation.get('documents')} != {expected_count}"
        )
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    return module, manifest, manifest_sha256


def _load_precomputed_checkpoint(
    index_name: str,
    collection: str,
    manifest_sha256: str,
    total_records: int,
    force: bool,
) -> int:
    """동일한 패키지의 마지막 성공 upsert 다음 위치를 반환한다."""
    if force:
        return 0
    path = _precomputed_checkpoint_path(index_name, collection)
    if not path.is_file():
        return 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("source_manifest_sha256") != manifest_sha256:
        raise ValueError(
            "기존 체크포인트와 현재 패키지가 다릅니다. 전체를 다시 올리려면 "
            "--force를 사용하세요."
        )
    if int(payload.get("total_records", 0)) != total_records:
        raise ValueError("기존 체크포인트와 현재 패키지의 레코드 수가 다릅니다.")
    next_index = int(payload.get("next_index", 0))
    if not 0 <= next_index <= total_records:
        raise ValueError(f"체크포인트 위치가 올바르지 않습니다: {next_index}")
    return next_index


def _save_precomputed_checkpoint(
    index_name: str,
    collection: str,
    source_dir: Path,
    manifest_sha256: str,
    total_records: int,
    next_index: int,
) -> None:
    """사전 계산 벡터의 마지막 성공 위치를 원자적으로 기록한다."""
    path = _precomputed_checkpoint_path(index_name, collection)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    payload = {
        "index_name": index_name,
        "namespace": collection,
        "embedding_model": EMBED_MODEL,
        "dimension": EMBED_DIMENSION,
        "source_dir": str(source_dir.resolve()),
        "source_manifest_sha256": manifest_sha256,
        "total_records": total_records,
        "next_index": next_index,
    }
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == MAX_ATTEMPTS:
                    raise
                time.sleep(0.1 * attempt)
    finally:
        temporary.unlink(missing_ok=True)


def _build_precomputed_vector(record: Mapping[str, Any], collection: str) -> dict[str, Any]:
    """사전 계산 레코드를 HEAPY 검색 메타데이터 규격으로 변환한다."""
    record_id = str(record.get("id", "")).strip()
    text = str(record.get("text", "")).strip()
    vector = record.get("embedding")
    if not record_id or not text or vector is None:
        raise ValueError("사전 계산 레코드에 id, text, embedding이 필요합니다.")
    if len(vector) != EMBED_DIMENSION:
        raise ValueError(
            f"임베딩 차원이 올바르지 않습니다: {record_id} "
            f"({len(vector)} != {EMBED_DIMENSION})"
        )

    raw_metadata = record.get("metadata", {})
    if not isinstance(raw_metadata, Mapping):
        raise ValueError(f"metadata는 JSON 객체여야 합니다: {record_id}")
    metadata = {
        str(key): normalized
        for key, value in raw_metadata.items()
        if (normalized := _normalize_metadata_value(value)) is not None
    }
    metadata.update(
        {
            "chunk_text": text,
            "collection": collection,
            "source_file": "documents.jsonl",
            "embedding_model": EMBED_MODEL,
        }
    )
    metadata_bytes = len(
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    if metadata_bytes > MAX_METADATA_BYTES:
        raise ValueError(
            f"메타데이터가 40KB를 초과합니다: {record_id} ({metadata_bytes} bytes)"
        )
    values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
    return {"id": record_id, "values": values, "metadata": metadata}


def ingest_precomputed(
    source: Path,
    collection: str,
    batch_size: int,
    limit: int | None,
    dry_run: bool,
    force: bool,
    timeout_seconds: int,
    index_name_override: str | None,
) -> None:
    """검증된 기존 임베딩을 재계산 없이 Pinecone namespace에 적재한다."""
    source_dir = source if source.is_absolute() else ROOT / source
    loader, manifest, manifest_sha256 = _load_precomputed_package(source_dir)
    total_records = int(manifest["records"]["document_count"])
    print(f"패키지 검증 완료: {total_records}건, {EMBED_DIMENSION}차원, {METRIC}")
    print(f"대상 namespace: {collection}")
    if dry_run:
        preview = next(loader.iter_matrix_records(validate=False))
        vector = _build_precomputed_vector(preview, collection)
        print("dry-run 완료: Pinecone 연결·upsert를 수행하지 않았습니다.")
        print(
            json.dumps(
                {
                    "id": vector["id"],
                    "dimension": len(vector["values"]),
                    "metadata_keys": sorted(vector["metadata"]),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    client, index_name = _create_client(index_name_override)
    index = _get_ready_index(client, index_name, timeout_seconds)
    next_index = _load_precomputed_checkpoint(
        index_name,
        collection,
        manifest_sha256,
        total_records,
        force,
    )
    stop_index = total_records
    if limit is not None:
        stop_index = min(next_index + limit, total_records)
    if next_index >= stop_index:
        print(f"적재할 신규 레코드가 없습니다: {next_index}/{total_records}")
        return

    batch: list[dict[str, Any]] = []
    batch_next_index = next_index
    for index_number, record in enumerate(loader.iter_matrix_records(validate=False)):
        if index_number < next_index:
            continue
        if index_number >= stop_index:
            break
        batch.append(_build_precomputed_vector(record, collection))
        batch_next_index = index_number + 1
        if len(batch) < batch_size and batch_next_index < stop_index:
            continue

        payload = batch
        _retry(
            "upsert",
            lambda payload=payload: index.upsert(
                vectors=payload,
                namespace=collection,
                show_progress=False,
            ),
        )
        _save_precomputed_checkpoint(
            index_name,
            collection,
            source_dir,
            manifest_sha256,
            total_records,
            batch_next_index,
        )
        print(f"  upsert {batch_next_index}/{total_records}", flush=True)
        batch = []

    print(
        f"사전 계산 벡터 적재 완료: index={index_name}, "
        f"namespace={collection}, 진행={batch_next_index}/{total_records}"
    )


def ingest(
    collection: str,
    batch_size: int,
    limit: int | None,
    dry_run: bool,
    delete_stale: bool,
    force: bool,
    timeout_seconds: int,
    index_name_override: str | None,
) -> None:
    """신규·변경 청크만 로컬 임베딩해 Pinecone namespace와 동기화한다."""
    if limit is not None and delete_stale:
        raise ValueError("--limit과 --delete-stale은 함께 사용할 수 없습니다.")

    records = load_records(collection, limit=limit)
    if not records:
        raise RuntimeError(f"적재할 청크가 없습니다: {collection}")

    _, index_name = _load_environment(index_name_override)
    manifest = _load_manifest(index_name, collection)
    source_hashes = {record["id"]: record["record_sha256"] for record in records}
    changed = [
        record
        for record in records
        if force or manifest.get(record["id"]) != record["record_sha256"]
    ]
    stale_ids = sorted(set(manifest) - set(source_hashes)) if delete_stale else []

    print(f"[{collection}] 신규·변경 {len(changed)}건")
    print(f"[{collection}] 변경 없음 {len(records) - len(changed)}건")
    print(f"[{collection}] 삭제 대상 {len(stale_ids)}건")

    if dry_run:
        print("dry-run 완료: 임베딩·upsert·delete를 수행하지 않았습니다.")
        if changed:
            preview = {key: value for key, value in changed[0].items() if key != "text"}
            print(json.dumps(preview, ensure_ascii=False, indent=2))
        return

    client, index_name = _create_client(index_name_override)
    index = _get_ready_index(client, index_name, timeout_seconds)
    checkpoint = dict(manifest)

    if changed:
        embeddings = _load_embeddings()
        completed = 0
        for batch in _batches(changed, batch_size):
            vectors = embeddings.embed_documents([record["text"] for record in batch])
            payload = []
            for record, vector in zip(batch, vectors, strict=True):
                if len(vector) != EMBED_DIMENSION:
                    raise ValueError(
                        f"임베딩 차원이 올바르지 않습니다: {record['id']} "
                        f"({len(vector)} != {EMBED_DIMENSION})"
                    )
                payload.append(
                    {
                        "id": record["id"],
                        "values": vector,
                        "metadata": record["metadata"],
                    }
                )

            _retry(
                "upsert",
                lambda payload=payload: index.upsert(
                    vectors=payload,
                    namespace=collection,
                    show_progress=False,
                ),
            )
            for record in batch:
                checkpoint[record["id"]] = record["record_sha256"]
            _save_manifest(index_name, collection, checkpoint)
            completed += len(batch)
            print(f"  upsert {completed}/{len(changed)}", flush=True)

    deleted = 0
    for batch in _batches(stale_ids, 1000):
        _retry(
            "delete",
            lambda batch=batch: index.delete(ids=batch, namespace=collection),
        )
        for record_id in batch:
            checkpoint.pop(record_id, None)
        _save_manifest(index_name, collection, checkpoint)
        deleted += len(batch)
        print(f"  delete {deleted}/{len(stale_ids)}", flush=True)

    if limit is None:
        for record_id, content_hash in source_hashes.items():
            if record_id in checkpoint:
                checkpoint[record_id] = content_hash
        _save_manifest(index_name, collection, checkpoint)

    print(
        f"동기화 완료: index={index_name}, namespace={collection}, "
        f"upsert={len(changed)}, delete={len(stale_ids)}"
    )


def search(
    collection: str,
    query: str,
    top_k: int,
    timeout_seconds: int,
    index_name_override: str | None,
) -> None:
    """질문을 로컬 임베딩한 뒤 Pinecone namespace에서 검색한다."""
    collection = _validate_collection_name(collection)
    client, index_name = _create_client(index_name_override)
    index = _get_ready_index(client, index_name, timeout_seconds)
    embeddings = _load_embeddings()
    query_vector = embeddings.embed_query(query)
    if len(query_vector) != EMBED_DIMENSION:
        raise ValueError(
            f"질문 임베딩 차원이 올바르지 않습니다: "
            f"{len(query_vector)} != {EMBED_DIMENSION}"
        )

    results = index.query(
        namespace=collection,
        vector=query_vector,
        top_k=top_k,
        include_values=False,
        include_metadata=True,
    )
    matches = list(_read_value(results, "matches", []) or [])

    print(f"질문: {query}")
    print(f"index: {index_name}")
    print(f"namespace: {collection}")
    print(f"검색 결과: {len(matches)}건")
    for rank, match in enumerate(matches, start=1):
        metadata = dict(_read_value(match, "metadata", {}) or {})
        score = float(_read_value(match, "score", 0.0) or 0.0)
        record_id = str(_read_value(match, "id", ""))
        print("=" * 80)
        print(f"{rank}위 | ID={record_id} | 점수={score:.4f}")
        print(f"출처: {metadata.get('source_label', '출처 미상')}")
        if metadata.get("source"):
            print(f"URL: {metadata['source']}")
        print(f"본문: {metadata.get('chunk_text', '')}")


def stats(timeout_seconds: int, index_name_override: str | None) -> None:
    """인덱스와 namespace별 레코드 수를 출력한다."""
    client, index_name = _create_client(index_name_override)
    index = _get_ready_index(client, index_name, timeout_seconds)
    response = index.describe_index_stats()
    namespaces = _read_value(response, "namespaces", {}) or {}

    print(f"index: {index_name}")
    print(f"dimension: {_read_value(response, 'dimension', EMBED_DIMENSION)}")
    collection_names = _local_collection_names() | set(namespaces)
    for collection in sorted(collection_names):
        namespace = _read_value(namespaces, collection, {})
        count = int(_read_value(namespace, "vector_count", 0) or 0)
        print(f"{collection}: {count}건")


def _positive_int(value: str) -> int:
    """1 이상의 정수 인자를 검증한다."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("1 이상의 정수를 입력하세요.")
    return parsed


def _batch_size(value: str) -> int:
    """2MB 요청 제한을 고려한 로컬 벡터 배치 크기를 검증한다."""
    parsed = _positive_int(value)
    if parsed > 100:
        raise argparse.ArgumentTypeError("벡터 적재 배치는 최대 100건입니다.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Pinecone 관리 명령행 파서를 구성한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-name",
        help=".env의 PINECONE_INDEX_NAME 대신 사용할 인덱스 이름",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_int,
        default=180,
        help="인덱스 준비 대기 제한 시간(기본값: 180초)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("create-index", help="768차원 dense 인덱스 생성")

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="신규·변경 청크 로컬 임베딩 및 namespace 동기화",
    )
    ingest_parser.add_argument(
        "--collection",
        required=True,
        type=_collection_argument,
        help="vdb/chunk 아래의 청크 폴더명이자 Pinecone namespace 이름",
    )
    ingest_parser.add_argument(
        "--batch-size",
        type=_batch_size,
        default=DEFAULT_BATCH_SIZE,
        help=f"임베딩·upsert 배치 크기(기본값: {DEFAULT_BATCH_SIZE}, 최대: 100)",
    )
    ingest_parser.add_argument("--limit", type=_positive_int)
    ingest_parser.add_argument("--dry-run", action="store_true")
    ingest_parser.add_argument(
        "--delete-stale",
        action="store_true",
        help="마지막 manifest에는 있지만 현재 청크에 없는 ID를 Pinecone에서도 삭제",
    )
    ingest_parser.add_argument(
        "--force",
        action="store_true",
        help="manifest 해시와 관계없이 모든 청크를 다시 임베딩·upsert",
    )

    precomputed_parser = subparsers.add_parser(
        "ingest-precomputed",
        help="검증된 기존 임베딩을 재계산 없이 namespace에 적재",
    )
    precomputed_parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="load_records.py와 manifest.json이 있는 패키지 폴더",
    )
    precomputed_parser.add_argument(
        "--collection",
        required=True,
        type=_collection_argument,
        help="Pinecone namespace 이름",
    )
    precomputed_parser.add_argument(
        "--batch-size",
        type=_batch_size,
        default=DEFAULT_BATCH_SIZE,
        help=f"upsert 배치 크기(기본값: {DEFAULT_BATCH_SIZE}, 최대: 100)",
    )
    precomputed_parser.add_argument(
        "--limit",
        type=_positive_int,
        help="이번 실행에서 추가로 적재할 최대 레코드 수",
    )
    precomputed_parser.add_argument("--dry-run", action="store_true")
    precomputed_parser.add_argument(
        "--force",
        action="store_true",
        help="기존 진행 위치를 무시하고 처음부터 다시 upsert",
    )

    search_parser = subparsers.add_parser("search", help="로컬 임베딩 벡터 검색")
    search_parser.add_argument(
        "--collection",
        required=True,
        type=_collection_argument,
        help="검색할 Pinecone namespace 이름",
    )
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--top-k", type=_positive_int, default=DEFAULT_TOP_K)

    subparsers.add_parser("stats", help="namespace별 레코드 수 확인")
    return parser


def main() -> int:
    """선택한 Pinecone 관리 명령을 실행한다."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "create-index":
            create_index(args.timeout_seconds, args.index_name)
        elif args.command == "ingest":
            ingest(
                collection=args.collection,
                batch_size=args.batch_size,
                limit=args.limit,
                dry_run=args.dry_run,
                delete_stale=args.delete_stale,
                force=args.force,
                timeout_seconds=args.timeout_seconds,
                index_name_override=args.index_name,
            )
        elif args.command == "ingest-precomputed":
            ingest_precomputed(
                source=args.source,
                collection=args.collection,
                batch_size=args.batch_size,
                limit=args.limit,
                dry_run=args.dry_run,
                force=args.force,
                timeout_seconds=args.timeout_seconds,
                index_name_override=args.index_name,
            )
        elif args.command == "search":
            search(
                collection=args.collection,
                query=args.query.strip(),
                top_k=args.top_k,
                timeout_seconds=args.timeout_seconds,
                index_name_override=args.index_name,
            )
        elif args.command == "stats":
            stats(args.timeout_seconds, args.index_name)
        else:
            parser.error(f"지원하지 않는 명령입니다: {args.command}")
    except (OSError, RuntimeError, TimeoutError, ValueError, json.JSONDecodeError) as error:
        print(f"오류: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

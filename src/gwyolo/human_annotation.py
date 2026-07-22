from __future__ import annotations

import json
import os
import platform
import re
import shlex
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from .io import atomic_write_json, atomic_write_text, file_sha256
from .mask_audit import _atomic_save_npz, _load_npz_mask


_ANNOTATOR_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_FORBIDDEN_TASK_FIELDS = {
    "glitch_id",
    "ifo",
    "ml_label",
    "network_gps_block",
    "numeric_sample_path",
    "numeric_sample_sha256",
    "observing_run",
    "weak_mask_key",
}
_FORBIDDEN_ARRAY_KEYS = {"mask", "glitch_mask", "chirp_mask", "weak_mask"}


def _execution_provenance() -> dict[str, Any]:
    return {
        "code_commit": os.environ.get("GWYOLO_CODE_COMMIT"),
        "exact_command": " ".join(shlex.quote(part) for part in sys.argv),
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }


def _load_blinded_annotation_tasks(
    manifest_path: str | Path,
) -> tuple[Path, list[dict[str, Any]]]:
    path = Path(manifest_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        tasks = [json.loads(line) for line in handle if line.strip()]
    if not tasks:
        raise ValueError("human annotation requires non-empty blinded tasks")
    audit_ids = [str(task.get("audit_id", "")) for task in tasks]
    if any(not value for value in audit_ids) or len(set(audit_ids)) != len(audit_ids):
        raise ValueError("human annotation task IDs are missing or repeated")
    for task in tasks:
        audit_id = str(task["audit_id"])
        if _FORBIDDEN_TASK_FIELDS & set(task):
            raise ValueError(f"annotator task exposes internal target metadata: {audit_id}")
        if task.get("annotation_status") != "pending":
            raise ValueError(f"annotator task is not a frozen pending task: {audit_id}")
        if task.get("required_independent_annotators") != 3:
            raise ValueError(f"annotator task does not require three reviewers: {audit_id}")
        if task.get("required_annotation_key") != "mask":
            raise ValueError(f"annotator task has the wrong output key: {audit_id}")
        if not str(task.get("annotation_task_hash", "")):
            raise ValueError(f"annotator task lacks its frozen hash: {audit_id}")
        blind_path = Path(str(task.get("blinded_input_path", ""))).resolve()
        if (
            not blind_path.is_file()
            or str(task.get("blinded_input_sha256", "")) != file_sha256(blind_path)
        ):
            raise ValueError(f"annotator blinded input hash mismatch: {audit_id}")
        with np.load(blind_path, allow_pickle=False) as arrays:
            keys = set(arrays.files)
            if "features" not in keys or keys & _FORBIDDEN_ARRAY_KEYS:
                raise ValueError(f"annotator input exposes a target: {audit_id}")
            if sorted(keys) != sorted(task.get("blinded_input_keys", [])):
                raise ValueError(f"annotator input key inventory mismatch: {audit_id}")
            features = np.asarray(arrays["features"])
        shape = tuple(int(value) for value in task.get("mask_shape", []))
        if (
            len(shape) < 2
            or features.shape != shape
            or not np.isfinite(features).all()
        ):
            raise ValueError(f"annotator feature tensor is invalid: {audit_id}")
    return path, tasks


def _annotation_row(
    task: dict[str, Any], annotator_id: str, protocol_version: str, mask_path: Path
) -> dict[str, Any]:
    return {
        "audit_id": str(task["audit_id"]),
        "annotator_id": annotator_id,
        "mask_path": str(mask_path.resolve()),
        "mask_sha256": file_sha256(mask_path),
        "blinded_to_weak_mask": True,
        "protocol_version": protocol_version,
        "annotation_task_hash": str(task["annotation_task_hash"]),
    }


class HumanMaskAnnotationSession:
    """Stateful, target-free annotation session for one independent reviewer."""

    def __init__(
        self,
        task_manifest: str | Path,
        annotator_id: str,
        output_dir: str | Path,
        protocol_version: str = "gravityspy_human_mask_blind_v1",
    ) -> None:
        if _ANNOTATOR_ID.fullmatch(annotator_id) is None:
            raise ValueError("annotator ID must be a stable non-identifying slug")
        if not protocol_version or len(protocol_version) > 128:
            raise ValueError("annotation protocol version is invalid")
        self.task_manifest, self.tasks = _load_blinded_annotation_tasks(task_manifest)
        self.annotator_id = annotator_id
        self.protocol_version = protocol_version
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.partial_manifest = self.output_dir / f"annotations.{annotator_id}.partial.jsonl"
        self.final_manifest = self.output_dir / f"annotations.{annotator_id}.jsonl"
        self.final_report = self.output_dir / f"annotations.{annotator_id}.report.json"
        self._task_by_id = {str(task["audit_id"]): task for task in self.tasks}
        self._rows: dict[str, dict[str, Any]] = {}
        source = self.final_manifest if self.final_manifest.exists() else self.partial_manifest
        if source.exists():
            with source.open("r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            for row in rows:
                self._validate_saved_row(row)
                audit_id = str(row["audit_id"])
                if audit_id in self._rows:
                    raise ValueError("annotation session repeats a saved task")
                self._rows[audit_id] = row

    @property
    def finalized(self) -> bool:
        return self.final_manifest.exists()

    def _validate_saved_row(self, row: dict[str, Any]) -> None:
        audit_id = str(row.get("audit_id", ""))
        task = self._task_by_id.get(audit_id)
        path = Path(str(row.get("mask_path", ""))).resolve()
        if (
            task is None
            or row.get("annotator_id") != self.annotator_id
            or row.get("protocol_version") != self.protocol_version
            or row.get("blinded_to_weak_mask") is not True
            or row.get("annotation_task_hash") != task.get("annotation_task_hash")
            or not path.is_file()
            or row.get("mask_sha256") != file_sha256(path)
        ):
            raise ValueError(f"saved annotation failed replay: {audit_id}")
        mask = _load_npz_mask(path, "mask")
        if list(mask.shape) != list(task["mask_shape"]):
            raise ValueError(f"saved annotation shape mismatch: {audit_id}")

    def progress(self) -> dict[str, Any]:
        return {
            "annotator_id": self.annotator_id,
            "protocol_version": self.protocol_version,
            "tasks": len(self.tasks),
            "completed": len(self._rows),
            "remaining": len(self.tasks) - len(self._rows),
            "finalized": self.finalized,
            "task_manifest_sha256": file_sha256(self.task_manifest),
            "target_fields_exposed": False,
        }

    def task_payload(self, index: int) -> dict[str, Any]:
        if not 0 <= index < len(self.tasks):
            raise IndexError("annotation task index is outside the frozen task set")
        task = self.tasks[index]
        audit_id = str(task["audit_id"])
        blind_path = Path(str(task["blinded_input_path"])).resolve()
        with np.load(blind_path, allow_pickle=False) as arrays:
            features = np.asarray(arrays["features"], dtype=np.float32)
            ifos = [str(value) for value in arrays["ifos"].tolist()] if "ifos" in arrays else []
            q_values = (
                [float(value) for value in arrays["q_values"].tolist()]
                if "q_values" in arrays
                else []
            )
        height, width = features.shape[-2:]
        planes = features.reshape(-1, height, width)
        normalized = np.zeros_like(planes, dtype=np.uint8)
        for plane_index, plane in enumerate(planes):
            lower, upper = np.percentile(plane, (1.0, 99.0))
            if upper > lower:
                scaled = np.clip((plane - lower) / (upper - lower), 0.0, 1.0)
                normalized[plane_index] = np.rint(255.0 * scaled).astype(np.uint8)
        labels = []
        if features.ndim == 4 and len(ifos) == features.shape[0] and len(q_values) == features.shape[1]:
            labels = [f"{ifo} / Q={q:g}" for ifo in ifos for q in q_values]
        elif features.ndim == 3 and len(q_values) == features.shape[0]:
            labels = [f"Q={q:g}" for q in q_values]
        else:
            labels = [f"plane {value + 1}" for value in range(len(planes))]
        saved = self._rows.get(audit_id)
        mask = np.zeros(features.shape, dtype=np.uint8)
        if saved is not None:
            mask = _load_npz_mask(saved["mask_path"], "mask").astype(np.uint8)
        return {
            "index": index,
            "total": len(self.tasks),
            "audit_id": audit_id,
            "annotation_task_hash": str(task["annotation_task_hash"]),
            "shape": list(features.shape),
            "height": height,
            "width": width,
            "plane_labels": labels,
            "features_u8": normalized.reshape(-1).tolist(),
            "mask": mask.reshape(-1).tolist(),
            "saved": saved is not None,
            "finalized": self.finalized,
            "target_fields_exposed": False,
        }

    def save_mask(self, index: int, values: Any) -> dict[str, Any]:
        if self.finalized:
            raise FileExistsError("finalized human annotations are immutable")
        if not 0 <= index < len(self.tasks):
            raise IndexError("annotation task index is outside the frozen task set")
        task = self.tasks[index]
        shape = tuple(int(value) for value in task["mask_shape"])
        array = np.asarray(values)
        if array.size != int(np.prod(shape)) or not np.isin(array, (0, 1, False, True)).all():
            raise ValueError("human annotation mask must be binary and exactly aligned")
        mask = array.astype(np.uint8).reshape(shape)
        target = self.output_dir / "masks" / f"{task['audit_id']}.npz"
        _atomic_save_npz(target, {"mask": mask})
        row = _annotation_row(task, self.annotator_id, self.protocol_version, target)
        self._rows[str(task["audit_id"])] = row
        ordered = [
            self._rows[str(item["audit_id"])]
            for item in self.tasks
            if str(item["audit_id"]) in self._rows
        ]
        atomic_write_text(
            self.partial_manifest,
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in ordered),
        )
        return {**self.progress(), "saved_audit_id": str(task["audit_id"])}

    def finalize(self) -> dict[str, Any]:
        if self.finalized:
            raise FileExistsError("finalized human annotations are immutable")
        if set(self._rows) != set(self._task_by_id):
            raise ValueError("cannot finalize an incomplete human annotation session")
        ordered = [self._rows[str(task["audit_id"])] for task in self.tasks]
        for row in ordered:
            self._validate_saved_row(row)
        atomic_write_text(
            self.final_manifest,
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in ordered),
        )
        result = {
            "status": "completed_independent_blinded_human_mask_annotation",
            "scientific_claim_allowed": False,
            "annotator_id": self.annotator_id,
            "protocol_version": self.protocol_version,
            "tasks": len(ordered),
            "complete_frozen_task_coverage": True,
            "blinded_to_weak_mask": True,
            "target_fields_exposed": False,
            "task_manifest_path": str(self.task_manifest),
            "task_manifest_sha256": file_sha256(self.task_manifest),
            "annotation_manifest_path": str(self.final_manifest),
            "annotation_manifest_sha256": file_sha256(self.final_manifest),
            **_execution_provenance(),
        }
        atomic_write_json(self.final_report, result)
        return result


def merge_human_mask_annotation_manifests(
    task_manifest: str | Path,
    annotation_manifests: list[str | Path],
    output_path: str | Path,
    minimum_annotators: int = 3,
) -> dict[str, Any]:
    """Merge complete independent reviewer manifests into the audit evaluator input."""

    if minimum_annotators < 3 or minimum_annotators % 2 == 0:
        raise ValueError("human mask consensus needs an odd panel of at least three")
    if len(annotation_manifests) != minimum_annotators:
        raise ValueError("human mask merge requires exactly the declared annotator panel")
    task_path, tasks = _load_blinded_annotation_tasks(task_manifest)
    output = Path(output_path).resolve()
    report_path = output.with_suffix(output.suffix + ".report.json")
    if output.exists() or report_path.exists():
        raise FileExistsError("merged human annotation manifests are immutable")
    task_by_id = {str(task["audit_id"]): task for task in tasks}
    expected_ids = set(task_by_id)
    annotators = set()
    protocol_versions = set()
    combined = []
    source_manifests = []
    for raw_path in annotation_manifests:
        path = Path(raw_path).resolve()
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        ids = [str(row.get("audit_id", "")) for row in rows]
        row_annotators = {str(row.get("annotator_id", "")) for row in rows}
        if len(rows) != len(tasks) or len(set(ids)) != len(rows) or set(ids) != expected_ids:
            raise ValueError("each annotator manifest must exactly cover the frozen tasks")
        if len(row_annotators) != 1 or "" in row_annotators:
            raise ValueError("each annotation manifest must contain one stable annotator ID")
        annotator = next(iter(row_annotators))
        if annotator in annotators:
            raise ValueError("human mask merge requires independent annotator IDs")
        annotators.add(annotator)
        for row in rows:
            audit_id = str(row["audit_id"])
            task = task_by_id[audit_id]
            mask_path = Path(str(row.get("mask_path", ""))).resolve()
            if (
                row.get("blinded_to_weak_mask") is not True
                or row.get("annotation_task_hash") != task.get("annotation_task_hash")
                or not str(row.get("protocol_version", ""))
                or not mask_path.is_file()
                or row.get("mask_sha256") != file_sha256(mask_path)
            ):
                raise ValueError(f"human annotation row failed replay: {audit_id}")
            mask = _load_npz_mask(mask_path, "mask")
            if list(mask.shape) != list(task["mask_shape"]):
                raise ValueError(f"human annotation row shape mismatch: {audit_id}")
            protocol_versions.add(str(row["protocol_version"]))
            combined.append(row)
        source_manifests.append({"path": str(path), "sha256": file_sha256(path)})
    if len(protocol_versions) != 1:
        raise ValueError("human mask annotators used different protocol versions")
    combined.sort(key=lambda row: (str(row["audit_id"]), str(row["annotator_id"])))
    atomic_write_text(
        output, "".join(json.dumps(row, sort_keys=True) + "\n" for row in combined)
    )
    result = {
        "status": "completed_blinded_human_mask_annotation_panel",
        "scientific_claim_allowed": False,
        "tasks": len(tasks),
        "annotators": sorted(annotators),
        "annotations": len(combined),
        "complete_frozen_task_coverage": True,
        "independent_annotator_ids": True,
        "blinded_to_weak_mask": True,
        "target_fields_exposed": False,
        "protocol_version": next(iter(protocol_versions)),
        "task_manifest_path": str(task_path),
        "task_manifest_sha256": file_sha256(task_path),
        "source_manifests": source_manifests,
        "output_path": str(output),
        "output_sha256": file_sha256(output),
        **_execution_provenance(),
    }
    atomic_write_json(report_path, result)
    return result


_ANNOTATION_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GW-YOLO blinded mask annotation</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#111;color:#eee}header,main{max-width:900px;margin:auto;padding:12px}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}button,select,input{font-size:15px;padding:7px}
#stage{position:relative;width:768px;height:768px;max-width:95vw;max-height:95vw;background:#000;margin-top:12px}
canvas{position:absolute;inset:0;width:100%;height:100%;image-rendering:pixelated;touch-action:none}
#status{color:#9fd}.warn{color:#fc8}button:disabled{opacity:.45}
</style></head><body><header><h2>Blinded glitch-mask annotation</h2>
<p class="warn">Only target-free numeric features are shown. Work independently; do not request weak masks or other annotators' results.</p>
<div id="status">Loading…</div></header><main>
<div class="row"><button id="prev">Previous</button><button id="next">Next</button>
<label>Plane <select id="plane"></select></label><label>Brush <input id="brush" type="range" min="1" max="10" value="3"></label>
<button id="draw">Draw</button><button id="erase">Erase</button><button id="clear">Clear plane</button>
<button id="save">Save task</button><button id="finalize">Finalize all</button></div>
<div id="stage"><canvas id="image"></canvas><canvas id="overlay"></canvas></div>
</main><script>
let state=null,task=null,index=0,plane=0,mode='draw',mask=[];
const image=document.querySelector('#image'),overlay=document.querySelector('#overlay');
const ictx=image.getContext('2d'),octx=overlay.getContext('2d'),planeSelect=document.querySelector('#plane');
async function api(path,opts){const r=await fetch(path,opts);const j=await r.json();if(!r.ok)throw Error(j.error||r.statusText);return j}
async function refreshState(){state=await api('/api/state');document.querySelector('#status').textContent=`Annotator ${state.annotator_id}: ${state.completed}/${state.tasks} saved; ${state.remaining} remaining${state.finalized?' — FINALIZED':''}`;document.querySelector('#finalize').disabled=state.remaining!==0||state.finalized}
async function load(i){index=Math.max(0,Math.min(state.tasks-1,i));task=await api('/api/task?index='+index);mask=task.mask.slice();plane=0;planeSelect.innerHTML='';task.plane_labels.forEach((x,n)=>{let o=document.createElement('option');o.value=n;o.textContent=x;planeSelect.appendChild(o)});resize();render();await refreshState()}
function resize(){image.width=overlay.width=task.width;image.height=overlay.height=task.height}
function render(){const n=task.width*task.height,off=plane*n,px=ictx.createImageData(task.width,task.height),op=octx.createImageData(task.width,task.height);for(let i=0;i<n;i++){const v=task.features_u8[off+i];px.data[4*i]=v;px.data[4*i+1]=v;px.data[4*i+2]=v;px.data[4*i+3]=255;if(mask[off+i]){op.data[4*i]=255;op.data[4*i+1]=45;op.data[4*i+2]=20;op.data[4*i+3]=150}}ictx.putImageData(px,0,0);octx.putImageData(op,0,0);document.querySelector('#save').textContent=task.saved?'Update saved task':'Save task'}
function paint(ev){if(!task||state.finalized)return;const r=overlay.getBoundingClientRect(),x=Math.floor((ev.clientX-r.left)*task.width/r.width),y=Math.floor((ev.clientY-r.top)*task.height/r.height),b=Number(document.querySelector('#brush').value),n=task.width*task.height,off=plane*n;for(let dy=-b;dy<=b;dy++)for(let dx=-b;dx<=b;dx++)if(dx*dx+dy*dy<=b*b){const xx=x+dx,yy=y+dy;if(xx>=0&&yy>=0&&xx<task.width&&yy<task.height)mask[off+yy*task.width+xx]=mode==='draw'?1:0}render()}
let down=false;overlay.onpointerdown=e=>{down=true;overlay.setPointerCapture(e.pointerId);paint(e)};overlay.onpointermove=e=>{if(down)paint(e)};overlay.onpointerup=()=>down=false;
planeSelect.onchange=()=>{plane=Number(planeSelect.value);render()};document.querySelector('#draw').onclick=()=>mode='draw';document.querySelector('#erase').onclick=()=>mode='erase';
document.querySelector('#clear').onclick=()=>{const n=task.width*task.height;mask.fill(0,plane*n,(plane+1)*n);render()};
document.querySelector('#save').onclick=async()=>{await api('/api/task?index='+index,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mask})});task.saved=true;await refreshState();render()};
document.querySelector('#prev').onclick=()=>load(index-1);document.querySelector('#next').onclick=()=>load(index+1);
document.querySelector('#finalize').onclick=async()=>{if(confirm('Freeze all annotations? They cannot be edited afterward.')){await api('/api/finalize',{method:'POST'});await refreshState()}};
(async()=>{await refreshState();await load(0)})().catch(e=>document.querySelector('#status').textContent=e.message);
</script></body></html>"""


def serve_human_mask_annotation(
    task_manifest: str | Path,
    annotator_id: str,
    output_dir: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    protocol_version: str = "gravityspy_human_mask_blind_v1",
) -> None:
    """Serve one localhost-only blinded annotation session."""

    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("human annotation server may bind only to localhost")
    if not 1 <= port <= 65535:
        raise ValueError("human annotation server port is invalid")
    session = HumanMaskAnnotationSession(
        task_manifest, annotator_id, output_dir, protocol_version
    )

    class Handler(BaseHTTPRequestHandler):
        server_version = "GWYOLOBlindAnnotation/1"

        def _json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(value, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _error(self, exc: Exception) -> None:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    body = _ANNOTATION_HTML.encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header(
                        "Content-Security-Policy",
                        "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'",
                    )
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.end_headers()
                    self.wfile.write(body)
                elif parsed.path == "/api/state":
                    self._json(session.progress())
                elif parsed.path == "/api/task":
                    index = int(parse_qs(parsed.query).get("index", ["0"])[0])
                    self._json(session.task_payload(index))
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except Exception as exc:  # pragma: no cover - exercised through HTTP smoke
                self._error(exc)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 10_000_000:
                    raise ValueError("annotation request body has an invalid size")
                body = json.loads(self.rfile.read(length)) if length else {}
                if parsed.path == "/api/task":
                    if length == 0:
                        raise ValueError("annotation task save requires a JSON mask")
                    index = int(parse_qs(parsed.query).get("index", ["-1"])[0])
                    self._json(session.save_mask(index, body.get("mask")))
                elif parsed.path == "/api/finalize":
                    self._json(session.finalize())
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except Exception as exc:  # pragma: no cover - exercised through HTTP smoke
                self._error(exc)

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write(f"annotation-server {self.address_string()} {fmt % args}\n")

    server = ThreadingHTTPServer((host, port), Handler)
    print(
        json.dumps(
            {
                "status": "serving_blinded_human_mask_annotation",
                "host": host,
                "port": port,
                **session.progress(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    server.serve_forever()

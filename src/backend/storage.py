import os, json, threading
from typing import Any, Dict, List, Callable, Optional

# ==== konfigurasi hardcode (tanpa env) ====
MAX_HISTORY_LINES = 500  # batas riwayat agar file tidak membengkak

# ==== lokasi penyimpanan ====
_BASE_DIR = os.path.abspath(os.path.dirname(__file__))
_DATA_DIR = os.path.join(_BASE_DIR, "data")
_UPLOADS_DIR = os.path.join(_BASE_DIR, "uploads")
_HISTORY_PATH = os.path.join(_DATA_DIR, "history.jsonl")

_LOCK = threading.Lock()

def ensure_storage_ready():
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        os.makedirs(_UPLOADS_DIR, exist_ok=True)
        if not os.path.exists(_HISTORY_PATH):
            with open(_HISTORY_PATH, "w", encoding="utf-8"):
                pass
    except Exception:
        pass

def uploads_dir() -> str:
    ensure_storage_ready()
    return _UPLOADS_DIR

def history_path() -> str:
    ensure_storage_ready()
    return _HISTORY_PATH

def load_history() -> List[Dict[str, Any]]:
    ensure_storage_ready()
    out: List[Dict[str, Any]] = []
    p = history_path()
    try:
        with _LOCK:
            if not os.path.exists(p):
                return out
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        continue
        return out
    except Exception:
        return out

def _trim_history_file(max_lines: int = MAX_HISTORY_LINES):
    p = history_path()
    try:
        with _LOCK:
            if not os.path.exists(p):
                return
            with open(p, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) > max_lines:
                tail = lines[-max_lines:]
                with open(p, "w", encoding="utf-8") as f:
                    f.writelines(tail)
    except Exception:
        pass

def append_history(record: Dict[str, Any]) -> None:
    ensure_storage_ready()
    line = json.dumps(record, ensure_ascii=False)
    p = history_path()
    try:
        with _LOCK:
            with open(p, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        _trim_history_file()
    except Exception:
        pass

def update_history_record(record_id: str, updater: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """
    Update satu record (by id) dalam history.jsonl dengan fungsi updater().
    """
    ensure_storage_ready()
    p = history_path()
    try:
        with _LOCK:
            if not os.path.exists(p):
                return None
            with open(p, "r", encoding="utf-8") as f:
                lines = f.readlines()

            updated_lines: List[str] = []
            updated_record: Optional[Dict[str, Any]] = None

            for line in lines:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except Exception:
                    updated_lines.append(line)
                    continue
                if isinstance(obj, dict) and obj.get("id") == record_id:
                    new_obj = updater(dict(obj))
                    if new_obj is None:
                        updated_lines.append(json.dumps(obj, ensure_ascii=False) + "\n")
                    else:
                        updated_record = new_obj
                        updated_lines.append(json.dumps(new_obj, ensure_ascii=False) + "\n")
                else:
                    updated_lines.append(json.dumps(obj, ensure_ascii=False) + "\n")

            if updated_record is not None:
                with open(p, "w", encoding="utf-8") as f:
                    f.writelines(updated_lines)
                return updated_record
            else:
                return None
    except Exception:
        return None

# --------- Helpers untuk hapus file lokal ---------
def _delete_upload_file_if_exists(url: Optional[str]) -> None:
    """
    Hapus file di folder uploads jika url berupa path relatif '/uploads/xxx'.
    """
    if not url or not isinstance(url, str): return
    if not url.startswith("/uploads/"): return
    fn = url.split("/uploads/")[-1]
    if not fn: return
    path = os.path.join(uploads_dir(), fn)
    try:
        if os.path.isfile(path):
            os.remove(path)
    except Exception:
        pass

# --------- Delete satu record ---------
def delete_history_record(record_id: str) -> bool:
    """
    Hapus satu record by id dari history.jsonl, sekaligus hapus file upload & foto monitoring terkait.
    Return True jika ada yang terhapus.
    """
    ensure_storage_ready()
    p = history_path()
    try:
        with _LOCK:
            if not os.path.exists(p):
                return False
            with open(p, "r", encoding="utf-8") as f:
                lines = f.readlines()

            changed = False
            kept: List[str] = []

            for line in lines:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except Exception:
                    kept.append(line)
                    continue
                if isinstance(obj, dict) and obj.get("id") == record_id:
                    changed = True
                    # hapus file utama
                    _delete_upload_file_if_exists(obj.get("image_url"))
                    # hapus foto monitoring
                    for item in obj.get("items", []):
                        mon = item.get("monitoring") or {}
                        for url in mon.get("photos", []):
                            _delete_upload_file_if_exists(url)
                    # skip (tidak ditulis kembali)
                else:
                    kept.append(json.dumps(obj, ensure_ascii=False) + "\n")

            if changed:
                with open(p, "w", encoding="utf-8") as f:
                    f.writelines(kept)
            return changed
    except Exception:
        return False

# --------- Delete bulk berdasarkan filter ---------
def delete_history_filtered(project_id: Optional[str] = None, column_id: Optional[str] = None) -> int:
    """
    Hapus banyak record yang memenuhi filter project_id/column_id.
    Mengembalikan jumlah record yang dihapus.
    """
    ensure_storage_ready()
    p = history_path()
    try:
        with _LOCK:
            if not os.path.exists(p):
                return 0
            with open(p, "r", encoding="utf-8") as f:
                lines = f.readlines()

            kept: List[str] = []
            deleted_count = 0

            for line in lines:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except Exception:
                    kept.append(line)
                    continue

                match_proj = (project_id is None or obj.get("project_id") == project_id)
                match_col = (column_id is None or obj.get("column_id") == column_id)
                if match_proj and match_col and (project_id is not None or column_id is not None):
                    deleted_count += 1
                    _delete_upload_file_if_exists(obj.get("image_url"))
                    for item in obj.get("items", []):
                        mon = item.get("monitoring") or {}
                        for url in mon.get("photos", []):
                            _delete_upload_file_if_exists(url)
                    # skip menulis
                else:
                    kept.append(json.dumps(obj, ensure_ascii=False) + "\n")

            if deleted_count > 0:
                with open(p, "w", encoding="utf-8") as f:
                    f.writelines(kept)
            return deleted_count
    except Exception:
        return 0

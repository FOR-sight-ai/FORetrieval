from typing import Any, Dict, List, Optional
from datetime import datetime
from .models_metadata import MetadataFilter


def _parse_iso(s: str) -> Optional[datetime]:
    """Parse ISO format datetime string."""
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _any_in(a: List[str], b: List[str]) -> bool:
    """Check if any element in a is in b."""
    sa = {x.strip().lower() for x in a}
    sb = {x.strip().lower() for x in b}
    return len(sa & sb) > 0


def _value_match(meta: Dict[str, Any], f: MetadataFilter) -> bool:
    """Check if metadata matches filter criteria."""
    checks = []

    if f.language is not None:
        mv = (meta.get("language") or "").strip().lower()
        if isinstance(f.language, list):
            checks.append(mv in [x.strip().lower() for x in f.language])
        else:
            checks.append(mv == f.language.strip().lower())

    if f.ext is not None:
        mv = (meta.get("ext") or "").strip().lower()
        candidates = f.ext if isinstance(f.ext, list) else [f.ext]
        checks.append(mv in candidates)

    if f.document_type is not None:
        mv = (meta.get("document_type") or "").strip().lower()
        cands = (
            f.document_type if isinstance(f.document_type, list) else [f.document_type]
        )
        checks.append(mv in [x.strip().lower() for x in cands])

    if f.tags is not None:
        mv = [str(t).strip().lower() for t in (meta.get("tags") or [])]
        cands = f.tags if isinstance(f.tags, list) else [f.tags]
        checks.append(_any_in(mv, [str(x).strip().lower() for x in cands]))

    if f.mtime is not None:
        m = meta.get("mtime")
        mdt = _parse_iso(m) if isinstance(m, str) else None
        if mdt is None:
            checks.append(False)
        else:
            ok = True
            for op, rhs in f.mtime.items():
                rdt = _parse_iso(rhs)
                if rdt is None:
                    ok = False
                    break
                if op == ">=" and not (mdt >= rdt):
                    ok = False
                if op == "<=" and not (mdt <= rdt):
                    ok = False
                if op == ">" and not (mdt > rdt):
                    ok = False
                if op == "<" and not (mdt < rdt):
                    ok = False
                if op == "==" and not (mdt == rdt):
                    ok = False
                if not ok:
                    break
            checks.append(ok)

    for k, v in f.__dict__.items():
        if k in {"language", "ext", "tags", "document_type", "mtime", "logic"}:
            continue
        if v is None:
            continue
        mv = meta.get(k)
        if isinstance(v, list):
            checks.append(
                str(mv).strip().lower() in [str(x).strip().lower() for x in v]
            )
        else:
            checks.append(str(mv).strip().lower() == str(v).strip().lower())

    if not checks:
        return True
    return all(checks) if f.logic.upper() == "AND" else any(checks)

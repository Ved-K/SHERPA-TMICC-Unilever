from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple
from datetime import datetime, timezone

_OP_ABBR = {
    "normal operations": "NO",
    "normal ops": "NO",
    "abnormal operations": "AO",
    "abnormal ops": "AO",
    "emergency": "EM",
    "emergency operations": "EM",
    "emergency ops": "EM",
    "maintenance": "MA",
    "cleaning": "CL",
}

_PHASE_ABBR = {
    "startup": "ST",
    "start up": "ST",
    "shutdown": "SD",
    "shut down": "SD",
    "running": "RN",
    "run": "RN",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _table_columns(con: sqlite3.Connection, table: str) -> List[str]:
    cur = con.execute(f"PRAGMA table_info({table})")
    return [r[1] for r in cur.fetchall()]


def _ensure_column(con: sqlite3.Connection, table: str, col: str, col_def: str) -> None:
    cols = _table_columns(con, table)
    if col in cols:
        return
    con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def}")


def _next_code(con, table: str, prefix: str, width: int = 0, where: str = "", params: tuple = ()) -> str:
    q = f"SELECT code FROM {table} {where}"
    rows = con.execute(q, params).fetchall()

    best = 0
    pat = re.compile(rf"^{re.escape(prefix)}0*(\d+)$")

    for r in rows:
        code = (r["code"] or "").strip()
        m = pat.match(code)
        if m:
            best = max(best, int(m.group(1)))

    n = best + 1
    num = str(n).zfill(width) if width and width > 0 else str(n)
    return f"{prefix}{num}"

def _abbr_operation_category(cat: str) -> str:
    s = (cat or "").strip().lower()
    if not s:
        return "NO"
    if s in _OP_ABBR:
        return _OP_ABBR[s]
    # fallback: take first letters of words, max 2
    parts = re.findall(r"[a-z0-9]+", s)
    if not parts:
        return "OT"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[1][0]).upper()

def _abbr_status_from_phases(phases: list[str]) -> str:
    # If multiple phases selected, prefer RN > ST > SD (default RN)
    ph = [(p or "").strip().lower() for p in (phases or [])]
    ph_set = set(ph)

    if any(k in ph_set for k in ("running", "run")):
        return "RN"
    if any(k in ph_set for k in ("startup", "start up")):
        return "ST"
    if any(k in ph_set for k in ("shutdown", "shut down")):
        return "SD"
    return "RN"

def _next_task_seq_for_machine(con, machine_id: int) -> int:
    rows = con.execute("SELECT code FROM tasks WHERE machine_id=?", (int(machine_id),)).fetchall()

    best = 0

    # New style: L1-M1-T12-EM-RN  (grab 12)
    pat_new = re.compile(r"^L0*(\d+)-M0*(\d+)-T0*(\d+)\b", re.IGNORECASE)

    # Old style: T12 / T012 / T001 etc
    pat_old = re.compile(r"^T0*(\d+)$", re.IGNORECASE)

    for r in rows:
        code = (r["code"] or "").strip()

        m = pat_new.match(code)
        if m:
            best = max(best, int(m.group(3)))
            continue

        m = pat_old.match(code)
        if m:
            best = max(best, int(m.group(1)))
            continue

    return best + 1

def build_task_code(con, machine_id: int, operation_category: str, phases: list[str]) -> str:
    m = con.execute("SELECT id, line_id, code FROM machines WHERE id=?", (int(machine_id),)).fetchone()
    if not m:
        # fallback if something is wrong
        seq = _next_task_seq_for_machine(con, machine_id)
        return f"T{seq}"

    l = con.execute("SELECT code FROM lines WHERE id=?", (int(m["line_id"]),)).fetchone()
    line_code = (l["code"] if l and "code" in l.keys() else "L1").strip()
    mach_code = (m["code"] or "M1").strip()

    seq = _next_task_seq_for_machine(con, machine_id)
    cat = _abbr_operation_category(operation_category)
    status = _abbr_status_from_phases(phases)

    return f"{line_code}-{mach_code}-T{seq}-{cat}-{status}"

SCHEMA = """
CREATE TABLE IF NOT EXISTS lines (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  code TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS machines (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  line_id INTEGER NOT NULL,
  code TEXT NOT NULL,
  name TEXT NOT NULL,
  machine_type TEXT DEFAULT '',
  sort_index INTEGER DEFAULT 0,
  UNIQUE(line_id, code),
  FOREIGN KEY(line_id) REFERENCES lines(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS tasks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  machine_id INTEGER NOT NULL,
  code TEXT NOT NULL,
  name TEXT NOT NULL,
  operation_category TEXT NOT NULL,
  phases_json TEXT NOT NULL,
  sort_index INTEGER DEFAULT 0,
  UNIQUE(machine_id, code),
  FOREIGN KEY(machine_id) REFERENCES machines(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS steps (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  task_id INTEGER NOT NULL,
  step_no INTEGER NOT NULL,
  step_desc TEXT NOT NULL,
  hazard_text TEXT NOT NULL,
  eng_controls TEXT NOT NULL,
  admin_controls TEXT NOT NULL,
  probability_code REAL NOT NULL,
  severity_code REAL NOT NULL,
  sort_index INTEGER DEFAULT 0,
  UNIQUE(task_id, step_no),
  FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
);
"""


def connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, check_same_thread=False)
    con.row_factory = sqlite3.Row

    # 🔧 Add these 3 lines:
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA busy_timeout=5000;") 

    con.execute("PRAGMA foreign_keys=ON;")
    con.executescript(SCHEMA)

    # migrations / backfills
    for table, col, col_def in [
        ("machines", "sort_index", "INTEGER DEFAULT 0"),
        ("tasks", "sort_index", "INTEGER DEFAULT 0"),
        ("steps", "sort_index", "INTEGER DEFAULT 0"),
        ("machines", "machine_type", "TEXT DEFAULT ''"),
        ("tasks", "phases_json", "TEXT NOT NULL DEFAULT '[]'"),
        ("steps", "eng_controls", "TEXT NOT NULL DEFAULT ''"),
        ("steps", "admin_controls", "TEXT NOT NULL DEFAULT ''"),
        ("steps", "hazard_text", "TEXT NOT NULL DEFAULT ''"),
    ]:
        try:
            _ensure_column(con, table, col, col_def)
        except sqlite3.OperationalError:
            pass

    con.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_machines_line_sort ON machines(line_id, sort_index, id);
        CREATE INDEX IF NOT EXISTS idx_tasks_machine_sort ON tasks(machine_id, sort_index, id);
        CREATE INDEX IF NOT EXISTS idx_steps_task_sort ON steps(task_id, step_no, sort_index, id);
        """
    )

    con.commit()
    return con


# -------------------------
# Lines
# -------------------------
def list_lines(con: sqlite3.Connection) -> List[Dict[str, Any]]:
    cur = con.execute("SELECT id, code, name, created_at FROM lines ORDER BY id")
    return [dict(id=r[0], code=r[1], name=r[2], created_at=r[3]) for r in cur.fetchall()]


def get_line(con: sqlite3.Connection, line_id: int) -> Optional[Dict[str, Any]]:
    cur = con.execute("SELECT id, code, name, created_at FROM lines WHERE id=?", (line_id,))
    r = cur.fetchone()
    if not r:
        return None
    return dict(id=r[0], code=r[1], name=r[2], created_at=r[3])


def create_line(con: sqlite3.Connection, name: str) -> int:
    created_at = now_iso()
    code = _next_code(con, "lines", "L", 0)
    cur = con.execute(
        "INSERT INTO lines(created_at, code, name) VALUES(?,?,?)",
        (created_at, code, name.strip()),
    )
    con.commit()
    return int(cur.lastrowid)


# -------------------------
# Machines
# -------------------------
def list_machines(con: sqlite3.Connection, line_id: int) -> List[Dict[str, Any]]:
    cur = con.execute(
        "SELECT id, line_id, code, name, machine_type, sort_index, created_at "
        "FROM machines WHERE line_id=? ORDER BY sort_index, id",
        (line_id,),
    )
    return [
        dict(
            id=r[0],
            line_id=r[1],
            code=r[2],
            name=r[3],
            machine_type=r[4],
            sort_index=r[5],
            created_at=r[6],
        )
        for r in cur.fetchall()
    ]


def get_machine(con: sqlite3.Connection, machine_id: int) -> Optional[Dict[str, Any]]:
    cur = con.execute(
        "SELECT id, line_id, code, name, machine_type, sort_index, created_at FROM machines WHERE id=?",
        (machine_id,),
    )
    r = cur.fetchone()
    if not r:
        return None
    return dict(
        id=r[0],
        line_id=r[1],
        code=r[2],
        name=r[3],
        machine_type=r[4],
        sort_index=r[5],
        created_at=r[6],
    )


def create_machine(con: sqlite3.Connection, line_id: int, name: str, machine_type: str = "", sort_index: int = 0) -> int:
    created_at = now_iso()
    code = _next_code(con, "machines", "M", 0, "WHERE line_id=?", (line_id,))
    cur = con.execute(
        "INSERT INTO machines(created_at, line_id, code, name, machine_type, sort_index) VALUES(?,?,?,?,?,?)",
        (created_at, line_id, code, name.strip(), (machine_type or "").strip(), int(sort_index)),
    )
    con.commit()
    return int(cur.lastrowid)


def bulk_add_machines(con: sqlite3.Connection, line_id: int, names: List[str]) -> Tuple[int, int]:
    existing = {m["name"].strip().lower() for m in list_machines(con, line_id)}
    added = 0
    skipped = 0
    for nm in [n.strip() for n in names if n.strip()]:
        if nm.lower() in existing:
            skipped += 1
            continue
        create_machine(con, line_id, nm)
        existing.add(nm.lower())
        added += 1
    return added, skipped


# -------------------------
# Tasks
# -------------------------
def list_tasks(con: sqlite3.Connection, machine_id: int) -> List[Dict[str, Any]]:
    cur = con.execute(
        "SELECT id, machine_id, code, name, operation_category, phases_json, sort_index, created_at "
        "FROM tasks WHERE machine_id=? ORDER BY sort_index, id",
        (machine_id,),
    )
    out = []
    for r in cur.fetchall():
        try:
            phases = json.loads(r[5]) if r[5] else []
        except Exception:
            phases = []
        out.append(
            dict(
                id=r[0],
                machine_id=r[1],
                code=r[2],
                name=r[3],
                operation_category=r[4],
                phases=phases,
                sort_index=r[6],
                created_at=r[7],
            )
        )
    return out


def get_task(con: sqlite3.Connection, task_id: int) -> Optional[Dict[str, Any]]:
    cur = con.execute(
        "SELECT id, machine_id, code, name, operation_category, phases_json, sort_index, created_at "
        "FROM tasks WHERE id=?",
        (task_id,),
    )
    r = cur.fetchone()
    if not r:
        return None
    try:
        phases = json.loads(r[5]) if r[5] else []
    except Exception:
        phases = []
    return dict(
        id=r[0],
        machine_id=r[1],
        code=r[2],
        name=r[3],
        operation_category=r[4],
        phases=phases,
        sort_index=r[6],
        created_at=r[7],
    )


def create_task(
    con: sqlite3.Connection,
    machine_id: int,
    name: str,
    operation_category: str,
    phases: List[str],
    sort_index: int = 0,
) -> int:
    created_at = now_iso()
    code = build_task_code(con, machine_id, operation_category, phases)
    phases_json = json.dumps(phases, ensure_ascii=False)
    cur = con.execute(
        "INSERT INTO tasks(created_at, machine_id, code, name, operation_category, phases_json, sort_index) "
        "VALUES(?,?,?,?,?,?,?)",
        (created_at, machine_id, code, name.strip(), operation_category.strip(), phases_json, int(sort_index)),
    )
    con.commit()
    return int(cur.lastrowid)


def update_task(con: sqlite3.Connection, task_id: int, name: str, operation_category: str, phases: List[str]) -> None:
    phases_json = json.dumps(phases, ensure_ascii=False)
    con.execute(
        "UPDATE tasks SET name=?, operation_category=?, phases_json=? WHERE id=?",
        (name.strip(), operation_category.strip(), phases_json, int(task_id)),
    )
    con.commit()


# -------------------------
# Steps
# -------------------------
def add_step(
    con: sqlite3.Connection,
    task_id: int,
    step_no: int,
    step_desc: str,
    hazard_text: str,
    eng_controls: str,
    admin_controls: str,
    probability_code: float,
    severity_code: float,
    sort_index: int = 0,
) -> int:
    created_at = now_iso()
    cur = con.execute(
        "INSERT INTO steps(created_at, task_id, step_no, step_desc, hazard_text, eng_controls, admin_controls, "
        "probability_code, severity_code, sort_index) VALUES(?,?,?,?,?,?,?,?,?,?)",
        (
            created_at,
            task_id,
            int(step_no),
            step_desc.strip(),
            hazard_text.strip(),
            eng_controls.strip(),
            admin_controls.strip(),
            float(probability_code),
            float(severity_code),
            int(sort_index),
        ),
    )
    con.commit()
    return int(cur.lastrowid)


def list_steps(con: sqlite3.Connection, task_id: int) -> List[Dict[str, Any]]:
    cur = con.execute(
        "SELECT id, task_id, step_no, step_desc, hazard_text, eng_controls, admin_controls, probability_code, severity_code, created_at "
        "FROM steps WHERE task_id=? ORDER BY step_no ASC",
        (task_id,),
    )
    return [
        dict(
            id=r[0],
            task_id=r[1],
            step_no=r[2],
            step_desc=r[3],
            hazard_text=r[4],
            eng_controls=r[5],
            admin_controls=r[6],
            probability_code=r[7],
            severity_code=r[8],
            created_at=r[9],
        )
        for r in cur.fetchall()
    ]


def get_step(con: sqlite3.Connection, task_id: int, step_no: int) -> Optional[Dict[str, Any]]:
    cur = con.execute(
        "SELECT id, task_id, step_no, step_desc, hazard_text, eng_controls, admin_controls, probability_code, severity_code, created_at "
        "FROM steps WHERE task_id=? AND step_no=?",
        (task_id, step_no),
    )
    r = cur.fetchone()
    if not r:
        return None
    return dict(
        id=r[0],
        task_id=r[1],
        step_no=r[2],
        step_desc=r[3],
        hazard_text=r[4],
        eng_controls=r[5],
        admin_controls=r[6],
        probability_code=r[7],
        severity_code=r[8],
        created_at=r[9],
    )


def update_step(
    con: sqlite3.Connection,
    *,
    step_id: int | None = None,
    task_id: int | None = None,
    step_no: int | None = None,
    step_desc: str = "",
    hazard_text: str = "",
    eng_controls: str = "",
    admin_controls: str = "",
    probability_code: float = 0.0,
    severity_code: float = 0.0,
) -> None:
    # allow updating either by step_id OR by (task_id, step_no)
    if step_id is None:
        if task_id is None or step_no is None:
            raise ValueError("update_step requires either step_id OR (task_id and step_no).")

        con.execute(
            """
            UPDATE steps
            SET step_desc=?,
                hazard_text=?,
                eng_controls=?,
                admin_controls=?,
                probability_code=?,
                severity_code=?
            WHERE task_id=? AND step_no=?
            """,
            (
                (step_desc or "").strip(),
                (hazard_text or "").strip(),
                (eng_controls or "").strip(),
                (admin_controls or "").strip(),
                float(probability_code),
                float(severity_code),
                int(task_id),
                int(step_no),
            ),
        )
    else:
        con.execute(
            """
            UPDATE steps
            SET step_desc=?,
                hazard_text=?,
                eng_controls=?,
                admin_controls=?,
                probability_code=?,
                severity_code=?
            WHERE id=?
            """,
            (
                (step_desc or "").strip(),
                (hazard_text or "").strip(),
                (eng_controls or "").strip(),
                (admin_controls or "").strip(),
                float(probability_code),
                float(severity_code),
                int(step_id),
            ),
        )

    con.commit()


def max_step_no(con: sqlite3.Connection, task_id: int) -> int:
    cur = con.execute("SELECT COALESCE(MAX(step_no), 0) FROM steps WHERE task_id=?", (task_id,))
    return int(cur.fetchone()[0] or 0)


def delete_step(con: sqlite3.Connection, task_id: int, step_no: int) -> None:
    con.execute("DELETE FROM steps WHERE task_id=? AND step_no=?", (int(task_id), int(step_no)))
    con.commit()


def renumber_steps(con: sqlite3.Connection, task_id: int) -> None:
    steps = list_steps(con, int(task_id))
    con.execute("BEGIN")
    try:
        # temp offset to avoid UNIQUE(task_id, step_no) collisions
        con.execute("UPDATE steps SET step_no = step_no + 10000 WHERE task_id=?", (int(task_id),))
        for i, s in enumerate(steps, start=1):
            con.execute(
                "UPDATE steps SET step_no=? WHERE task_id=? AND id=?",
                (int(i), int(task_id), int(s["id"])),
            )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise


def swap_steps(con: sqlite3.Connection, task_id: int, a: int, b: int) -> None:
    if a == b:
        return
    con.execute("BEGIN")
    try:
        con.execute("UPDATE steps SET step_no=-1 WHERE task_id=? AND step_no=?", (int(task_id), int(a)))
        con.execute("UPDATE steps SET step_no=? WHERE task_id=? AND step_no=?", (int(a), int(task_id), int(b)))
        con.execute("UPDATE steps SET step_no=? WHERE task_id=? AND step_no=-1", (int(b), int(task_id)))
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise


def duplicate_step(con: sqlite3.Connection, task_id: int, step_no: int) -> int:
    src = get_step(con, int(task_id), int(step_no))
    if not src:
        raise ValueError("Step not found.")
    new_no = max_step_no(con, int(task_id)) + 1
    return add_step(
        con,
        task_id=int(task_id),
        step_no=new_no,
        step_desc=src["step_desc"],
        hazard_text=src["hazard_text"],
        eng_controls=src["eng_controls"],
        admin_controls=src["admin_controls"],
        probability_code=float(src["probability_code"]),
        severity_code=float(src["severity_code"]),
    )


# -------------------------
# DB utilities (optional but very useful for debugging)
# -------------------------
def db_counts(con: sqlite3.Connection) -> Dict[str, int]:
    def _count(tbl: str) -> int:
        return int(con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0] or 0)

    return {
        "lines": _count("lines"),
        "machines": _count("machines"),
        "tasks": _count("tasks"),
        "steps": _count("steps"),
    }


def task_counts_by_machine(con: sqlite3.Connection, line_id: Optional[int] = None) -> List[Dict[str, Any]]:
    where = ""
    params: List[Any] = []
    if line_id is not None:
        where = "WHERE m.line_id = ?"
        params.append(int(line_id))

    cur = con.execute(
        f"""
        SELECT
          m.id, m.code, m.name,
          COALESCE(t.cnt, 0) AS task_count
        FROM machines m
        LEFT JOIN (
          SELECT machine_id, COUNT(*) AS cnt
          FROM tasks
          GROUP BY machine_id
        ) t ON t.machine_id = m.id
        {where}
        ORDER BY task_count DESC, m.id ASC
        """,
        tuple(params),
    )
    return [{"machine_id": r[0], "code": r[1], "name": r[2], "task_count": r[3]} for r in cur.fetchall()]


# -------------------------
# Tasks (FIXED)
# -------------------------
def list_tasks_for_machine(con: sqlite3.Connection, machine_id: int) -> List[Dict[str, Any]]:
    """
    Returns list[dict] with step_count.
    Uses tasks.phases_json (your real schema), not a non-existent tasks.phases column.
    """
    cur = con.execute(
        """
        SELECT
          t.id, t.machine_id, t.code, t.name, t.operation_category,
          t.phases_json, t.sort_index, t.created_at,
          COALESCE(s.cnt, 0) AS step_count
        FROM tasks t
        LEFT JOIN (
          SELECT task_id, COUNT(*) AS cnt
          FROM steps
          GROUP BY task_id
        ) s ON s.task_id = t.id
        WHERE t.machine_id = ?
        ORDER BY t.sort_index, t.id
        """,
        (int(machine_id),),
    )

    out: List[Dict[str, Any]] = []
    for r in cur.fetchall():
        try:
            phases = json.loads(r[5]) if r[5] else []
        except Exception:
            phases = []
        out.append(
            dict(
                id=r[0],
                machine_id=r[1],
                code=r[2],
                name=r[3],
                operation_category=r[4],
                phases=phases,
                sort_index=r[6],
                created_at=r[7],
                step_count=r[8],
            )
        )
    return out


# -------------------------
# Steps (optional convenience)
# -------------------------
def list_steps_for_task(con: sqlite3.Connection, task_id: int) -> List[Dict[str, Any]]:
    cur = con.execute(
        """
        SELECT
          id, task_id, step_no, step_desc,
          hazard_text, eng_controls, admin_controls,
          probability_code, severity_code, created_at
        FROM steps
        WHERE task_id = ?
        ORDER BY step_no ASC
        """,
        (int(task_id),),
    )
    return [
        dict(
            id=r[0],
            task_id=r[1],
            step_no=r[2],
            step_desc=r[3],
            hazard_text=r[4],
            eng_controls=r[5],
            admin_controls=r[6],
            probability_code=r[7],
            severity_code=r[8],
            created_at=r[9],
        )
        for r in cur.fetchall()
    ]

def delete_task(con: sqlite3.Connection, task_id: int) -> None:
    con.execute("DELETE FROM tasks WHERE id=?", (int(task_id),))
    con.commit()

def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

# -----------------------
# LINES
# -----------------------
def ensure_line(con, name: str, code: str = None) -> int:
    name = (name or "").strip()
    code = (code or "").strip()
    existing = con.execute("SELECT id, name, code FROM lines WHERE code=? OR name=?", (code, name)).fetchone()

    if existing:
        updates = []
        vals = []
        if name and existing["name"] != name:
            updates.append("name=?"); vals.append(name)
        if code and existing["code"] != code:
            updates.append("code=?"); vals.append(code)
        if updates:
            vals.append(existing["id"])
            con.execute(f"UPDATE lines SET {', '.join(updates)} WHERE id=?", vals)
        return existing["id"]

    created_at = now_iso()
    if not code:
        code = f"L{int(datetime.now().timestamp())%10000:04d}"
    cur = con.execute(
        "INSERT INTO lines(created_at, code, name) VALUES(?,?,?)",
        (created_at, code, name),
    )
    return cur.lastrowid


# -----------------------
# MACHINES
# -----------------------
def ensure_machine(con, line_id: int, code: str, name: str = None) -> int:
    code = (code or "").strip()
    name = (name or "").strip()

    existing = con.execute(
        "SELECT id, code, name FROM machines WHERE line_id=? AND (code=? OR name=?)",
        (line_id, code, name),
    ).fetchone()

    if existing:
        updates = []
        vals = []
        if name and existing["name"] != name:
            updates.append("name=?"); vals.append(name)
        if code and existing["code"] != code:
            updates.append("code=?"); vals.append(code)
        if updates:
            vals.append(existing["id"])
            con.execute(f"UPDATE machines SET {', '.join(updates)} WHERE id=?", vals)
        return existing["id"]

    created_at = now_iso()
    if not code:
        code = f"M{int(datetime.now().timestamp())%10000:04d}"
    cur = con.execute(
        "INSERT INTO machines(created_at, line_id, code, name) VALUES(?,?,?,?)",
        (created_at, line_id, code, name or code),
    )
    return cur.lastrowid


# -----------------------
# TASKS
# -----------------------
def ensure_task(con, machine_id: int, code: str, name: str, category: str, phases: list[str]) -> int:
    code = (code or "").strip()
    name = (name or "").strip()
    category = (category or "").strip()
    phases_json = json.dumps(phases or [], ensure_ascii=False)

    existing = con.execute(
        "SELECT id, name, operation_category, phases_json FROM tasks WHERE machine_id=? AND code=?",
        (machine_id, code),
    ).fetchone()

    if existing:
        updates = []
        vals = []
        if name and existing["name"] != name:
            updates.append("name=?"); vals.append(name)
        if category and existing["operation_category"] != category:
            updates.append("operation_category=?"); vals.append(category)
        if existing["phases_json"] != phases_json:
            updates.append("phases_json=?"); vals.append(phases_json)
        if updates:
            vals.append(existing["id"])
            con.execute(f"UPDATE tasks SET {', '.join(updates)} WHERE id=?", vals)
        return existing["id"]

    created_at = now_iso()
    if not code:
        code = f"T{int(datetime.now().timestamp())%10000:04d}"
    cur = con.execute(
        """
        INSERT INTO tasks(created_at, machine_id, code, name, operation_category, phases_json)
        VALUES(?,?,?,?,?,?)
        """,
        (created_at, machine_id, code, name or code, category, phases_json),
    )
    return cur.lastrowid


# -----------------------
# STEPS
# -----------------------
def ensure_step(con, task_id: int, step_no: int, desc: str, hazard: str,
                eng_controls: str, admin_controls: str, p, s) -> int:
    step_no = int(step_no or 1)
    p = p if p not in (None, "") else "0"
    s = s if s not in (None, "") else "0"
    created_at = now_iso()

    existing = con.execute(
        "SELECT id FROM steps WHERE task_id=? AND step_no=?",
        (int(task_id), int(step_no)),
    ).fetchone()

    if existing:
        con.execute(
            """
            UPDATE steps
            SET step_desc=?, hazard_text=?, eng_controls=?, admin_controls=?,
                probability_code=?, severity_code=?
            WHERE id=?
            """,
            (desc, hazard, eng_controls, admin_controls, p, s, existing["id"]),
        )
        return existing["id"]

    cur = con.execute(
        """
        INSERT INTO steps(
            created_at, task_id, step_no, step_desc, hazard_text,
            eng_controls, admin_controls, probability_code, severity_code
        )
        VALUES(?,?,?,?,?,?,?,?,?)
        """,
        (created_at, int(task_id), step_no, desc, hazard, eng_controls, admin_controls, p, s),
    )
    return cur.lastrowid


# -----------------------
# UPDATE HELPERS
# -----------------------
def update_machine(con, machine_id, **changes):
    if not changes:
        return
    cols = ", ".join(f"{k}=?" for k in changes.keys())
    values = list(changes.values()) + [machine_id]
    con.execute(f"UPDATE machines SET {cols} WHERE id=?", values)


def update_task(con, task_id, **changes):
    if not changes:
        return
    if "phases" in changes:
        changes["phases_json"] = json.dumps(changes.pop("phases"), ensure_ascii=False)
    cols = ", ".join(f"{k}=?" for k in changes.keys())
    values = list(changes.values()) + [task_id]
    con.execute(f"UPDATE tasks SET {cols} WHERE id=?", values)

def clear_database(con: sqlite3.Connection) -> None:
    """
    Clears *core* data tables (Lines, Machines, Tasks, Steps).
    Keeps schema + any other reference tables intact.
    """
    # Order matters if FK constraints exist and CASCADE isn't set everywhere
    tables = ["steps", "tasks", "machines", "lines"]

    con.execute("PRAGMA foreign_keys = OFF;")
    try:
        for t in tables:
            con.execute(f"DELETE FROM {t};")

        # Reset AUTOINCREMENT counters (optional but nice)
        con.execute(
            "DELETE FROM sqlite_sequence WHERE name IN ('steps','tasks','machines','lines');"
        )

        con.commit()
    finally:
        con.execute("PRAGMA foreign_keys = ON;")

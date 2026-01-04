from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import streamlit as st

import pyodbc

# ---------------------------------------------------------------------
# Azure SQL (SQL Server) migration of your original sqlite-based store.py
# Keeps the same table names, column names, and return shapes.
# ---------------------------------------------------------------------

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
    """ISO8601 with timezone offset, seconds precision (matches your prior behaviour)."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _exec(con, sql, params=()):
    try:
        cur = con.cursor()
        cur.execute(sql, params)
        return cur
    except Exception as e:
        print(f"SQL failed: {sql}\nError: {e}")
        return None


def _fetchone_dict(cur: pyodbc.Cursor) -> Optional[Dict[str, Any]]:
    row = cur.fetchone()
    if not row:
        return None
    cols = [c[0] for c in cur.description]
    return dict(zip(cols, row))


def _fetchall_dict(cur):
    if cur is None:
        return []
    rows = cur.fetchall()
    columns = [col[0] for col in cur.description]
    return [dict(zip(columns, row)) for row in rows]


# ---------------------------------------------------------------------
# Schema (dbo.lines / machines / tasks / steps)
# ---------------------------------------------------------------------

_SCHEMA_SQL = r"""
/* lines */
IF OBJECT_ID(N'dbo.lines', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.lines (
    id         INT IDENTITY(1,1) NOT NULL PRIMARY KEY,
    created_at NVARCHAR(35) NOT NULL,
    code       NVARCHAR(50) NOT NULL UNIQUE,
    name       NVARCHAR(255) NOT NULL
  );
END;

/* machines */
IF OBJECT_ID(N'dbo.machines', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.machines (
    id           INT IDENTITY(1,1) NOT NULL PRIMARY KEY,
    created_at   NVARCHAR(35) NOT NULL,
    line_id      INT NOT NULL,
    code         NVARCHAR(50) NOT NULL,
    name         NVARCHAR(255) NOT NULL,
    machine_type NVARCHAR(255) NOT NULL CONSTRAINT DF_machines_machine_type DEFAULT(''),
    sort_index   INT NOT NULL CONSTRAINT DF_machines_sort_index DEFAULT(0),
    CONSTRAINT UQ_machines_line_code UNIQUE (line_id, code),
    CONSTRAINT FK_machines_lines FOREIGN KEY (line_id) REFERENCES dbo.lines(id) ON DELETE CASCADE
  );
END;

/* tasks */
IF OBJECT_ID(N'dbo.tasks', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.tasks (
    id                 INT IDENTITY(1,1) NOT NULL PRIMARY KEY,
    created_at         NVARCHAR(35) NOT NULL,
    machine_id         INT NOT NULL,
    code               NVARCHAR(100) NOT NULL,
    name               NVARCHAR(255) NOT NULL,
    operation_category NVARCHAR(255) NOT NULL,
    phases_json        NVARCHAR(MAX) NOT NULL CONSTRAINT DF_tasks_phases_json DEFAULT('[]'),
    sort_index         INT NOT NULL CONSTRAINT DF_tasks_sort_index DEFAULT(0),
    CONSTRAINT UQ_tasks_machine_code UNIQUE (machine_id, code),
    CONSTRAINT FK_tasks_machines FOREIGN KEY (machine_id) REFERENCES dbo.machines(id) ON DELETE CASCADE
  );
END;

/* steps */
IF OBJECT_ID(N'dbo.steps', N'U') IS NULL
BEGIN
  CREATE TABLE dbo.steps (
    id               INT IDENTITY(1,1) NOT NULL PRIMARY KEY,
    created_at       NVARCHAR(35) NOT NULL,
    task_id          INT NOT NULL,
    step_no          INT NOT NULL,
    step_desc        NVARCHAR(MAX) NOT NULL,
    hazard_text      NVARCHAR(MAX) NOT NULL CONSTRAINT DF_steps_hazard_text DEFAULT(''),
    eng_controls     NVARCHAR(MAX) NOT NULL CONSTRAINT DF_steps_eng_controls DEFAULT(''),
    admin_controls   NVARCHAR(MAX) NOT NULL CONSTRAINT DF_steps_admin_controls DEFAULT(''),
    probability_code FLOAT NOT NULL,
    severity_code    FLOAT NOT NULL,
    sort_index       INT NOT NULL CONSTRAINT DF_steps_sort_index DEFAULT(0),
    CONSTRAINT UQ_steps_task_stepno UNIQUE (task_id, step_no),
    CONSTRAINT FK_steps_tasks FOREIGN KEY (task_id) REFERENCES dbo.tasks(id) ON DELETE CASCADE
  );
END;

/* indexes */
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'idx_machines_line_sort' AND object_id = OBJECT_ID('dbo.machines'))
  CREATE INDEX idx_machines_line_sort ON dbo.machines(line_id, sort_index, id);

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'idx_tasks_machine_sort' AND object_id = OBJECT_ID('dbo.tasks'))
  CREATE INDEX idx_tasks_machine_sort ON dbo.tasks(machine_id, sort_index, id);

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'idx_steps_task_sort' AND object_id = OBJECT_ID('dbo.steps'))
  CREATE INDEX idx_steps_task_sort ON dbo.steps(task_id, step_no, sort_index, id);
"""


def _column_exists(con: pyodbc.Connection, table: str, col: str) -> bool:
    row = _fetchone(
        con,
        """
        SELECT 1
        FROM sys.columns
        WHERE object_id = OBJECT_ID(?, 'U') AND name = ?
        """,
        (f"dbo.{table}", col),
    )
    return row is not None


def _fetchone(con: pyodbc.Connection, sql: str, params: Tuple[Any, ...] = ()):
    cur = con.cursor()
    try:
        cur.execute(sql, params)
        return cur.fetchone()
    finally:
        cur.close()


def _ensure_column(con: pyodbc.Connection, table: str, col: str, col_def: str) -> None:
    if _column_exists(con, table, col):
        return
    _exec(con, f"ALTER TABLE dbo.{table} ADD {col} {col_def}")


def _init_schema(con: pyodbc.Connection) -> None:
    """
    Create tables/indexes if they don't exist + apply 'migrations' similar to your sqlite version.
    Safe to call repeatedly.
    """
    _exec(con, _SCHEMA_SQL)

    # Backfill/ensure columns that were added via sqlite migrations in your original file.
    # On a brand-new Azure SQL DB, these are already included in CREATE TABLE above.
    migrations = [
        (
            "machines",
            "sort_index",
            "INT NOT NULL CONSTRAINT DF_machines_sort_index2 DEFAULT(0)",
        ),
        (
            "tasks",
            "sort_index",
            "INT NOT NULL CONSTRAINT DF_tasks_sort_index2 DEFAULT(0)",
        ),
        (
            "steps",
            "sort_index",
            "INT NOT NULL CONSTRAINT DF_steps_sort_index2 DEFAULT(0)",
        ),
        (
            "machines",
            "machine_type",
            "NVARCHAR(255) NOT NULL CONSTRAINT DF_machines_machine_type2 DEFAULT('')",
        ),
        (
            "tasks",
            "phases_json",
            "NVARCHAR(MAX) NOT NULL CONSTRAINT DF_tasks_phases_json2 DEFAULT('[]')",
        ),
        (
            "steps",
            "eng_controls",
            "NVARCHAR(MAX) NOT NULL CONSTRAINT DF_steps_eng_controls2 DEFAULT('')",
        ),
        (
            "steps",
            "admin_controls",
            "NVARCHAR(MAX) NOT NULL CONSTRAINT DF_steps_admin_controls2 DEFAULT('')",
        ),
        (
            "steps",
            "hazard_text",
            "NVARCHAR(MAX) NOT NULL CONSTRAINT DF_steps_hazard_text2 DEFAULT('')",
        ),
    ]
    for table, col, col_def in migrations:
        try:
            _ensure_column(con, table, col, col_def)
        except pyodbc.Error:
            # e.g. default constraint name collision; ignore to keep init idempotent.
            pass

    con.commit()


def get_connection(*, init_schema: bool = True) -> pyodbc.Connection:
    conn_str = st.secrets["azure_sql"]["connection_string"]

    con = pyodbc.connect(conn_str)
    con.autocommit = False

    if init_schema:
        _init_schema(con)

    return con


def connect(conn_str: str, *, init_schema: bool = True) -> pyodbc.Connection:
    """
    Connect to Azure SQL Server using a full ODBC connection string.

    Example:
      DRIVER={ODBC Driver 18 for SQL Server};
      SERVER=tcp:myserver.database.windows.net,1433;
      DATABASE=mydb;
      UID=myuser;
      PWD=mypassword;
      Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;
    """
    con = pyodbc.connect(conn_str)
    con.autocommit = False
    if init_schema:
        _init_schema(con)
    return con


# -------------------------
# Code-generation helpers
# -------------------------
def _next_code(
    con: pyodbc.Connection,
    table: str,
    prefix: str,
    width: int = 0,
    where: str = "",
    params: Tuple[Any, ...] = (),
) -> str:
    q = f"SELECT code FROM dbo.{table} {where}"
    cur = _exec(con, q, params)
    rows = cur.fetchall()

    best = 0
    pat = re.compile(rf"^{re.escape(prefix)}0*(\d+)$")

    for r in rows:
        code = (r[0] or "").strip()
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
    parts = re.findall(r"[a-z0-9]+", s)
    if not parts:
        return "OT"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[1][0]).upper()


def _abbr_status_from_phases(phases: list[str]) -> str:
    ph = [(p or "").strip().lower() for p in (phases or [])]
    ph_set = set(ph)

    if any(k in ph_set for k in ("running", "run")):
        return "RN"
    if any(k in ph_set for k in ("startup", "start up")):
        return "ST"
    if any(k in ph_set for k in ("shutdown", "shut down")):
        return "SD"
    return "RN"


def _next_task_seq_for_machine(con: pyodbc.Connection, machine_id: int) -> int:
    cur = _exec(
        con, "SELECT code FROM dbo.tasks WHERE machine_id=?", (int(machine_id),)
    )
    rows = cur.fetchall()

    best = 0
    pat_new = re.compile(r"^L0*(\d+)-M0*(\d+)-T0*(\d+)\b", re.IGNORECASE)
    pat_old = re.compile(r"^T0*(\d+)$", re.IGNORECASE)

    for r in rows:
        code = (r[0] or "").strip()

        m = pat_new.match(code)
        if m:
            best = max(best, int(m.group(3)))
            continue

        m = pat_old.match(code)
        if m:
            best = max(best, int(m.group(1)))
            continue

    return best + 1


def build_task_code(
    con: pyodbc.Connection, machine_id: int, operation_category: str, phases: list[str]
) -> str:
    cur = _exec(
        con, "SELECT id, line_id, code FROM dbo.machines WHERE id=?", (int(machine_id),)
    )
    m = cur.fetchone()
    if not m:
        seq = _next_task_seq_for_machine(con, machine_id)
        return f"T{seq}"

    line_id = int(m[1])

    cur2 = _exec(con, "SELECT code FROM dbo.lines WHERE id=?", (line_id,))
    l = cur2.fetchone()
    line_code = (l[0] if l and l[0] else "L1").strip()

    mach_code = (m[2] or "M1").strip()

    seq = _next_task_seq_for_machine(con, machine_id)
    cat = _abbr_operation_category(operation_category)
    status = _abbr_status_from_phases(phases)

    return f"{line_code}-{mach_code}-T{seq}-{cat}-{status}"


# -------------------------
# Lines
# -------------------------
def list_lines(con: pyodbc.Connection) -> List[Dict[str, Any]]:
    cur = _exec(con, "SELECT id, code, name, created_at FROM dbo.lines ORDER BY id")
    return _fetchall_dict(cur)


def get_line(con: pyodbc.Connection, line_id: int) -> Optional[Dict[str, Any]]:
    cur = _exec(
        con,
        "SELECT id, code, name, created_at FROM dbo.lines WHERE id=?",
        (int(line_id),),
    )
    return _fetchone_dict(cur)


def create_line(con: pyodbc.Connection, name: str) -> int:
    created_at = now_iso()
    code = _next_code(con, "lines", "L", 0)

    cur = _exec(
        con,
        "INSERT INTO dbo.lines(created_at, code, name) OUTPUT INSERTED.id VALUES(?,?,?)",
        (created_at, code, (name or "").strip()),
    )
    new_id = int(cur.fetchone()[0])
    con.commit()
    return new_id


# -------------------------
# Machines
# -------------------------
def list_machines(con: pyodbc.Connection, line_id: int) -> List[Dict[str, Any]]:
    cur = _exec(
        con,
        """
        SELECT id, line_id, code, name, machine_type, sort_index, created_at
        FROM dbo.machines
        WHERE line_id=?
        ORDER BY sort_index, id
        """,
        (int(line_id),),
    )
    return _fetchall_dict(cur)


def get_machine(con: pyodbc.Connection, machine_id: int) -> Optional[Dict[str, Any]]:
    cur = _exec(
        con,
        """
        SELECT id, line_id, code, name, machine_type, sort_index, created_at
        FROM dbo.machines
        WHERE id=?
        """,
        (int(machine_id),),
    )
    return _fetchone_dict(cur)


def create_machine(
    con: pyodbc.Connection,
    line_id: int,
    name: str,
    machine_type: str = "",
    sort_index: int = 0,
) -> int:
    created_at = now_iso()
    code = _next_code(con, "machines", "M", 0, "WHERE line_id=?", (int(line_id),))

    cur = _exec(
        con,
        """
        INSERT INTO dbo.machines(created_at, line_id, code, name, machine_type, sort_index)
        OUTPUT INSERTED.id
        VALUES(?,?,?,?,?,?)
        """,
        (
            created_at,
            int(line_id),
            code,
            (name or "").strip(),
            (machine_type or "").strip(),
            int(sort_index),
        ),
    )
    new_id = int(cur.fetchone()[0])
    con.commit()
    return new_id


def bulk_add_machines(
    con: pyodbc.Connection, line_id: int, names: List[str]
) -> Tuple[int, int]:
    existing = {str(m["name"]).strip().lower() for m in list_machines(con, line_id)}
    added = 0
    skipped = 0
    for nm in [n.strip() for n in names if (n or "").strip()]:
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
def list_tasks(con: pyodbc.Connection, machine_id: int) -> List[Dict[str, Any]]:
    cur = _exec(
        con,
        """
        SELECT id, machine_id, code, name, operation_category, phases_json, sort_index, created_at
        FROM dbo.tasks
        WHERE machine_id=?
        ORDER BY sort_index, id
        """,
        (int(machine_id),),
    )
    rows = _fetchall_dict(cur)
    out: List[Dict[str, Any]] = []
    for r in rows:
        phases_raw = r.get("phases_json") or "[]"
        try:
            phases = json.loads(phases_raw)
        except Exception:
            phases = []
        out.append(
            dict(
                id=r["id"],
                machine_id=r["machine_id"],
                code=r["code"],
                name=r["name"],
                operation_category=r["operation_category"],
                phases=phases,
                sort_index=r["sort_index"],
                created_at=r["created_at"],
            )
        )
    return out


def get_task(con: pyodbc.Connection, task_id: int) -> Optional[Dict[str, Any]]:
    cur = _exec(
        con,
        """
        SELECT id, machine_id, code, name, operation_category, phases_json, sort_index, created_at
        FROM dbo.tasks
        WHERE id=?
        """,
        (int(task_id),),
    )
    r = _fetchone_dict(cur)
    if not r:
        return None
    phases_raw = r.get("phases_json") or "[]"
    try:
        phases = json.loads(phases_raw)
    except Exception:
        phases = []
    return dict(
        id=r["id"],
        machine_id=r["machine_id"],
        code=r["code"],
        name=r["name"],
        operation_category=r["operation_category"],
        phases=phases,
        sort_index=r["sort_index"],
        created_at=r["created_at"],
    )


def create_task(
    con: pyodbc.Connection,
    machine_id: int,
    name: str,
    operation_category: str,
    phases: List[str],
    sort_index: int = 0,
) -> int:
    created_at = now_iso()
    code = build_task_code(con, machine_id, operation_category, phases)
    phases_json = json.dumps(phases or [], ensure_ascii=False)

    cur = _exec(
        con,
        """
        INSERT INTO dbo.tasks(created_at, machine_id, code, name, operation_category, phases_json, sort_index)
        OUTPUT INSERTED.id
        VALUES(?,?,?,?,?,?,?)
        """,
        (
            created_at,
            int(machine_id),
            code,
            (name or "").strip(),
            (operation_category or "").strip(),
            phases_json,
            int(sort_index),
        ),
    )
    new_id = int(cur.fetchone()[0])
    con.commit()
    return new_id


def update_task(
    con: pyodbc.Connection,
    task_id: int,
    name: Optional[str] = None,
    operation_category: Optional[str] = None,
    phases: Optional[List[str]] = None,
    **changes,
) -> None:
    """
    Backwards-compatible update_task:

    - Supports your earlier explicit signature: update_task(con, task_id, name, operation_category, phases)
    - ALSO supports your later 'partial update' style: update_task(con, task_id, sort_index=..., phases=[...], ...)
    """
    update_cols: Dict[str, Any] = {}

    # If caller passed the explicit "full update" style
    if name is not None or operation_category is not None or phases is not None:
        if name is not None:
            update_cols["name"] = (name or "").strip()
        if operation_category is not None:
            update_cols["operation_category"] = (operation_category or "").strip()
        if phases is not None:
            update_cols["phases_json"] = json.dumps(phases or [], ensure_ascii=False)

    # Merge in any partial updates
    if "phases" in changes:
        update_cols["phases_json"] = json.dumps(
            changes.pop("phases") or [], ensure_ascii=False
        )
    update_cols.update(changes)

    if not update_cols:
        return

    cols = ", ".join(f"{k}=?" for k in update_cols.keys())
    values = list(update_cols.values()) + [int(task_id)]
    _exec(con, f"UPDATE dbo.tasks SET {cols} WHERE id=?", tuple(values))
    con.commit()


def delete_task(con: pyodbc.Connection, task_id: int) -> None:
    _exec(con, "DELETE FROM dbo.tasks WHERE id=?", (int(task_id),))
    con.commit()


# -------------------------
# Steps
# -------------------------
def add_step(
    con: pyodbc.Connection,
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
    cur = _exec(
        con,
        """
        INSERT INTO dbo.steps(
          created_at, task_id, step_no, step_desc, hazard_text, eng_controls, admin_controls,
          probability_code, severity_code, sort_index
        )
        OUTPUT INSERTED.id
        VALUES(?,?,?,?,?,?,?,?,?,?)
        """,
        (
            created_at,
            int(task_id),
            int(step_no),
            (step_desc or "").strip(),
            (hazard_text or "").strip(),
            (eng_controls or "").strip(),
            (admin_controls or "").strip(),
            float(probability_code),
            float(severity_code),
            int(sort_index),
        ),
    )
    new_id = int(cur.fetchone()[0])
    con.commit()
    return new_id


def list_steps(con: pyodbc.Connection, task_id: int) -> List[Dict[str, Any]]:
    cur = _exec(
        con,
        """
        SELECT id, task_id, step_no, step_desc, hazard_text, eng_controls, admin_controls,
               probability_code, severity_code, created_at
        FROM dbo.steps
        WHERE task_id=?
        ORDER BY step_no ASC
        """,
        (int(task_id),),
    )
    return _fetchall_dict(cur)


def get_step(
    con: pyodbc.Connection, task_id: int, step_no: int
) -> Optional[Dict[str, Any]]:
    cur = _exec(
        con,
        """
        SELECT id, task_id, step_no, step_desc, hazard_text, eng_controls, admin_controls,
               probability_code, severity_code, created_at
        FROM dbo.steps
        WHERE task_id=? AND step_no=?
        """,
        (int(task_id), int(step_no)),
    )
    return _fetchone_dict(cur)


def update_step(
    con: pyodbc.Connection,
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
    if step_id is None:
        if task_id is None or step_no is None:
            raise ValueError(
                "update_step requires either step_id OR (task_id and step_no)."
            )

        _exec(
            con,
            """
            UPDATE dbo.steps
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
        _exec(
            con,
            """
            UPDATE dbo.steps
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


def max_step_no(con: pyodbc.Connection, task_id: int) -> int:
    cur = _exec(
        con,
        "SELECT ISNULL(MAX(step_no), 0) AS mx FROM dbo.steps WHERE task_id=?",
        (int(task_id),),
    )
    row = cur.fetchone()
    return int(row[0] or 0)


def delete_step(con: pyodbc.Connection, task_id: int, step_no: int) -> None:
    _exec(
        con,
        "DELETE FROM dbo.steps WHERE task_id=? AND step_no=?",
        (int(task_id), int(step_no)),
    )
    con.commit()


def renumber_steps(con: pyodbc.Connection, task_id: int) -> None:
    steps = list_steps(con, int(task_id))
    try:
        # temp offset to avoid UNIQUE(task_id, step_no) collisions
        _exec(
            con,
            "UPDATE dbo.steps SET step_no = step_no + 10000 WHERE task_id=?",
            (int(task_id),),
        )
        for i, s in enumerate(steps, start=1):
            _exec(
                con,
                "UPDATE dbo.steps SET step_no=? WHERE task_id=? AND id=?",
                (int(i), int(task_id), int(s["id"])),
            )
        con.commit()
    except Exception:
        con.rollback()
        raise


def swap_steps(con: pyodbc.Connection, task_id: int, a: int, b: int) -> None:
    if a == b:
        return
    try:
        _exec(
            con,
            "UPDATE dbo.steps SET step_no=-1 WHERE task_id=? AND step_no=?",
            (int(task_id), int(a)),
        )
        _exec(
            con,
            "UPDATE dbo.steps SET step_no=? WHERE task_id=? AND step_no=?",
            (int(a), int(task_id), int(b)),
        )
        _exec(
            con,
            "UPDATE dbo.steps SET step_no=? WHERE task_id=? AND step_no=-1",
            (int(b), int(task_id)),
        )
        con.commit()
    except Exception:
        con.rollback()
        raise


def duplicate_step(con: pyodbc.Connection, task_id: int, step_no: int) -> int:
    src = get_step(con, int(task_id), int(step_no))
    if not src:
        raise ValueError("Step not found.")
    new_no = max_step_no(con, int(task_id)) + 1
    return add_step(
        con,
        task_id=int(task_id),
        step_no=new_no,
        step_desc=str(src["step_desc"]),
        hazard_text=str(src["hazard_text"]),
        eng_controls=str(src["eng_controls"]),
        admin_controls=str(src["admin_controls"]),
        probability_code=float(src["probability_code"]),
        severity_code=float(src["severity_code"]),
    )


# -------------------------
# DB utilities
# -------------------------
def db_counts(con: pyodbc.Connection) -> Dict[str, int]:
    def _count(tbl: str) -> int:
        cur = _exec(con, f"SELECT COUNT(*) FROM dbo.{tbl}")
        return int(cur.fetchone()[0] or 0)

    return {
        "lines": _count("lines"),
        "machines": _count("machines"),
        "tasks": _count("tasks"),
        "steps": _count("steps"),
    }


def task_counts_by_machine(
    con: pyodbc.Connection, line_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    where = ""
    params: List[Any] = []
    if line_id is not None:
        where = "WHERE m.line_id = ?"
        params.append(int(line_id))

    cur = _exec(
        con,
        f"""
        SELECT
          m.id, m.code, m.name,
          ISNULL(t.cnt, 0) AS task_count
        FROM dbo.machines m
        LEFT JOIN (
          SELECT machine_id, COUNT(*) AS cnt
          FROM dbo.tasks
          GROUP BY machine_id
        ) t ON t.machine_id = m.id
        {where}
        ORDER BY task_count DESC, m.id ASC
        """,
        tuple(params),
    )
    rows = cur.fetchall()
    return [
        {"machine_id": r[0], "code": r[1], "name": r[2], "task_count": r[3]}
        for r in rows
    ]


def list_tasks_for_machine(
    con: pyodbc.Connection, machine_id: int
) -> List[Dict[str, Any]]:
    """
    Returns list[dict] with step_count.
    Uses tasks.phases_json (your real schema).
    """
    cur = _exec(
        con,
        """
        SELECT
          t.id, t.machine_id, t.code, t.name, t.operation_category,
          t.phases_json, t.sort_index, t.created_at,
          ISNULL(s.cnt, 0) AS step_count
        FROM dbo.tasks t
        LEFT JOIN (
          SELECT task_id, COUNT(*) AS cnt
          FROM dbo.steps
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


def list_steps_for_task(con: pyodbc.Connection, task_id: int) -> List[Dict[str, Any]]:
    cur = _exec(
        con,
        """
        SELECT
          id, task_id, step_no, step_desc,
          hazard_text, eng_controls, admin_controls,
          probability_code, severity_code, created_at
        FROM dbo.steps
        WHERE task_id = ?
        ORDER BY step_no ASC
        """,
        (int(task_id),),
    )
    return _fetchall_dict(cur)


# -----------------------
# Import/Upsert helpers (kept; migrated to SQL Server)
# -----------------------
def ensure_line(con: pyodbc.Connection, name: str, code: str | None = None) -> int:
    name = (name or "").strip()
    code = (code or "").strip()

    cur = _exec(
        con, "SELECT id, name, code FROM dbo.lines WHERE code=? OR name=?", (code, name)
    )
    existing = _fetchone_dict(cur)

    if existing:
        updates: List[str] = []
        vals: List[Any] = []
        if name and existing["name"] != name:
            updates.append("name=?")
            vals.append(name)
        if code and existing["code"] != code:
            updates.append("code=?")
            vals.append(code)
        if updates:
            vals.append(int(existing["id"]))
            _exec(
                con,
                f"UPDATE dbo.lines SET {', '.join(updates)} WHERE id=?",
                tuple(vals),
            )
            con.commit()
        return int(existing["id"])

    created_at = now_iso()
    if not code:
        code = f"L{int(datetime.now().timestamp()) % 10000:04d}"

    cur2 = _exec(
        con,
        "INSERT INTO dbo.lines(created_at, code, name) OUTPUT INSERTED.id VALUES(?,?,?)",
        (created_at, code, name),
    )
    new_id = int(cur2.fetchone()[0])
    con.commit()
    return new_id


def ensure_machine(
    con: pyodbc.Connection, line_id: int, code: str, name: str | None = None
) -> int:
    code = (code or "").strip()
    name = (name or "").strip()

    cur = _exec(
        con,
        "SELECT id, code, name FROM dbo.machines WHERE line_id=? AND (code=? OR name=?)",
        (int(line_id), code, name),
    )
    existing = _fetchone_dict(cur)

    if existing:
        updates: List[str] = []
        vals: List[Any] = []
        if name and existing["name"] != name:
            updates.append("name=?")
            vals.append(name)
        if code and existing["code"] != code:
            updates.append("code=?")
            vals.append(code)
        if updates:
            vals.append(int(existing["id"]))
            _exec(
                con,
                f"UPDATE dbo.machines SET {', '.join(updates)} WHERE id=?",
                tuple(vals),
            )
            con.commit()
        return int(existing["id"])

    created_at = now_iso()
    if not code:
        code = f"M{int(datetime.now().timestamp()) % 10000:04d}"
    cur2 = _exec(
        con,
        "INSERT INTO dbo.machines(created_at, line_id, code, name) OUTPUT INSERTED.id VALUES(?,?,?,?)",
        (created_at, int(line_id), code, (name or code)),
    )
    new_id = int(cur2.fetchone()[0])
    con.commit()
    return new_id


def ensure_task(
    con: pyodbc.Connection,
    machine_id: int,
    code: str,
    name: str,
    category: str,
    phases: list[str],
) -> int:
    code = (code or "").strip()
    name = (name or "").strip()
    category = (category or "").strip()
    phases_json = json.dumps(phases or [], ensure_ascii=False)

    cur = _exec(
        con,
        "SELECT id, name, operation_category, phases_json FROM dbo.tasks WHERE machine_id=? AND code=?",
        (int(machine_id), code),
    )
    existing = _fetchone_dict(cur)

    if existing:
        updates: List[str] = []
        vals: List[Any] = []
        if name and existing["name"] != name:
            updates.append("name=?")
            vals.append(name)
        if category and existing["operation_category"] != category:
            updates.append("operation_category=?")
            vals.append(category)
        if existing["phases_json"] != phases_json:
            updates.append("phases_json=?")
            vals.append(phases_json)
        if updates:
            vals.append(int(existing["id"]))
            _exec(
                con,
                f"UPDATE dbo.tasks SET {', '.join(updates)} WHERE id=?",
                tuple(vals),
            )
            con.commit()
        return int(existing["id"])

    created_at = now_iso()
    if not code:
        code = f"T{int(datetime.now().timestamp()) % 10000:04d}"

    cur2 = _exec(
        con,
        """
        INSERT INTO dbo.tasks(created_at, machine_id, code, name, operation_category, phases_json)
        OUTPUT INSERTED.id
        VALUES(?,?,?,?,?,?)
        """,
        (created_at, int(machine_id), code, (name or code), category, phases_json),
    )
    new_id = int(cur2.fetchone()[0])
    con.commit()
    return new_id


def ensure_step(
    con: pyodbc.Connection,
    task_id: int,
    step_no: int,
    desc: str,
    hazard: str,
    eng_controls: str,
    admin_controls: str,
    p,
    s,
) -> int:
    step_no = int(step_no or 1)
    p_val = p if p not in (None, "") else 0
    s_val = s if s not in (None, "") else 0
    created_at = now_iso()

    cur = _exec(
        con,
        "SELECT id FROM dbo.steps WHERE task_id=? AND step_no=?",
        (int(task_id), int(step_no)),
    )
    existing = _fetchone_dict(cur)

    if existing:
        _exec(
            con,
            """
            UPDATE dbo.steps
            SET step_desc=?, hazard_text=?, eng_controls=?, admin_controls=?,
                probability_code=?, severity_code=?
            WHERE id=?
            """,
            (
                desc,
                hazard,
                eng_controls,
                admin_controls,
                float(p_val),
                float(s_val),
                int(existing["id"]),
            ),
        )
        con.commit()
        return int(existing["id"])

    cur2 = _exec(
        con,
        """
        INSERT INTO dbo.steps(
            created_at, task_id, step_no, step_desc, hazard_text,
            eng_controls, admin_controls, probability_code, severity_code
        )
        OUTPUT INSERTED.id
        VALUES(?,?,?,?,?,?,?,?,?)
        """,
        (
            created_at,
            int(task_id),
            int(step_no),
            desc,
            hazard,
            eng_controls,
            admin_controls,
            float(p_val),
            float(s_val),
        ),
    )
    new_id = int(cur2.fetchone()[0])
    con.commit()
    return new_id


# -----------------------
# Partial update helpers (kept)
# -----------------------
def update_machine(con: pyodbc.Connection, machine_id: int, **changes) -> None:
    if not changes:
        return
    cols = ", ".join(f"{k}=?" for k in changes.keys())
    values = list(changes.values()) + [int(machine_id)]
    _exec(con, f"UPDATE dbo.machines SET {cols} WHERE id=?", tuple(values))
    con.commit()


def clear_database(con: pyodbc.Connection) -> None:
    """
    Clears core tables and reseeds IDENTITY values.
    """
    try:
        _exec(con, "DELETE FROM dbo.steps")
        _exec(con, "DELETE FROM dbo.tasks")
        _exec(con, "DELETE FROM dbo.machines")
        _exec(con, "DELETE FROM dbo.lines")

        for tbl in ("steps", "tasks", "machines", "lines"):
            _exec(con, f"DBCC CHECKIDENT('dbo.{tbl}', RESEED, 0)")

        con.commit()
    except Exception:
        con.rollback()
        raise

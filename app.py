from __future__ import annotations

import json
from pathlib import Path
from io import BytesIO
import hashlib
from typing import Any, Dict, List
import pandas as pd
import streamlit as st

import store
from exporter import export_to_excel
from importer import import_from_excel
from risk_matrix import load_risk_matrix, lookup_rating

import base64
from pathlib import Path

# -----------------------------
# Configuration & Constants
# -----------------------------
APP_NAME = "SHERPA"
DB_PATH = str((Path(__file__).resolve().parent / "data" / "she_risk.db"))
UMS_PATH = "UMS.xlsx"
LIB_PATH = Path("libraries.json")

OP_CATEGORIES = [
    "Normal operations",
    "Abnormal operations",
    "Emergency operations",
    "CIP (cleaning in place)",
    "Changeover",
    "Maintenance (planned)",
    "Maintenance (autonomous)",
]
PHASES = ["Startup", "Running", "Shutdown"]

DEFAULT_LIB = {
    "hazards": [
        {"title": "Slip / trip / fall", "desc": "Wet floors, product, CIP spills."},
        {"title": "Pinch / entrapment", "desc": "Nips, rollers, moving parts."},
        {"title": "Cut / laceration", "desc": "Blades, sharp edges, cutters."},
        {"title": "Hot surface / burns", "desc": "Heat tunnels, hot pipes."},
        {"title": "Electrical shock", "desc": "Live panels, damaged cables."},
        {"title": "Manual handling / strain", "desc": "Lifting, awkward postures."},
        {"title": "Pressurised release", "desc": "Pneumatics, pressure lines."},
        {"title": "Chemical exposure (CIP)", "desc": "Caustic, acid, sanitiser."},
        {"title": "Noise exposure", "desc": "High noise zones, prolonged exposure."},
        {"title": "Unexpected start-up (LOTO)", "desc": "Isolation failure or bypass."},
    ],
    "eng_controls": [
        {"title": "Fixed guarding / interlocks", "desc": "Guarding prevents access to hazard zones."},
        {"title": "Emergency stop accessible", "desc": "E-stops reachable, tested, labelled."},
        {"title": "Pressure dump / bleed valve", "desc": "Safe release of pneumatic pressure."},
        {"title": "Spill containment / bunding", "desc": "Contain CIP/product spills."},
        {"title": "Anti-slip flooring / mats", "desc": "Reduce slip risk in wet areas."},
        {"title": "Isolation points labelled", "desc": "Clearly identified & accessible."},
    ],
    "admin_controls": [
        {"title": "SOP / SWMS in place", "desc": "Documented method and critical steps."},
        {"title": "LOTO procedure & permits", "desc": "Isolation, lock, verify, permit."},
        {"title": "Training / competency verified", "desc": "Operator trained, assessed, signed off."},
        {"title": "Signage & demarcation", "desc": "Warning signs, floor marking."},
        {"title": "Pre-start checks", "desc": "Start-up checklist and verification."},
        {"title": "Maintenance schedule", "desc": "Planned servicing and inspections."},
    ],
}


# -----------------------------
# Library Management
# -----------------------------
def _migrate_lib(data: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = {}
    for k in ["hazards", "eng_controls", "admin_controls"]:
        items = data.get(k, [])
        if items and isinstance(items[0], str):
            out[k] = [{"title": s, "desc": ""} for s in items]
        else:
            cleaned = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                title = str(it.get("title", "")).strip()
                desc = str(it.get("desc", "")).strip()
                if title:
                    cleaned.append({"title": title, "desc": desc})
            out[k] = cleaned
        if not out[k]:
            out[k] = DEFAULT_LIB[k]
    return out


def load_lib() -> Dict[str, List[Dict[str, str]]]:
    if not LIB_PATH.exists():
        LIB_PATH.write_text(json.dumps(DEFAULT_LIB, indent=2, ensure_ascii=False), encoding="utf-8")
        return DEFAULT_LIB
    try:
        data = json.loads(LIB_PATH.read_text(encoding="utf-8"))
        data = _migrate_lib(data)
        for k in DEFAULT_LIB.keys():
            data.setdefault(k, DEFAULT_LIB[k])
        return data
    except Exception:
        LIB_PATH.write_text(json.dumps(DEFAULT_LIB, indent=2, ensure_ascii=False), encoding="utf-8")
        return DEFAULT_LIB


def lib_titles(lib: Dict[str, List[Dict[str, str]]], key: str) -> List[str]:
    return [x["title"] for x in lib.get(key, [])]


def lib_default_desc(lib: Dict[str, List[Dict[str, str]]], key: str, title: str) -> str:
    for x in lib.get(key, []):
        if x["title"] == title:
            return x.get("desc", "") or ""
    return ""


# -----------------------------
# Data Transformation (Rows ⇄ Bullets)
# -----------------------------
def rows_to_bullets(rows: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if r.get("Remove"):
            continue
        title = str(r.get("Title", "")).strip()
        desc = str(r.get("Description", "")).strip()
        if not title:
            continue
        lines.append(f"- {title}: {desc}" if desc else f"- {title}")
    
    seen = set()
    out = []
    for ln in lines:
        k = ln.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(ln)
    return "\n".join(out).strip()


def bullets_to_rows(text: str) -> List[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return []
    rows = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if ln.startswith("-"):
            ln = ln.lstrip("-").strip()
        if ":" in ln:
            title, desc = ln.split(":", 1)
            rows.append({"Title": title.strip(), "Description": desc.strip(), "Remove": False})
        else:
            rows.append({"Title": ln.strip(), "Description": "", "Remove": False})
    return rows


# -----------------------------
# State Management
# -----------------------------
def init_state() -> None:
    defaults = {
        "page": "home",
        "line_id": None,
        "machine_id": None,

        "tasks_section": "Task",
        "append_task_id": None,

        # Draft task
        "draft_task_name": "",
        "draft_operation_category": OP_CATEGORIES[0],
        "draft_phases": ["Running"],
        "draft_steps": [],

        # Step inputs
        "step_desc": "",
        "hazard_pick": [],
        "eng_pick": [],
        "admin_pick": [],
        "hazard_rows": [],
        "eng_rows": [],
        "admin_rows": [],
        "prob_pick": None,
        "sev_pick": None,

        "step_error": "",
        "task_error": "",
        "new_line_err": "",
        "machines_bulk_result": "",
        "step_tab": "Hazards",

        # Edit saved
        "edit_task_id": None,
        "edit_step_no": None,
        "_edit_loaded_for": None,
        "edit_step_desc": "",
        "edit_hazard_rows": [],
        "edit_eng_rows": [],
        "edit_admin_rows": [],
        "edit_prob": None,
        "edit_sev": None,
        "edit_msg": "",

        # Edit task header
        "edit_task_name": "",
        "edit_task_cat": OP_CATEGORIES[0],
        "edit_task_phases": ["Running"],
        "task_hdr_msg": "",
        "confirm_delete_task": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def set_page(page: str) -> None:
    st.session_state["page"] = page


def set_section(section: str) -> None:
    st.session_state["tasks_section"] = section
    if section == "Task":
        st.session_state["append_task_id"] = None


def reset_step_inputs_cb() -> None:
    st.session_state["step_desc"] = ""
    st.session_state["hazard_pick"] = []
    st.session_state["eng_pick"] = []
    st.session_state["admin_pick"] = []
    st.session_state["hazard_rows"] = []
    st.session_state["eng_rows"] = []
    st.session_state["admin_rows"] = []
    st.session_state["step_error"] = ""
    st.session_state["step_tab"] = "Hazards"
    
    for k in ("hazard_editor", "eng_editor", "admin_editor"):
        if k in st.session_state:
            del st.session_state[k]


def reset_task_draft_cb():
    st.session_state["append_task_id"] = None
    st.session_state["draft_steps"] = []
    st.session_state["draft_task_name"] = ""
    st.session_state["draft_operation_category"] = OP_CATEGORIES[0]
    st.session_state["draft_phases"] = ["Running"]


def ensure_context(con) -> None:
    lines = store.list_lines(con)
    if lines and st.session_state["line_id"] is None:
        st.session_state["line_id"] = lines[0]["id"]
        machines = store.list_machines(con, lines[0]["id"])
        st.session_state["machine_id"] = machines[0]["id"] if machines else None


def on_line_change_cb():
    con = st.session_state["_con"]
    line_id = st.session_state["line_id"]
    machines = store.list_machines(con, line_id) if line_id else []
    st.session_state["machine_id"] = machines[0]["id"] if machines else None
    # only reset if user isn't mid-task
    if not st.session_state.get("draft_task_name"):
        reset_task_draft_cb()

def on_machine_change_cb():
    if not st.session_state.get("draft_task_name"):
        reset_task_draft_cb()


# -----------------------------
# Sync Multiselect ⇄ Rows
# -----------------------------
def sync_selected_to_rows_cb(rows_key: str, pick_key: str, lib_key: str) -> None:
    lib = st.session_state["_lib"]
    selected = st.session_state.get(pick_key, []) or []
    selected = [str(x) for x in selected]

    current = st.session_state.get(rows_key, []) or []
    current = [r for r in current if isinstance(r, dict)]
    lib_set = set(lib_titles(lib, lib_key))

    kept = []
    for r in current:
        title = str(r.get("Title", "")).strip()
        if not title:
            continue
        if title in selected or title not in lib_set:
            kept.append(r)

    kept_titles = {str(r.get("Title", "")).strip() for r in kept}

    for t in selected:
        if t not in kept_titles:
            kept.append({"Title": t, "Description": lib_default_desc(lib, lib_key, t), "Remove": False})

    st.session_state[rows_key] = kept


def add_custom_row_cb(rows_key: str) -> None:
    rows = st.session_state.get(rows_key, []) or []
    rows.append({"Title": "", "Description": "", "Remove": False})
    st.session_state[rows_key] = rows


# -----------------------------
# Risk Defaults
# -----------------------------
def ensure_risk_defaults() -> None:
    prob_opts = st.session_state["_probability"]
    sev_opts = st.session_state["_severity"]
    if st.session_state.get("prob_pick") is None and prob_opts:
        st.session_state["prob_pick"] = prob_opts[0].code
    if st.session_state.get("sev_pick") is None and sev_opts:
        st.session_state["sev_pick"] = sev_opts[0].code


def sidebar_center_logo_and_title(logo_path: str, title: str, logo_width_px: int = 100) -> None:
    p = Path(logo_path)
    img_html = ""
    if p.exists():
        b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        suffix = p.suffix.lower().lstrip(".")  # png/jpg/jpeg/webp
        mime = "png" if suffix == "png" else ("jpeg" if suffix in ["jpg", "jpeg"] else suffix)
        img_html = f"""
        <div style="display:flex; justify-content:center; margin-bottom: 8px;">
          <img src="data:image/{mime};base64,{b64}" style="width:{logo_width_px}px; height:auto;" />
        </div>
        """

    st.sidebar.markdown(
        f"""
        <div style="text-align:center; margin-bottom: 40px;">
        {img_html}
        <div style="font-size: 1.1rem; font-weight: 700; margin-top: 4px;">
            {title}
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Task & Step Logic
# -----------------------------
def add_step_cb():
    con = st.session_state["_con"]
    append_task_id = st.session_state.get("append_task_id")
    draft_steps = st.session_state.get("draft_steps", [])

    hazard_text = rows_to_bullets(st.session_state.get("hazard_rows", []))
    eng_controls = rows_to_bullets(st.session_state.get("eng_rows", []))
    admin_controls = rows_to_bullets(st.session_state.get("admin_rows", []))
    probability_code = float(st.session_state.get("prob_pick") or 0)
    severity_code = float(st.session_state.get("sev_pick") or 0)
    step_desc = (st.session_state.get("step_desc") or "").strip()

    if not step_desc:
        st.session_state["step_error"] = "Step description is required."
        return
    if not hazard_text:
        st.session_state["step_error"] = "At least one hazard is required."
        return

    st.session_state["step_error"] = ""

    # Append to existing task
    if append_task_id:
        task = store.get_task(con, append_task_id)
        if not task:
            st.error(f"Task {append_task_id} not found.")
            return

        store.add_step(
            con,
            task_id=append_task_id,
            step_no=len(store.list_steps_for_task(con, append_task_id)) + 1,
            step_desc=step_desc,
            hazard_text=hazard_text,
            eng_controls=eng_controls,
            admin_controls=admin_controls,
            probability_code=probability_code,
            severity_code=severity_code,
        )
        st.success(f"Step added directly to '{task['name']}'")
        reset_step_inputs_cb()
        return

    # Add to draft
    rating = lookup_rating(probability_code, severity_code, st.session_state["_rating_map"])
    draft_steps.append({
        "step_no": len(draft_steps) + 1,
        "step_desc": step_desc,
        "hazard_text": hazard_text,
        "eng_controls": eng_controls,
        "admin_controls": admin_controls,
        "probability_code": probability_code,
        "severity_code": severity_code,
        "rating": rating,
    })
    st.session_state["draft_steps"] = draft_steps
    reset_step_inputs_cb()
    st.session_state["tasks_section"] = "Steps"


def save_task_cb():
    con = st.session_state.get("_con")
    machine_id = st.session_state.get("machine_id")
    append_task_id = st.session_state.get("append_task_id")

    if not con:
        st.error("No database connection found.")
        return
    if not machine_id:
        st.error("No machine selected.")
        return

    # Append Mode
    if append_task_id:
        task = store.get_task(con, append_task_id)
        if task:
            st.success(f"'{task['name']}' updated. All steps already saved.")
        else:
            st.warning(f"Could not find task {append_task_id} to update.")
        st.session_state["append_task_id"] = None
        st.session_state["tasks_section"] = "Task"
        reset_step_inputs_cb()
        return

    # New Task Mode
    task_name = (st.session_state.get("draft_task_name") or "").strip()
    if not task_name and "last_task_name_input" in st.session_state:
        task_name = st.session_state["last_task_name_input"]
    op_category = (st.session_state.get("draft_operation_category") or "").strip()
    selected_phases = st.session_state.get("draft_phases") or []
    draft_steps = st.session_state.get("draft_steps") or []

    if not draft_steps:
        st.warning("Add at least one step before saving.")
        return

    try:
        task_id = store.create_task(con, machine_id, task_name, op_category, selected_phases)

        for idx, step in enumerate(draft_steps, start=1):
            store.add_step(
                con,
                task_id=task_id,
                step_no=idx,
                step_desc=step.get("step_desc", ""),
                hazard_text=step.get("hazard_text", ""),
                eng_controls=step.get("eng_controls", ""),
                admin_controls=step.get("admin_controls", ""),
                probability_code=step.get("probability_code", 0),
                severity_code=step.get("severity_code", 0),
            )

        # ✅ Save “last saved” info BEFORE resetting draft state
        st.session_state["last_saved_task_id"] = task_id
        st.session_state["last_saved_task_msg"] = f"✅ Task '{task_name}' saved successfully!"
        st.success(f"Task '{task_name}' saved with {len(draft_steps)} step(s).")

        reset_task_draft_cb()
        st.session_state["tasks_section"] = "Task"
        st.session_state["tasks_section"] = "Task"

    except Exception as e:
        st.error(f"Failed to save task: {e}")


def start_append_mode_cb(task_id: int) -> None:
    st.session_state["append_task_id"] = int(task_id)
    st.session_state["tasks_section"] = "Steps"
    if "step_tab" not in st.session_state or not st.session_state["step_tab"]:
        st.session_state["step_tab"] = "Hazards"
    reset_step_inputs_cb()


def enter_append_mode_cb(task_id: int):
    st.session_state["append_task_id"] = int(task_id)
    st.session_state["tasks_section"] = "Steps"     # ✅ go to step editor
    st.session_state["step_tab"] = "Hazards"        # ✅ default tab
    reset_step_inputs_cb()                          # ✅ clear inputs for new step

    # optional: you don't really need draft_steps in append mode, but harmless
    con = st.session_state["_con"]
    steps = store.list_steps_for_task(con, int(task_id))
    st.session_state["draft_steps"] = [dict(s) for s in steps]



def exit_append_mode_cb() -> None:
    st.session_state["append_task_id"] = None
    reset_step_inputs_cb()
    st.session_state["tasks_section"] = "Task"


# -----------------------------
# Edit Saved Steps
# -----------------------------
def open_edit_step_cb(task_id: int, step_no: int) -> None:
    st.session_state["tasks_section"] = "Edit saved"
    st.session_state["edit_step_no"] = int(step_no)
    st.session_state["_edit_loaded_for"] = None
    st.session_state["edit_msg"] = ""


def load_edit_if_needed(con) -> None:
    task_id = st.session_state.get("edit_task_id")
    step_no = st.session_state.get("edit_step_no")
    if not task_id or not step_no:
        return
    loaded_for = (int(task_id), int(step_no))
    if st.session_state.get("_edit_loaded_for") == loaded_for:
        return
    step = store.get_step(con, int(task_id), int(step_no))
    if not step:
        return

    st.session_state["_edit_loaded_for"] = loaded_for
    st.session_state["edit_step_desc"] = step["step_desc"]
    st.session_state["edit_hazard_rows"] = bullets_to_rows(step["hazard_text"])
    st.session_state["edit_eng_rows"] = bullets_to_rows(step["eng_controls"])
    st.session_state["edit_admin_rows"] = bullets_to_rows(step["admin_controls"])
    st.session_state["edit_prob"] = float(step["probability_code"])
    st.session_state["edit_sev"] = float(step["severity_code"])
    st.session_state["edit_msg"] = ""


def save_edit_step_cb(exit_after: bool = False) -> None:
    con = st.session_state["_con"]
    st.session_state["edit_msg"] = ""

    task_id = st.session_state.get("edit_task_id")
    step_no = st.session_state.get("edit_step_no")
    if not task_id or not step_no:
        st.session_state["edit_msg"] = "Select a task and step first."
        return

    desc = (st.session_state.get("edit_step_desc") or "").strip()
    if not desc:
        st.session_state["edit_msg"] = "Step description is required."
        return

    hazards = rows_to_bullets(st.session_state.get("edit_hazard_rows", []) or [])
    if not hazards:
        st.session_state["edit_msg"] = "At least one hazard is required."
        return

    eng = rows_to_bullets(st.session_state.get("edit_eng_rows", []) or [])
    admin = rows_to_bullets(st.session_state.get("edit_admin_rows", []) or [])

    prob = st.session_state.get("edit_prob")
    sev = st.session_state.get("edit_sev")
    if prob is None or sev is None:
        st.session_state["edit_msg"] = "Probability and Severity are required."
        return

    store.update_step(
        con,
        task_id=int(task_id),
        step_no=int(step_no),
        step_desc=desc,
        hazard_text=hazards,
        eng_controls=eng,
        admin_controls=admin,
        probability_code=float(prob),
        severity_code=float(sev),
    )
    st.session_state["_edit_loaded_for"] = None
    st.session_state["edit_msg"] = "Changes saved."
    if exit_after:
        st.session_state["tasks_section"] = "Task"
        st.session_state["_trigger_rerun"] = True


def open_edit_saved_cb(task_id: int) -> None:
    st.session_state["tasks_section"] = "Edit saved"


# -----------------------------
# Task Header Editor
# -----------------------------
def load_task_header_cb() -> None:
    con = st.session_state["_con"]
    tid = st.session_state.get("edit_task_id")
    if not tid:
        st.info("Select a task to edit from the list above.")
        return
    t = store.get_task(con, int(tid))
    if not t:
        return
    st.session_state["edit_task_name"] = t["name"]
    st.session_state["edit_task_cat"] = t["operation_category"]
    st.session_state["edit_task_phases"] = t["phases"]
    st.session_state["task_hdr_msg"] = ""


def save_task_header_cb() -> None:
    con = st.session_state["_con"]
    tid = st.session_state.get("edit_task_id")
    if not tid:
        st.info("Select a task to edit from the list above.")
        return
    nm = (st.session_state.get("edit_task_name") or "").strip()
    if not nm:
        st.session_state["task_hdr_msg"] = "Task name is required."
        return
    cat = st.session_state.get("edit_task_cat") or OP_CATEGORIES[0]
    phases = st.session_state.get("edit_task_phases") or ["Running"]
    store.update_task(con, int(tid), nm, cat, phases)
    st.session_state["task_hdr_msg"] = "Task header saved."


# -----------------------------
# Step Utilities (Move/Dup/Delete)
# -----------------------------
def move_step_cb(task_id: int, step_no: int, direction: int) -> None:
    con = st.session_state["_con"]
    steps = store.list_steps(con, int(task_id))
    nums = [s["step_no"] for s in steps]
    if step_no not in nums:
        return
    idx = nums.index(step_no)
    j = idx + direction
    if j < 0 or j >= len(nums):
        return
    store.swap_steps(con, int(task_id), int(nums[idx]), int(nums[j]))
    st.session_state["_edit_loaded_for"] = None


def delete_step_cb(task_id: int, step_no: int) -> None:
    con = st.session_state["_con"]
    store.delete_step(con, int(task_id), int(step_no))
    store.renumber_steps(con, int(task_id))
    st.session_state["_edit_loaded_for"] = None


def duplicate_step_cb(task_id: int, step_no: int) -> None:
    con = st.session_state["_con"]
    store.duplicate_step(con, int(task_id), int(step_no))
    st.session_state["_edit_loaded_for"] = None


# -----------------------------
# UI Components
# -----------------------------
def top_nav(con) -> None:
    # Don’t render nav on Home
    if st.session_state.get("page", "home") == "home":
        return

    lines = store.list_lines(con)
    has_line = bool(lines)
    has_machine = bool(st.session_state.get("machine_id"))

    c1, c2, c3, c4 = st.columns(4, gap="small")
    c1.button("Home", icon=":material/home:", use_container_width=True, on_click=set_page, args=("home",))
    c2.button("Create Line", icon=":material/add:", use_container_width=True, on_click=set_page, args=("create_line",))
    c3.button("Add Machines", icon=":material/settings:", use_container_width=True, on_click=set_page, args=("add_machines",), disabled=not has_line)
    c4.button("Add Tasks", icon=":material/assignment:", use_container_width=True, on_click=set_page, args=("add_tasks",), disabled=not (has_line and has_machine))


def sidebar(con) -> None:
    sidebar_center_logo_and_title("assets/logo.png", APP_NAME, logo_width_px=100)

    # --- Context ---
    with st.sidebar.expander("Context", expanded=False):
        line = store.get_line(con, st.session_state.get("line_id")) if st.session_state.get("line_id") else None
        machine = store.get_machine(con, st.session_state.get("machine_id")) if st.session_state.get("machine_id") else None

        st.write(f"**Line:** {line['code']} — {line['name']}" if line else "**Line:** _None_")
        st.write(f"**Machine:** {machine['code']} — {machine['name']}" if machine else "**Machine:** _None_")

    st.sidebar.divider()

    # --- Data management ---
    st.sidebar.markdown("#### Data management")

    # EXPORT action
    if st.sidebar.button(
        "Export to Excel",
        icon=":material/download:",
        use_container_width=True,
        key="btn_export_excel",
    ):
        out_path = Path("SHERPA_Data.xlsx")

        with st.spinner("Exporting…"):
            export_to_excel(
                con=con,
                out_path=str(out_path),
                rating_map=st.session_state["_rating_map"],
            )

        # Your exporter should create this alongside the xlsx (same folder)
        legend_path = Path("SHERPA_Legend.pdf")

        st.session_state["_export_ready"] = str(out_path) if out_path.exists() else None
        st.session_state["_legend_ready"] = str(legend_path) if legend_path.exists() else None
        st.sidebar.success("Export complete.")

    # DOWNLOADS (separate from import)
    export_ready = st.session_state.get("_export_ready")
    legend_ready = st.session_state.get("_legend_ready")

    with st.sidebar.expander("Downloads", expanded=bool(export_ready or legend_ready)):
        if export_ready and Path(export_ready).exists():
            st.download_button(
                "Download Excel (SHERPA_Data.xlsx)",
                data=Path(export_ready).read_bytes(),
                file_name="SHERPA_Data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="dl_export_excel",
            )
        else:
            st.caption("No Excel export available yet.")

        if legend_ready and Path(legend_ready).exists():
            st.download_button(
                "Download Legend (SHERPA_Legend.pdf)",
                data=Path(legend_ready).read_bytes(),
                file_name="SHERPA_Legend.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="dl_legend_pdf",
            )
        else:
            st.caption("No legend PDF available yet.")

    st.sidebar.divider()

    # IMPORT (still under Data management, but clearly separated)
    st.sidebar.markdown("### Data Management")

    nonce = st.session_state.get("_import_uploader_nonce", 0)
    uploader_key = f"import_xlsx_{nonce}"

    uploaded = st.sidebar.file_uploader(
        "Upload SHERPA_Data.xlsx to sync:",
        type=["xlsx"],
        accept_multiple_files=False,
        key=uploader_key,
    )

    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        already = st.session_state.get("_last_import_hash") == file_hash

        st.caption(
            "Syncs Lines, Machines, Tasks, and Steps from the spreadsheet into the local database."
            + (" (This exact file was already imported.)" if already else "")
        )

        if st.sidebar.button(
            "Sync from Excel",
            icon=":material/upload_file:",
            use_container_width=True,
            disabled=already,
            key="btn_import_sync",
        ):
            with st.spinner("Importing & syncing…"):
                import_from_excel(con, BytesIO(file_bytes))

            st.session_state["_last_import_hash"] = file_hash
            st.session_state["_post_import_rerun"] = True
            st.sidebar.success("Import complete — database synced.")

    # Safe rerun after import (outside the button callback)
    if st.session_state.get("_post_import_rerun"):
        st.session_state["_post_import_rerun"] = False

        # reset uploader + go home
        st.session_state["_import_uploader_nonce"] = st.session_state.get("_import_uploader_nonce", 0) + 1
        st.session_state["page"] = "home"
        st.rerun()

    st.sidebar.divider()
    if st.session_state["page"] == "add_tasks":
        current = st.session_state.get("tasks_section", "Task")
        task_ready = bool((st.session_state.get("draft_task_name") or "").strip()) or bool(st.session_state.get("append_task_id"))
        steps_ready = len(st.session_state.get("draft_steps") or []) > 0 or bool(st.session_state.get("append_task_id"))

        st.sidebar.markdown("#### Workflow")

        def nav_btn(label: str, section: str, disabled: bool = False, icon: str = None) -> None:
            st.sidebar.button(
                label,
                key=f"nav_{section}",
                use_container_width=True,
                on_click=set_section,
                args=(section,),
                disabled=disabled,
                type="primary" if current == section else "secondary",
                icon=icon
            )

        nav_btn("1. Task Details", "Task", disabled=False)
        nav_btn("2. Steps", "Steps", disabled=not task_ready)
        nav_btn("3. Review & Save", "Review", disabled=not steps_ready)
        nav_btn("Edit Saved", "Edit saved", disabled=False, icon=":material/edit:")


def home_page() -> None:
    st.header("SHERPA Dashboard", anchor=False)
    st.caption("Select an action to begin.")
    
    c1, c2, c3 = st.columns(3, gap="large")
    c1.button("Create New Line", icon=":material/add_circle:", type="primary", use_container_width=True, on_click=set_page, args=("create_line",))
    c2.button("Manage Machines", icon=":material/settings_applications:", use_container_width=True, on_click=set_page, args=("add_machines",))
    c3.button("Manage Tasks", icon=":material/list_alt:", use_container_width=True, on_click=set_page, args=("add_tasks",))


def create_line_page(con) -> None:
    st.header("Create Line", anchor=False)
    with st.container(border=True):
        st.text_input("Line Name", key="new_line_name", placeholder="e.g. Line 1 — Stick line")

        def create_line_cb():
            nm = (st.session_state.get("new_line_name") or "").strip()
            if not nm:
                st.session_state["new_line_err"] = "Line name is required."
                return
            lid = store.create_line(con, nm)
            st.session_state["line_id"] = lid
            st.session_state["new_line_name"] = ""
            st.session_state["new_line_err"] = ""
            set_page("add_machines")

        if st.session_state.get("new_line_err"):
            st.error(st.session_state["new_line_err"])
        st.button("Create & Continue", type="primary", on_click=create_line_cb)

    st.divider()
    st.subheader("Existing Lines", anchor=False)
    st.dataframe(store.list_lines(con), use_container_width=True, hide_index=True)


def add_machines_page(con) -> None:
    st.header("Manage Machines", anchor=False)

    lines = store.list_lines(con)
    if not lines:
        st.warning("Please create a line first.")
        st.button("Go to Create Line", type="primary", on_click=set_page, args=("create_line",))
        return

    line_lookup = {l["id"]: f"{l['code']} — {l['name']}" for l in lines}
    st.selectbox(
        "Select Line",
        options=list(line_lookup.keys()),
        format_func=lambda lid: line_lookup[lid],
        key="line_id",
        on_change=on_line_change_cb,
    )

    machines = store.list_machines(con, st.session_state["line_id"])
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Existing Machines", anchor=False)
        st.dataframe(
            [{"Code": m["code"], "Machine": m["name"], "Type": m.get("machine_type", "")} for m in machines],
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.subheader("Bulk Add Machines", anchor=False)
        with st.container(border=True):
            st.caption("Paste machine names (one per line). Duplicates will be skipped.")
            st.text_area("Machine List", key="machines_bulk", height=200)

            def bulk_add_cb():
                names = (st.session_state.get("machines_bulk") or "").splitlines()
                added, skipped = store.bulk_add_machines(con, st.session_state["line_id"], names)
                st.session_state["machines_bulk_result"] = f"Added {added} machines. Skipped {skipped} duplicates."
                st.session_state["machines_bulk"] = ""

            st.button("Add Machines", type="primary", on_click=bulk_add_cb, use_container_width=True)
            if st.session_state.get("machines_bulk_result"):
                st.success(st.session_state["machines_bulk_result"])

    st.divider()
    st.button("Next: Add Tasks", type="primary", on_click=set_page, args=("add_tasks",))


def add_tasks_page(con) -> None:
    st.header("Task Management", anchor=False)
    if st.session_state.get("_trigger_rerun"):
        st.session_state["_trigger_rerun"] = False
        st.rerun()


    lines = store.list_lines(con)
    if not lines:
        st.warning("Create a line first.")
        st.button("Create Line", type="primary", on_click=set_page, args=("create_line",))
        return

    line_lookup = {l["id"]: f"{l['code']} — {l['name']}" for l in lines}
    st.selectbox(
        "Line",
        options=list(line_lookup.keys()),
        format_func=lambda lid: line_lookup[lid],
        key="line_id",
        on_change=on_line_change_cb,
    )

    machines = store.list_machines(con, st.session_state["line_id"])
    if not machines:
        st.warning("No machines on this line yet.")
        st.button("Add Machines", type="primary", on_click=set_page, args=("add_machines",))
        return

    mach_lookup = {m["id"]: f"{m['code']} — {m['name']}" for m in machines}
    st.selectbox(
        "Machine",
        options=list(mach_lookup.keys()),
        format_func=lambda mid: mach_lookup[mid],
        key="machine_id",
        on_change=on_machine_change_cb,
    )

    st.divider()

    section = st.session_state.get("tasks_section", "Task")
    lib = st.session_state["_lib"]

    machine_id = st.session_state.get("machine_id")
    if machine_id:
        render_existing_tasks_for_machine(con, machine_id)
    else:
        st.info("Select a machine to view tasks.")

    msg = st.session_state.get("last_saved_task_msg", "")
    if msg:
        st.success(msg)
        st.session_state["last_saved_task_msg"] = ""

    # TASK
    if section == "Task":
        st.subheader("Create a New Task")
        if st.session_state.get("append_task_id"):
            t = store.get_task(con, int(st.session_state["append_task_id"]))
            st.warning(f"Append mode active: Adding steps to **{t['code']} — {t['name']}**.")
            st.button("Cancel & Start New Task", on_click=reset_task_draft_cb, use_container_width=True)

        with st.container(border=True):
            name_input = st.text_input("Task Name", key="draft_task_name", placeholder="Enter task name")
            if name_input.strip():
                st.session_state["last_task_name_input"] = name_input.strip()

            st.selectbox("Operation Category", OP_CATEGORIES, key="draft_operation_category")
            st.multiselect("Phase(s)", PHASES, key="draft_phases")
            st.session_state["last_task_draft_saved"] = True

        task_ready = bool((st.session_state.get("draft_task_name") or "").strip())
        st.button("Next: Define Steps", type="primary", disabled=not task_ready, on_click=set_section, args=("Steps",))

    # STEPS
    elif section == "Steps":
        ensure_risk_defaults()

        if st.session_state.get("step_tab") not in ["Hazards", "Engineering controls", "Admin controls"]:
            st.session_state["step_tab"] = "Hazards"


        append_task_id = st.session_state.get("append_task_id")
        if append_task_id:
            t = store.get_task(con, int(append_task_id))
            st.subheader(f"Adding Steps to: {t['code']} — {t['name']}")
        else:
            st.subheader("Define Steps")

        with st.container(border=True):
            st.text_area("Step Description", key="step_desc", height=90)

            tab = st.segmented_control(
                "Controls Category",
                ["Hazards", "Engineering controls", "Admin controls"],
                key="step_tab",
                label_visibility="collapsed",
            )

            if tab == "Hazards":
                st.multiselect(
                    "Standard Hazards",
                    options=lib_titles(lib, "hazards"),
                    key="hazard_pick",
                    on_change=sync_selected_to_rows_cb,
                    args=("hazard_rows", "hazard_pick", "hazards"),
                )
                st.button("Add Custom Hazard", on_click=add_custom_row_cb, args=("hazard_rows",))
                edited = st.data_editor(
                    st.session_state.get("hazard_rows", []),
                    key="hazard_editor",
                    use_container_width=True,
                    num_rows="dynamic",
                )
                st.session_state["hazard_rows"] = edited

            elif tab == "Engineering controls":
                st.multiselect(
                    "Standard Engineering Controls",
                    options=lib_titles(lib, "eng_controls"),
                    key="eng_pick",
                    on_change=sync_selected_to_rows_cb,
                    args=("eng_rows", "eng_pick", "eng_controls"),
                )
                st.button("Add Custom Control", on_click=add_custom_row_cb, args=("eng_rows",))
                edited = st.data_editor(
                    st.session_state.get("eng_rows", []),
                    key="eng_editor",
                    use_container_width=True,
                    num_rows="dynamic",
                )
                st.session_state["eng_rows"] = edited

            else:
                st.multiselect(
                    "Standard Admin Controls",
                    options=lib_titles(lib, "admin_controls"),
                    key="admin_pick",
                    on_change=sync_selected_to_rows_cb,
                    args=("admin_rows", "admin_pick", "admin_controls"),
                )
                st.button("Add Custom Control", on_click=add_custom_row_cb, args=("admin_rows",))
                edited = st.data_editor(
                    st.session_state.get("admin_rows", []),
                    key="admin_editor",
                    use_container_width=True,
                    num_rows="dynamic",
                )
                st.session_state["admin_rows"] = edited

            prob_opts = st.session_state["_probability"]
            sev_opts = st.session_state["_severity"]

            c1, c2, c3 = st.columns([1, 1, 1], gap="large")
            with c1:
                st.selectbox(
                    "Probability",
                    options=[p.code for p in prob_opts],
                    format_func=lambda c: next((p.label for p in prob_opts if p.code == c), str(c)),
                    key="prob_pick",
                )
            with c2:
                st.selectbox(
                    "Severity",
                    options=[s.code for s in sev_opts],
                    format_func=lambda c: next((s.label for s in sev_opts if s.code == c), str(c)),
                    key="sev_pick",
                )
            with c3:
                rating = lookup_rating(float(st.session_state["prob_pick"]), float(st.session_state["sev_pick"]), st.session_state["_rating_map"])
                st.metric("Risk Rating", rating)

            st.button("Add Step", on_click=add_step_cb, use_container_width=True, icon=":material/add:")

        if st.session_state.get("step_error"):
            st.error(st.session_state["step_error"])

        # Show Steps Table
        if append_task_id:
            st.markdown("#### Saved Steps (Current Task)")
            steps = store.list_steps(con, int(append_task_id))
            st.dataframe(
                [
                    {
                        "#": s["step_no"],
                        "Step": s["step_desc"],
                        "Hazards": s["hazard_text"].replace("\n", " • "),
                        "Eng Controls": s["eng_controls"].replace("\n", " • "),
                        "Admin Controls": s["admin_controls"].replace("\n", " • "),
                        "Probability": s["probability_code"],
                        "Severity": s["severity_code"],
                        "Risk": lookup_rating(float(s["probability_code"]), float(s["severity_code"]), st.session_state["_rating_map"]),
                    }
                    for s in steps
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.markdown("#### Review Changes")
            steps = st.session_state.get("draft_steps") or []
            st.dataframe(
                [
                    {
                        "#": s["step_no"],
                        "Step": s["step_desc"],
                        "Hazards": s["hazard_text"].replace("\n", " • "),
                        "Eng Controls": s["eng_controls"].replace("\n", " • "),
                        "Admin Controls": s["admin_controls"].replace("\n", " • "),
                        "Risk": s["rating"],
                    }
                    for s in steps
                ],
                use_container_width=True,
                hide_index=True,
            )
            st.button("Next: Review", type="primary", disabled=len(steps) == 0, on_click=set_section, args=("Review",))

    # REVIEW
    elif section == "Review":
        st.subheader("Commit Changes")
        steps = st.session_state.get("draft_steps") or []
        if steps:
            st.dataframe(
                [
                    {
                        "#": s["step_no"],
                        "Step": s["step_desc"],
                        "Hazards": s["hazard_text"].replace("\n", " • "),
                        "Risk Rating": s["rating"],
                    }
                    for s in steps
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No draft steps available.")

        st.button("Save Task", on_click=save_task_cb, use_container_width=True, type="primary", icon=":material/save:")

        if st.session_state.get("task_error"):
            st.error(st.session_state["task_error"])

        if st.session_state.get("append_task_id"):
            t = store.get_task(con, int(st.session_state["append_task_id"]))
            st.success(f"Saved {t['code']} — {t['name']}")
            c1, c2 = st.columns(2, gap="large")
            c1.button("Add Another Step", type="primary", use_container_width=True, on_click=start_append_mode_cb, args=(int(t["id"]),))
            c2.button("Start New Task", use_container_width=True, on_click=reset_task_draft_cb)

    # EDIT SAVED
    else:
        st.subheader("Edit Saved Tasks")
        if st.button(
            "Exit Edit Mode",
            icon=":material/close:",
            use_container_width=True,
            key="exit_edit_mode",
        ):
            st.session_state["tasks_section"] = "Task"
            # st.session_state["edit_task_id"] = None
            st.session_state["edit_step_no"] = None
            st.session_state["_edit_loaded_for"] = None
            st.session_state["edit_msg"] = ""
            st.session_state["task_hdr_msg"] = ""
            st.rerun()

        tasks = store.list_tasks(con, st.session_state["machine_id"])
        if not tasks:
            st.info("No tasks available to edit.")
            return

        task_lookup = {t["id"]: f"{t['code']} — {t['name']}" for t in tasks}
        if st.session_state.get("edit_task_id") not in task_lookup:
            fallback_tid = list(task_lookup.keys())[0]
        else:
            fallback_tid = st.session_state["edit_task_id"]

        # --- SELECTBOX ---
        tid = st.selectbox(
            "Select Task to Edit",
            options=list(task_lookup.keys()),
            format_func=lambda tid: task_lookup[tid],
            key="edit_task_id",
            index=list(task_lookup.keys()).index(fallback_tid),
            on_change=load_task_header_cb,
        )
        if not tid:
            st.info("Select a task to edit from the list above.")
            return

        # Task Header Editor
        with st.container(border=True):
            st.markdown("#### Task Details")
            st.text_input("Task Name", key="edit_task_name")
            st.selectbox("Operation Category", OP_CATEGORIES, key="edit_task_cat")
            st.multiselect("Phases", PHASES, key="edit_task_phases")
            st.button("Save Task Details", type="primary", on_click=save_task_header_cb)
            if st.session_state.get("task_hdr_msg"):
                st.success(st.session_state["task_hdr_msg"])

        tid = st.session_state.get("edit_task_id")
        if not tid:
            st.warning("Please select a task to edit.")
            return
        steps = store.list_steps(con, int(tid))
        st.markdown("#### Step Management")
        if not steps:
            st.info("No steps found for this task.")
        else:
            for s in steps:
                row = st.container(border=True)
                with row:
                    left, a, b, c, d, e = st.columns([10, 1, 1, 1, 1, 1], gap="small")
                    with left:
                        st.write(f"**Step {s['step_no']}** — {s['step_desc']}")
                        st.caption(f"Risk: {lookup_rating(float(s['probability_code']), float(s['severity_code']), st.session_state['_rating_map'])}")

                    a.button(" ", key=f"up_{s['id']}",  icon=":material/arrow_upward:",
                            on_click=move_step_cb, args=(int(s["task_id"]), int(s["step_no"]), -1),
                            help="Move step up")

                    b.button(" ", key=f"dn_{s['id']}",  icon=":material/arrow_downward:",
                            on_click=move_step_cb, args=(int(s["task_id"]), int(s["step_no"]), +1),
                            help="Move step down")

                    c.button(" ", key=f"dup_{s['id']}", icon=":material/content_copy:",
                            on_click=duplicate_step_cb, args=(int(s["task_id"]), int(s["step_no"])),
                            help="Duplicate step")

                    d.button(" ", key=f"del_{s['id']}", icon=":material/delete:",
                            on_click=delete_step_cb, args=(int(s["task_id"]), int(s["step_no"])),
                            help="Delete step")

                    e.button(" ", key=f"edit_{s['id']}", icon=":material/edit:",
                            on_click=open_edit_step_cb, args=(int(s["task_id"]), int(s["step_no"])),
                            help="Edit step")


        st.divider()

        if st.session_state.get("edit_step_no") is not None and st.session_state["tasks_section"] == "Edit saved":
            # render editor box

            # Step Editor
            tid = st.session_state.get("edit_task_id")
            if not tid:
                st.warning("Please select a task to edit.")
                return
            steps = store.list_steps(con, int(tid))
            if steps:
                step_nos = [s["step_no"] for s in steps]
                if st.session_state.get("edit_step_no") not in step_nos:
                    st.session_state["edit_step_no"] = step_nos[0]

                st.selectbox("Select Step Number", options=step_nos, key="edit_step_no", format_func=lambda x: f"Step {x}")
                load_edit_if_needed(con)

                prob_opts = st.session_state["_probability"]
                sev_opts = st.session_state["_severity"]

                with st.container(border=True):
                    st.text_area("Step Description", key="edit_step_desc", height=90)

                    tab_h, tab_e, tab_a = st.tabs(["Hazards", "Engineering", "Administrative"])
                    with tab_h:
                        edited = st.data_editor(st.session_state.get("edit_hazard_rows", []), key="edit_hazard_editor", use_container_width=True, num_rows="dynamic")
                        st.session_state["edit_hazard_rows"] = edited
                    with tab_e:
                        edited = st.data_editor(st.session_state.get("edit_eng_rows", []), key="edit_eng_editor", use_container_width=True, num_rows="dynamic")
                        st.session_state["edit_eng_rows"] = edited
                    with tab_a:
                        edited = st.data_editor(st.session_state.get("edit_admin_rows", []), key="edit_admin_editor", use_container_width=True, num_rows="dynamic")
                        st.session_state["edit_admin_rows"] = edited

                    c1, c2, c3 = st.columns([1, 1, 1], gap="large")
                    with c1:
                        st.selectbox(
                            "Probability",
                            options=[p.code for p in prob_opts],
                            format_func=lambda c: next((p.label for p in prob_opts if p.code == c), str(c)),
                            key="edit_prob",
                        )
                    with c2:
                        st.selectbox(
                            "Severity",
                            options=[s.code for s in sev_opts],
                            format_func=lambda c: next((s.label for s in sev_opts if s.code == c), str(c)),
                            key="edit_sev",
                        )
                    with c3:
                        rating = lookup_rating(float(st.session_state["edit_prob"]), float(st.session_state["edit_sev"]), st.session_state["_rating_map"])
                        st.metric("Risk Rating", rating)

                    st.button(
                        "Save Changes",
                        icon=":material/save:",
                        type="primary",
                        on_click=save_edit_step_cb,
                        kwargs={"exit_after": True},
                    )
                    if st.session_state.get("edit_msg"):
                        st.success(st.session_state.get("edit_msg"))


def render_existing_tasks_for_machine(con, machine_id: int) -> None:
    with st.expander("View Existing Tasks", expanded=False):
        tasks = store.list_tasks_for_machine(con, int(machine_id))

        if not tasks:
            st.info("No tasks found for this machine.")
            return

        # Search
        q = (st.text_input("Search Tasks (Code or Name)", key="existing_task_search") or "").strip().lower()
        filtered = (
            [t for t in tasks if (q in t["code"].lower() or q in t["name"].lower())]
            if q
            else tasks
        )

        # Summary Table
        df = pd.DataFrame(
            [
                {
                    "Task ID": t["id"],
                    "Code": t["code"],
                    "Name": t["name"],
                    "Category": t["operation_category"],
                    "Steps": int(t.get("step_count", 0) or 0),
                }
                for t in filtered
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Inspect Task
        labels = [f"{t['code']} — {t['name']} ({int(t.get('step_count', 0) or 0)} steps)" for t in filtered]
        label_to_id = {labels[i]: filtered[i]["id"] for i in range(len(filtered))}

        default_idx = 0
        last_id = st.session_state.get("last_saved_task_id")
        if last_id is not None:
            for i, t in enumerate(filtered):
                if int(t["id"]) == int(last_id):
                    default_idx = i
                    break

        picked = st.selectbox("Select Task to Inspect", options=labels, index=default_idx, key="existing_task_pick")
        task_id = int(label_to_id[picked])

        in_append = bool(st.session_state.get("append_task_id"))

        if in_append:
            c1, c2, c3, c4 = st.columns([1, 1, 1, 1], gap="small")
        else:
            c1, c2, c3 = st.columns([1, 1, 1], gap="small")

        with c1:
            st.button(
                "Append Steps",
                icon=":material/playlist_add:",
                on_click=enter_append_mode_cb,
                args=(task_id,),
                use_container_width=True,
            )

        # ✅ Exit Append Mode appears right after Append Steps
        if in_append:
            with c2:
                st.button(
                    "Exit Append Mode",
                    icon=":material/close:",
                    on_click=exit_append_mode_cb,
                    use_container_width=True,
                    key=f"exit_append_{task_id}",
                )
            edit_col = c3
            del_col = c4
        else:
            edit_col = c2
            del_col = c3

        with edit_col:
            st.button(
                "Edit Task",
                icon=":material/edit:",
                on_click=open_edit_saved_cb,
                args=(task_id,),
                use_container_width=True,
            )

        with del_col:
            if st.button(
                "Delete Task",
                icon=":material/delete:",
                type="secondary",
                use_container_width=True,
                key=f"delete_task_{task_id}",
            ):
                st.session_state["confirm_delete_task"] = int(task_id)



        # Task Preview
        for t in filtered:
            if t["id"] == task_id:
                st.markdown(f"#### {t['code']} — {t['name']}")
                steps = store.list_steps(con, int(t["id"]))
                if not steps:
                    st.caption("No steps recorded.")
                    continue

                step_rows = [
                    {
                        "#": s["step_no"],
                        "Step": s["step_desc"],
                        "Hazards": s["hazard_text"].replace("\n", " • "),
                        "Eng Controls": s["eng_controls"].replace("\n", " • "),
                        "Admin Controls": s["admin_controls"].replace("\n", " • "),
                        "Probability": s["probability_code"],
                        "Severity": s["severity_code"],
                        "Risk": lookup_rating(float(s["probability_code"]), float(s["severity_code"]), st.session_state["_rating_map"]),
                    }

                    for s in steps
                ]
                st.dataframe(pd.DataFrame(step_rows), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title=APP_NAME, page_icon=":material/security:", layout="wide")
    init_state()

    con = store.connect(DB_PATH)
    st.session_state["_con"] = con

    if not Path(UMS_PATH).exists():
        st.error(f"Missing configuration file: {UMS_PATH}")
        st.stop()

    probability, severity, rating_map = load_risk_matrix(UMS_PATH)
    st.session_state["_probability"] = probability
    st.session_state["_severity"] = severity
    st.session_state["_rating_map"] = rating_map

    st.session_state["_lib"] = load_lib()
    ensure_context(con)

    top_nav(con)
    sidebar(con)

    page = st.session_state.get("page", "home")
    if page == "home":
        home_page()
    elif page == "create_line":
        create_line_page(con)
    elif page == "add_machines":
        add_machines_page(con)
    elif page == "add_tasks":
        add_tasks_page(con)
    else:
        set_page("home")
        home_page()


if __name__ == "__main__":
    main()
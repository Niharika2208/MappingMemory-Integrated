# db_utils.py
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from typing import List, Tuple

# Load .env once
load_dotenv()

# Prefer full URL, else build from parts
DB_URL = os.getenv("DB_URL") or (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

engine = create_engine(DB_URL, pool_pre_ping=True, pool_size=5, max_overflow=5)

def ping():
    """Simple connectivity check."""
    with engine.connect() as conn:
        return conn.execute(text("select current_database(), current_user, current_date")).fetchone()

def insert_project(origin_project_id: str, origin_project_title: str | None = None) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO projects (origin_project_id, origin_project_title)
                VALUES (:pid, :ptitle)
                ON CONFLICT (origin_project_id) DO UPDATE
                SET origin_project_title = COALESCE(EXCLUDED.origin_project_title, projects.origin_project_title)
            """),
            {"pid": origin_project_id, "ptitle": origin_project_title}
        )

def upsert_common_concept(origin_project_id: str, sys_unitclass_lib: str, sys_unitclass: str) -> str:
    with engine.begin() as conn:
        row = conn.execute(
            text("""
            INSERT INTO common_concept(origin_project_id, sys_unitclass_lib, sys_unitclass)
                VALUES(:pid, :lib, :cls)
                ON CONFLICT(origin_project_id, sys_unitclass_lib, sys_unitclass) DO UPDATE
                SET sys_unitclass_lib = EXCLUDED.sys_unitclass_lib
                RETURNING concept_id
                """),
            {"pid": origin_project_id, "lib": sys_unitclass_lib, "cls": sys_unitclass}
        ).fetchone()
        return str(row[0])

def upsert_common_concept_attr(origin_project_id: str, concept_id: str, attr_names: list[str], sys_unitclass: str) -> int:
    rows = [{"pid": origin_project_id, "cid": concept_id, "attr": a.strip(), "cls": sys_unitclass}
            for a in attr_names if a and a.strip()]

    with engine.begin() as conn:
        result = conn.execute(
            text("""
                INSERT INTO common_concept_attr (origin_project_id, concept_id, attr_names, sys_unitclass)
                VALUES (:pid, :cid, :attr, :cls)
                ON CONFLICT (concept_id, attr_names, sys_unitclass) DO NOTHING
            """),
            rows
        )
        return getattr(result, "rowcount", 0)


# db_utils.py
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

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

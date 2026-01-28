
# --- src/database.py ---
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, BLOB, DateTime, Enum, insert, select
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import numpy as np
import pymysql

DATABASE_URL = "mysql+pymysql://root:Yeo030508@localhost/face_auth_system"
engine = create_engine(DATABASE_URL, echo=False, future=True)
metadata = MetaData()

users = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(255), nullable=False, unique=True),
    Column("mask_embedding", BLOB),
    Column("no_mask_embedding", BLOB),
    Column("created_at", DateTime, default=datetime.utcnow)
)

punch_log = Table(
    "punch_log", metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(255)),
    Column("timestamp", DateTime),
    Column("spoof_status", String(50)),
    Column("mask_status", String(50)),
    Column("type", Enum("login", "logout")),
    Column("created_at", DateTime, default=datetime.utcnow)
)

SessionLocal = sessionmaker(bind=engine, autoflush=False)


def insert_user(name, mask_emb=None, no_mask_emb=None):
    with engine.begin() as conn:
        # Check if user exists
        result = conn.execute(select(users).where(users.c.name == name)).first()
        if result:
            raise ValueError("User already exists.")

        values = {
            "name": name,
            "mask_embedding": mask_emb.tobytes() if mask_emb is not None else None,
            "no_mask_embedding": no_mask_emb.tobytes() if no_mask_emb is not None else None,
        }
        conn.execute(insert(users).values(**values))


def get_user_embeddings(name):
    with engine.connect() as conn:
        result = conn.execute(select(users).where(users.c.name == name)).first()
        if result:
            return {
                "name": result.name,
                "mask_embedding": np.frombuffer(result.mask_embedding, dtype=np.float32) if result.mask_embedding else None,
                "no_mask_embedding": np.frombuffer(result.no_mask_embedding, dtype=np.float32) if result.no_mask_embedding else None
            }
        return None


def insert_log_entry(name, timestamp, spoof, mask, punch_type):
    with engine.begin() as conn:
        conn.execute(insert(punch_log).values(
            name=name,
            timestamp=timestamp,
            spoof_status=spoof,
            mask_status=mask,
            type=punch_type
        ))


def fetch_users():
    with engine.connect() as conn:
        result = conn.execute(select(users)).fetchall()
        user_list = []
        for row in result:
            user_list.append({
                "name": row.name,
                "mask": np.frombuffer(row.mask_embedding, dtype=np.float32) if row.mask_embedding else None,
                "no_mask": np.frombuffer(row.no_mask_embedding, dtype=np.float32) if row.no_mask_embedding else None

            })
        return user_list


def fetch_punch_logs():
    with engine.connect() as conn:
        result = conn.execute(select(punch_log)).fetchall()
        return [dict(row._mapping) for row in result]


def get_log_by_id(log_id):
    with engine.connect() as conn:
        query = select(punch_log).where(punch_log.c.id == log_id)
        result = conn.execute(query).first()
        return dict(result._mapping) if result else None


def update_log_entry(log_id, name, spoof_status, mask_status, punch_type):
    with engine.begin() as conn:
        stmt = (
            update(punch_log)
            .where(punch_log.c.id == log_id)
            .values(name=name, spoof_status=spoof_status, mask_status=mask_status, type=punch_type)
        )
        conn.execute(stmt)


def fetch_all_users():
    with engine.connect() as conn:
        result = conn.execute(select(users)).fetchall()
        users_list = []
        for row in result:
            users_list.append({
                "id": row.id,
                "name": row.name,
                "mask_embedding": row.mask_embedding,
                "no_mask_embedding": row.no_mask_embedding,
                "created_at": row.created_at,
                "mask_registered": bool(row.mask_embedding),
                "no_mask_registered": bool(row.no_mask_embedding),
            })
        return users_list
    

def delete_user_by_id(user_id: int):
    with engine.begin() as conn:
        conn.execute(users.delete().where(users.c.id == user_id))


def fetch_latest_log(name: str):
    with engine.connect() as conn:
        query = (
            select(punch_log.c.type)
            .where(punch_log.c.name == name)
            .order_by(punch_log.c.timestamp.desc())
            .limit(1)
        )
        result = conn.execute(query).first()
        return result.type if result else None
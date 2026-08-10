import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DB_DIR = os.path.join(PROJECT_ROOT, "db")


SQLALCHEMY_DATABASE_URL = f"sqlite:///{os.path.join(DB_DIR, 'scholarai.db')}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
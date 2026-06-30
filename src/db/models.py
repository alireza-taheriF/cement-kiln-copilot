"""SQLAlchemy ORM models and engine/session helpers.

Defines the production persistence schema for cement-process optimization.

The schema follows a normalized, tag-centric time-series design:

* :class:`Tag`        - metadata describing a single measured/derived signal.
* :class:`Signal`     - the time-series fact table (one value per tag per ts).
* :class:`LabResult`  - laboratory quality-control measurements.

The database URL is resolved from the ``DATABASE_URL`` environment variable,
falling back to a local SQLite database for development.
"""

from __future__ import annotations

import os

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------
class Tag(Base):
    """Metadata describing a single plant signal (a "tag").

    A tag is the dictionary entry that gives meaning to the raw values stored
    in :class:`Signal`. Tags cover process measurements, operator/optimizer
    setpoints, control variables, and engineered/derived series.
    """

    __tablename__ = "tags"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(128), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    unit = Column(String(32), nullable=True)
    # One of: process | setpoint | control | derived
    type = Column(String(32), nullable=False, default="process")
    is_target = Column(Boolean, nullable=False, default=False)

    signals = relationship(
        "Signal",
        back_populates="tag",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Tag id={self.id} name={self.name!r} type={self.type!r}>"


class Signal(Base):
    """A single time-stamped value for a given tag (the fact table).

    The primary key is the composite of ``(ts, tag_id)`` so that at most one
    value exists per tag per timestamp. A secondary ``(tag_id, ts)`` index
    supports the common access pattern of pulling one tag's history over a
    time window.
    """

    __tablename__ = "signals"

    ts = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    tag_id = Column(
        Integer,
        ForeignKey("tags.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    value = Column(Float, nullable=False)

    tag = relationship("Tag", back_populates="signals")

    __table_args__ = (
        Index("ix_signals_tag_id_ts", "tag_id", "ts"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Signal tag_id={self.tag_id} ts={self.ts} value={self.value}>"


class LabResult(Base):
    """A laboratory quality-control measurement.

    Lab results arrive irregularly (per sample) and are later aligned with the
    high-frequency :class:`Signal` stream during feature engineering.
    """

    __tablename__ = "lab_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(DateTime(timezone=True), nullable=False, index=True)
    parameter = Column(String(64), nullable=False, index=True)
    value = Column(Float, nullable=False)
    sample_id = Column(String(64), nullable=True)
    notes = Column(Text, nullable=True)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<LabResult id={self.id} parameter={self.parameter!r} ts={self.ts}>"


# ---------------------------------------------------------------------------
# Engine / session helpers
# ---------------------------------------------------------------------------
def _resolve_database_url() -> str:
    """Resolve the DB URL from env, defaulting to local SQLite."""
    return os.getenv("DATABASE_URL", "sqlite:///./cement_copilot_dev.db")


def get_engine(database_url: str | None = None, echo: bool = False) -> Engine:
    """Create a SQLAlchemy engine.

    Parameters
    ----------
    database_url:
        Explicit SQLAlchemy URL. When omitted, it is resolved from the
        ``DATABASE_URL`` environment variable (SQLite dev fallback).
    echo:
        Whether SQLAlchemy should log emitted SQL.
    """
    url = database_url or _resolve_database_url()
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, echo=echo, pool_pre_ping=True, connect_args=connect_args)


def get_session_factory(engine: Engine | None = None) -> sessionmaker:
    """Return a configured ``sessionmaker`` bound to ``engine``."""
    engine = engine or get_engine()
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db(engine: Engine | None = None) -> None:
    """Create all tables. Idempotent; safe to call on startup for dev."""
    engine = engine or get_engine()
    Base.metadata.create_all(engine)

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text

from config import settings

engine = create_async_engine(settings.DATABASE_URL, echo=False, pool_pre_ping=True)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()


# 平台集成新增列 (已有数据库的轻量迁移: ADD COLUMN IF NOT EXISTS)
_EXPERIMENT_MIGRATIONS = [
    ("canonical_config", "JSON"),
    ("git_commit", "VARCHAR(64)"),
    ("environment", "JSON"),
    ("task_dir", "VARCHAR(512)"),
    ("last_round", "INTEGER"),
    ("exit_code", "INTEGER"),
    ("error_tail", "TEXT"),
]


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # 迁移: 为历史 experiments 表补齐平台集成列, 并回填默认值
        for col, ctype in _EXPERIMENT_MIGRATIONS:
            await conn.execute(text(
                f"ALTER TABLE experiments ADD COLUMN IF NOT EXISTS {col} {ctype}"))
        await conn.execute(text(
            "UPDATE experiments SET canonical_config = '{}' "
            "WHERE canonical_config IS NULL"))
        await conn.execute(text(
            "UPDATE experiments SET git_commit = '' WHERE git_commit IS NULL"))
        await conn.execute(text(
            "UPDATE experiments SET environment = '{}' "
            "WHERE environment IS NULL"))
        await conn.execute(text(
            "UPDATE experiments SET task_dir = '' WHERE task_dir IS NULL"))
        await conn.commit()

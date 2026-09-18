import uuid
from sqlalchemy import create_engine, Column, String, Text, Integer, ForeignKey, text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector

# SQLAlchemy 선언적 Base 모델
Base = declarative_base()

class ParentChunk(Base):
    """
    'Small-to-Big Retrieval'에서 전체 문맥(Big)을 저장하는 부모 청크 모델
    """
    __tablename__ = 'parent_chunks'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    section_title = Column(String)
    content = Column(Text)

    # 자식 청크와의 1:N 관계 설정 (부모 삭제 시 자식도 삭제)
    children = relationship("ChildChunk", back_populates="parent", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<ParentChunk(id={self.id}, section_title='{self.section_title}')>"


class ChildChunk(Base):
    """
    실제 임베딩되어 검색(Small) 대상이 되는 자식 청크 모델
    """
    __tablename__ = 'child_chunks'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    parent_id = Column(UUID(as_uuid=True), ForeignKey('parent_chunks.id', ondelete='CASCADE'), nullable=False)
    content = Column(Text)
    page_start = Column(Integer)
    page_end = Column(Integer)
    
    # pgvector를 사용한 768차원 벡터 컬럼
    embedding = Column(Vector(768))

    # 부모 청크와의 관계 설정
    parent = relationship("ParentChunk", back_populates="children")

    def __repr__(self):
        return f"<ChildChunk(id={self.id}, parent_id={self.parent_id})>"


def init_db(db_url: str):
    """
    데이터베이스 연결을 초기화하고 pgvector 익스텐션과 테이블을 생성합니다.
    
    Args:
        db_url (str): 데이터베이스 연결 URL (예: 'postgresql://user:password@localhost:5432/dbname')
        
    Returns:
        tuple: (Engine, SessionMaker) 객체
    """
    # 동기식(sync) 엔진 생성
    engine = create_engine(db_url, echo=False)
    
    # 1. pgvector 익스텐션 활성화
    # 확장 기능 설치는 트랜잭션 내에서 실행해야 하므로 engine.begin() 사용
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    # 2. 모든 테이블 생성 (이미 존재하면 무시)
    Base.metadata.create_all(engine)
    
    # 3. 세션 팩토리 생성 및 반환
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    return engine, SessionLocal

if __name__ == "__main__":
    pass

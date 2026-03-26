from sqlalchemy import Column, Integer, String, Text
from app.core.database import Base

class Incident(Base):
    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, index=True)
    issue = Column(Text)
    category = Column(String)
    recommendation = Column(Text)
    status = Column(String)
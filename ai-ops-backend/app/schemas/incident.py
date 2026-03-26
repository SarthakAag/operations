from pydantic import BaseModel

class IncidentCreate(BaseModel):
    cpu: int
    text: str
    error: str

class IncidentResponse(BaseModel):
    anomaly: str
    category: str
    recommendation: str
    incident_id: int
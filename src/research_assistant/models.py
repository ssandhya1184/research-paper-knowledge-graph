
from pydantic import BaseModel, Field


class ResearchEntity(BaseModel):
    name: str
    entity_type: str = Field(
        description="One of: Method, Dataset, Metric, Concept, Finding"
    )
    description: str = ""


class ResearchRelationship(BaseModel):
    source: str
    relation: str
    target: str


class ResearchExtraction(BaseModel):
    methods: list[ResearchEntity] = Field(default_factory=list)
    datasets: list[ResearchEntity] = Field(default_factory=list)
    metrics: list[ResearchEntity] = Field(default_factory=list)
    concepts: list[ResearchEntity] = Field(default_factory=list)
    findings: list[ResearchEntity] = Field(default_factory=list)
    relationships: list[ResearchRelationship] = Field(default_factory=list)

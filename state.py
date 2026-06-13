"""The shared "notebook" for the agent graph.

`ResearchState` is the LangGraph state passed between nodes — each agent reads
the query/document and writes into its own `*_notes` slot, so the three research
agents can run in parallel without clobbering each other. `ResearchReport` is the
structured schema the Synthesis agent is forced to fill in.
"""
from typing import List, TypedDict

from pydantic import BaseModel, Field


class DataPoint(BaseModel):
    """A single headline stat for the dashboard metric cards."""

    label: str = Field(description="Short name of the metric or fact, e.g. 'Claimed ARR'")
    value: str = Field(
        description="The value or short finding, e.g. '$2.5M', '40% MoM', or 'No peer-reviewed support'"
    )
    source: str = Field(
        description="Origin of this fact: one of 'document', 'web', or 'academic'"
    )
    source_ids: List[int] = Field(
        description=(
            "IDs from the numbered SOURCES list provided in the prompt that back this "
            "fact. Use only IDs that appear in that list. Empty list if it comes purely "
            "from the internal document or general reasoning."
        )
    )


class ScoreItem(BaseModel):
    """One dimension of the adaptive evaluation scorecard."""

    dimension: str = Field(
        description=(
            "A concise evaluation dimension chosen to fit THIS query (e.g. 'Market "
            "Opportunity', 'Technical Defensibility', 'Evidence Strength')."
        )
    )
    score: int = Field(description="Integer rating from 1 (poor) to 5 (excellent).")
    rationale: str = Field(description="One sentence justifying the score.")


class ResearchReport(BaseModel):
    """Strict output schema for the Synthesis agent — every field is required."""

    executive_summary: str = Field(
        description="2-4 sentence neutral overview answering the user's query."
    )
    key_data_points: List[DataPoint] = Field(
        description=(
            "3 to 5 of the most important facts or findings for this query — quantitative "
            "where the evidence is quantitative, but never fabricate numbers to fill a slot."
        )
    )
    sentiment_analysis: str = Field(
        description="What the market / public / press currently says (from web research)."
    )
    technical_reality: str = Field(
        description="What the academic & technical evidence supports or refutes."
    )
    contradictions_found: List[str] = Field(
        description=(
            "The crucial value-add: each item is one specific, named conflict in the "
            "evidence — internal document vs. external reality, document vs. document, or "
            "source vs. source / consensus vs. minority view. Include conspicuous absences "
            "of expected evidence. Empty list if none found."
        )
    )
    scorecard: List[ScoreItem] = Field(
        description=(
            "3-5 evaluation dimensions chosen to fit this specific query, each scored "
            "1-5 with a one-line rationale. Pick dimensions appropriate to the subject "
            "(a startup vs. a scientific claim need different rubrics)."
        )
    )
    bottom_line: str = Field(
        description=(
            "A direct, decision-oriented one-to-two sentence answer to the user's query — "
            "the single most important takeaway (a recommendation, a yes/no with its key "
            "reason, or the headline conclusion), grounded in the strongest evidence."
        )
    )
    confidence: str = Field(
        description="Your confidence in this assessment: exactly 'High', 'Medium', or 'Low'."
    )
    recommended_questions: List[str] = Field(
        description=(
            "3-5 sharp, specific questions a careful reviewer should investigate next to "
            "close the biggest remaining gaps or verify the most load-bearing claims."
        )
    )


class ResearchState(TypedDict, total=False):
    # Inputs
    user_query: str
    document_text: str          # raw extracted text from the uploaded file ("" if none)
    has_document: bool
    api_key: str
    model: str
    doc_name: str
    doc_names: list   # individual file names when multiple documents are provided
    # Agent outputs (each agent owns its own slots -> safe for parallel writes)
    rag_notes: str
    web_notes: str
    web_sources: list      # [{"title": str, "url": str}]
    academic_notes: str
    academic_sources: list  # [{"title": str, "url": str}]
    # Final
    final_report: dict

"""Pydantic schemas and data transfer objects for AI Hallucination Hunter.

Defines typed request/response models, enumeration types, and internal structured LLM
payloads with rigorous validation.
"""

from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, Field, field_validator


class VerdictEnum(str, Enum):
    """Verdict for an individual factual claim evaluated against supplied evidence.
    
    IMPORTANT: These verdicts evaluate alignment with the provided evidence ONLY,
    not universal objective reality or absolute truth.
    """
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIED = "UNVERIFIED"


class OverallStatusEnum(str, Enum):
    """High-level risk assessment status based on claim verdicts."""
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    HIGH_RISK = "HIGH_RISK"
    NO_CLAIMS_FOUND = "NO_CLAIMS_FOUND"


class EvidenceItem(BaseModel):
    """An individual piece of reference evidence supplied by the caller."""
    id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_\-\.:]+$",
        description="Unique identifier for the evidence item (alphanumeric, dash, underscore, dot, colon)",
    )
    text: str = Field(
        ...,
        min_length=1,
        description="The trusted reference text passage",
    )
    source: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Optional origin citation, URL, or document title",
    )

    @field_validator("id")
    @classmethod
    def validate_id_not_whitespace(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Evidence ID cannot be empty or whitespace only.")
        return stripped

    @field_validator("text")
    @classmethod
    def validate_text_not_whitespace(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Evidence text cannot be empty or whitespace only.")
        return stripped


class HuntRequest(BaseModel):
    """Request payload for hallucination detection."""
    question: Optional[str] = Field(
        default=None,
        max_length=5000,
        description="The prompt or question that elicited the candidate answer (optional context)",
    )
    candidate_answer: str = Field(
        ...,
        min_length=1,
        description="The AI-generated answer to evaluate for hallucinations",
    )
    evidence_items: List[EvidenceItem] = Field(
        ...,
        min_length=1,
        description="List of trusted reference evidence items to evaluate against",
    )

    @field_validator("candidate_answer")
    @classmethod
    def validate_answer_not_whitespace(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("candidate_answer cannot be empty or whitespace only.")
        return stripped


class ExtractedClaim(BaseModel):
    """An atomic factual claim extracted from the candidate answer."""
    claim_id: str = Field(..., description="Unique ID for the extracted claim, e.g. 'c1'")
    claim_text: str = Field(..., description="A discrete, verifiable factual assertion")


class ClaimEvaluation(BaseModel):
    """Evaluation result for a single claim against reference evidence."""
    claim_id: str = Field(..., description="The ID of the evaluated claim")
    claim_text: str = Field(..., description="The text of the factual claim evaluated")
    verdict: VerdictEnum = Field(..., description="Verdict: SUPPORTED, CONTRADICTED, or UNVERIFIED")
    supporting_evidence_ids: List[str] = Field(
        default_factory=list,
        description="IDs of supplied evidence items that substantiate the claim",
    )
    contradicting_evidence_ids: List[str] = Field(
        default_factory=list,
        description="IDs of supplied evidence items that refute the claim",
    )
    explanation: str = Field(
        ...,
        description="Clear reasoning explaining why this verdict was reached based on the evidence",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Model certainty in textual entailment/contradiction with provided evidence (0.0 to 1.0). "
            "NOTE: This measures semantic alignment with the supplied evidence, NOT absolute real-world truth."
        ),
    )


class HuntSummary(BaseModel):
    """High-level summary of the hallucination assessment."""
    total_claims: int = Field(..., ge=0, description="Total atomic factual claims extracted")
    supported_claims: int = Field(..., ge=0, description="Claims substantiated by provided evidence")
    contradicted_claims: int = Field(..., ge=0, description="Claims refuted by provided evidence")
    unverified_claims: int = Field(..., ge=0, description="Claims neither supported nor refuted by provided evidence")
    hallucination_risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Calculated composite risk score: (contradictions * 1.0 + unverified * 0.5) / total_claims",
    )
    status: OverallStatusEnum = Field(..., description="Risk category: SAFE, CAUTION, HIGH_RISK, or NO_CLAIMS_FOUND")
    explanation: str = Field(..., description="Executive summary of the evaluation findings")


class HuntResponse(BaseModel):
    """Full API response for POST /hunt."""
    request_id: str = Field(..., description="Unique UUID for this evaluation request")
    claims: List[ClaimEvaluation] = Field(..., description="Detailed evaluation for each claim")
    summary: HuntSummary = Field(..., description="Aggregated risk metrics and verdict summary")
    execution_time_ms: float = Field(..., ge=0.0, description="End-to-end execution latency in milliseconds")


class HealthResponse(BaseModel):
    """Response payload for GET /health."""
    status: str = Field(default="ok", description="Service health indicator")
    app_name: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    llm_provider_configured: bool = Field(..., description="Whether an LLM API key is present in environment")
    model: str = Field(..., description="Configured evaluation model")


class ErrorResponse(BaseModel):
    """Standardized error response body."""
    error: str = Field(..., description="High-level error classification")
    detail: Optional[Any] = Field(default=None, description="Detailed error information or field breakdown")


# Internal schemas for LLM structured output parsing
class ClaimExtractionLLMOutput(BaseModel):
    """Schema expected from the LLM when extracting claims."""
    claims: List[ExtractedClaim] = Field(default_factory=list)


class ClaimVerificationLLMItem(BaseModel):
    """Single claim evaluation schema from the LLM."""
    claim_id: str
    verdict: VerdictEnum
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    contradicting_evidence_ids: List[str] = Field(default_factory=list)
    explanation: str
    confidence_score: float = Field(ge=0.0, le=1.0)


class ClaimVerificationLLMOutput(BaseModel):
    """Schema expected from the LLM when verifying claims."""
    evaluations: List[ClaimVerificationLLMItem] = Field(default_factory=list)


class SecondPassVerificationItem(BaseModel):
    """Schema for individual claim-evidence pair verification from second-pass LLM."""
    claim_id: str
    evidence_id: str
    verdict_confirmed: bool
    quote: Optional[str] = None
    reasoning: str


class SecondPassLLMOutput(BaseModel):
    """Schema expected from second-pass entailment verification."""
    verifications: List[SecondPassVerificationItem] = Field(default_factory=list)

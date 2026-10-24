"""Hallucination detection orchestrator.

Coordinates input validation, atomic claim extraction, evidence alignment verification,
citation integrity auditing, and risk scoring aggregation into a structured report.
"""

import time
import uuid
from typing import Dict, List, Optional, Set, Tuple

from ai_engine import AIEngine
from config import Settings
from evidence import EvidenceManager, EvidenceValidationError
from schemas import (
    ClaimEvaluation,
    ClaimVerificationLLMItem,
    ExtractedClaim,
    HuntRequest,
    HuntResponse,
    HuntSummary,
    OverallStatusEnum,
    SecondPassVerificationItem,
    VerdictEnum,
)


class InputValidationError(ValueError):
    """Raised when request payload violates size limits or input integrity."""
    pass


class HallucinationDetector:
    """End-to-end orchestrator for factual claim verification and hallucination risk detection."""

    def __init__(self, ai_engine: AIEngine, settings: Settings):
        self.ai_engine = ai_engine
        self.settings = settings

    def _validate_input_limits(self, request: HuntRequest) -> None:
        """Validate input length constraints to guard against resource exhaustion."""
        answer_len = len(request.candidate_answer)
        if answer_len > self.settings.max_candidate_answer_chars:
            raise InputValidationError(
                f"Candidate answer exceeds character limit: {answer_len} > "
                f"{self.settings.max_candidate_answer_chars} chars."
            )

        question_len = len(request.question) if request.question else 0
        total_evidence_chars = sum(len(item.text) for item in request.evidence_items)
        total_chars = answer_len + question_len + total_evidence_chars

        if total_chars > self.settings.max_total_chars:
            raise InputValidationError(
                f"Total request payload exceeds maximum character threshold: "
                f"{total_chars} > {self.settings.max_total_chars} chars."
            )

    def _audit_citations_and_align(
        self,
        extracted_claims: List[ExtractedClaim],
        raw_evaluations: List[ClaimVerificationLLMItem],
        evidence_manager: EvidenceManager,
    ) -> List[ClaimEvaluation]:
        """Audit LLM output to prevent citation hallucination and enforce schema integrity.
        
        Guarantees that:
        1. All extracted claims are represented.
        2. Cited evidence IDs strictly belong to the supplied evidence set.
        3. A claim cannot claim 'SUPPORTED' or 'CONTRADICTED' if it cites zero valid evidence items.
        4. Conflicting citations (both supporting and contradicting) downgrade to UNVERIFIED with confidence 0.0.
        5. Duplicate evaluations for the same claim ID are reconciled without silent overwrite.
        """
        # Group evaluations by claim_id to detect and handle duplicates safely
        evals_by_id: Dict[str, List[ClaimVerificationLLMItem]] = {}
        for item in raw_evaluations:
            evals_by_id.setdefault(item.claim_id, []).append(item)

        audited_evaluations: List[ClaimEvaluation] = []

        for claim in extracted_claims:
            eval_list = evals_by_id.get(claim.claim_id, [])

            if not eval_list:
                # LLM omitted this claim - mark as UNVERIFIED with clear explanation
                audited_evaluations.append(
                    ClaimEvaluation(
                        claim_id=claim.claim_id,
                        claim_text=claim.claim_text,
                        verdict=VerdictEnum.UNVERIFIED,
                        supporting_evidence_ids=[],
                        contradicting_evidence_ids=[],
                        explanation="Evaluator model did not return an assessment for this claim.",
                        confidence_score=0.0,
                    )
                )
                continue

            # Check for duplicate evaluations for the same claim
            if len(eval_list) > 1:
                verdicts = {e.verdict for e in eval_list}
                if len(verdicts) > 1:
                    # Conflicting verdicts returned for the exact same claim
                    audited_evaluations.append(
                        ClaimEvaluation(
                            claim_id=claim.claim_id,
                            claim_text=claim.claim_text,
                            verdict=VerdictEnum.UNVERIFIED,
                            supporting_evidence_ids=[],
                            contradicting_evidence_ids=[],
                            explanation=(
                                f"[Evaluation Audit: Model returned multiple conflicting evaluations "
                                f"({sorted([v.value for v in verdicts])}) for this claim. Degraded to UNVERIFIED.]"
                            ),
                            confidence_score=0.0,
                        )
                    )
                    continue
                # If duplicates agree on verdict, pick the first
                raw_eval = eval_list[0]
            else:
                raw_eval = eval_list[0]

            # Verify and sanitize citations against actual evidence store
            valid_supporting, invalid_supporting = evidence_manager.filter_valid_ids(
                raw_eval.supporting_evidence_ids
            )
            valid_contradicting, invalid_contradicting = evidence_manager.filter_valid_ids(
                raw_eval.contradicting_evidence_ids
            )

            verdict = raw_eval.verdict
            explanation = raw_eval.explanation
            confidence_score = raw_eval.confidence_score

            # Safeguard 1: Conflicting citations (model cited both supporting and contradicting evidence)
            if valid_supporting and valid_contradicting:
                verdict = VerdictEnum.UNVERIFIED
                confidence_score = 0.0
                explanation = (
                    f"[Citation Audit: Claim contains conflicting citations (both supporting: {valid_supporting} "
                    f"and contradicting: {valid_contradicting}). Degraded to UNVERIFIED.]"
                )
                valid_supporting = []
                valid_contradicting = []

            # Safeguard 2: If model claimed SUPPORTED without valid evidence, degrade to UNVERIFIED
            elif verdict == VerdictEnum.SUPPORTED and not valid_supporting:
                verdict = VerdictEnum.UNVERIFIED
                confidence_score = 0.0
                explanation = (
                    f"[Citation Audit: Evaluator marked claim as SUPPORTED but cited no valid evidence IDs "
                    f"(invalid citations stripped: {invalid_supporting}). Degraded to UNVERIFIED.]"
                )

            # Safeguard 3: If model claimed CONTRADICTED without valid evidence, degrade to UNVERIFIED
            elif verdict == VerdictEnum.CONTRADICTED and not valid_contradicting:
                verdict = VerdictEnum.UNVERIFIED
                confidence_score = 0.0
                explanation = (
                    f"[Citation Audit: Evaluator marked claim as CONTRADICTED but cited no valid evidence IDs "
                    f"(invalid citations stripped: {invalid_contradicting}). Degraded to UNVERIFIED.]"
                )

            elif verdict == VerdictEnum.UNVERIFIED:
                # An unverified claim by definition has no valid supporting evidence
                valid_supporting = []
                valid_contradicting = []

            audited_evaluations.append(
                ClaimEvaluation(
                    claim_id=claim.claim_id,
                    claim_text=claim.claim_text,
                    verdict=verdict,
                    supporting_evidence_ids=valid_supporting,
                    contradicting_evidence_ids=valid_contradicting,
                    explanation=explanation,
                    confidence_score=confidence_score,
                )
            )

        return audited_evaluations

    def _compute_summary(self, evaluations: List[ClaimEvaluation]) -> HuntSummary:
        """Aggregate evaluations into a composite risk score and qualitative status."""
        total = len(evaluations)
        if total == 0:
            return HuntSummary(
                total_claims=0,
                supported_claims=0,
                contradicted_claims=0,
                unverified_claims=0,
                hallucination_risk_score=0.0,
                status=OverallStatusEnum.NO_CLAIMS_FOUND,
                explanation="No verifiable factual claims were found in the candidate answer.",
            )

        supported = sum(1 for c in evaluations if c.verdict == VerdictEnum.SUPPORTED)
        contradicted = sum(1 for c in evaluations if c.verdict == VerdictEnum.CONTRADICTED)
        unverified = sum(1 for c in evaluations if c.verdict == VerdictEnum.UNVERIFIED)

        # Composite risk formula:
        # Contradictions are severe errors (weight 1.0)
        # Unverified claims are ungrounded risk (weight 0.5)
        # Supported claims contribute 0 risk
        risk_score = round(((contradicted * 1.0) + (unverified * 0.5)) / total, 4)
        unverified_ratio = unverified / total

        # Status categorization:
        # Any contradiction, risk >= 0.6, or >80% unverified claims triggers HIGH_RISK
        if contradicted > 0 or risk_score >= 0.6 or unverified_ratio > 0.8:
            status = OverallStatusEnum.HIGH_RISK
            if unverified_ratio > 0.8 and contradicted == 0:
                explanation = (
                    f"High hallucination risk detected. {unverified} of {total} claims "
                    f"({int(unverified_ratio * 100)}%) could not be verified against the reference evidence."
                )
            else:
                explanation = (
                    f"High hallucination risk detected. {contradicted} of {total} claims contradicted "
                    f"by reference evidence, and {unverified} unverified."
                )
        elif unverified > 0 or risk_score > 0.2:
            status = OverallStatusEnum.CAUTION
            explanation = (
                f"Caution advised. {supported} of {total} claims supported, but {unverified} claims "
                "could not be verified against the provided evidence."
            )
        else:
            status = OverallStatusEnum.SAFE
            explanation = (
                f"All {total} factual claims are supported by the provided reference evidence."
            )

        return HuntSummary(
            total_claims=total,
            supported_claims=supported,
            contradicted_claims=contradicted,
            unverified_claims=unverified,
            hallucination_risk_score=risk_score,
            status=status,
            explanation=explanation,
        )

    async def _reverify_semantic_entailment(
        self,
        audited_claims: List[ClaimEvaluation],
        evidence_manager: EvidenceManager,
    ) -> List[ClaimEvaluation]:
        """Perform second-pass adversarial entailment reverification on claims marked SUPPORTED/CONTRADICTED.
        
        Reverifies that the cited passage strictly entails the claim and verifies exact substring quotes.
        Fails safely to UNVERIFIED if verification disagrees, times out, or errors.
        """
        claims_to_reverify = [
            c for c in audited_claims
            if c.verdict in (VerdictEnum.SUPPORTED, VerdictEnum.CONTRADICTED)
        ]

        if not claims_to_reverify:
            return audited_claims

        # Prepare reverification targets
        targets = []
        target_keys: Set[Tuple[str, str]] = set()
        for claim in claims_to_reverify:
            cited_ids = (
                claim.supporting_evidence_ids
                if claim.verdict == VerdictEnum.SUPPORTED
                else claim.contradicting_evidence_ids
            )
            # Deduplicate citations for reverification
            for ev_id in dict.fromkeys(cited_ids):
                ev_item = evidence_manager.get(ev_id)
                if ev_item and (claim.claim_id, ev_id) not in target_keys:
                    target_keys.add((claim.claim_id, ev_id))
                    targets.append({
                        "claim_id": claim.claim_id,
                        "claim_text": claim.claim_text,
                        "evidence_id": ev_id,
                        "evidence_text": ev_item.text,
                        "verdict_to_confirm": claim.verdict.value,
                    })

        if not targets:
            return audited_claims

        # Execute second-pass call with safe fallback on error, timeout, or malformed response
        second_pass_error: Optional[str] = None
        second_pass_map: Dict[Tuple[str, str], SecondPassVerificationItem] = {}
        try:
            results = await self.ai_engine.reverify_claims_entailment(targets)
            for r in results:
                key = (r.claim_id, r.evidence_id)
                if key in second_pass_map:
                    # If duplicate results conflict, safely mark unconfirmed
                    if second_pass_map[key].verdict_confirmed != r.verdict_confirmed:
                        second_pass_map[key].verdict_confirmed = False
                        second_pass_map[key].reasoning = "Conflicting duplicate second-pass verification results."
                else:
                    second_pass_map[key] = r
        except Exception as exc:
            second_pass_error = str(exc)

        # Audit and apply second-pass verification results
        for claim in audited_claims:
            if claim.verdict not in (VerdictEnum.SUPPORTED, VerdictEnum.CONTRADICTED):
                continue

            # Fail-safe: if second-pass timed out, failed, or errored, downgrade to UNVERIFIED
            if second_pass_error is not None:
                claim.verdict = VerdictEnum.UNVERIFIED
                claim.confidence_score = 0.0
                claim.supporting_evidence_ids = []
                claim.contradicting_evidence_ids = []
                claim.explanation = (
                    f"[Semantic Audit Fail-Safe: Second-pass entailment verification failed ({second_pass_error}). "
                    f"Safely degraded to UNVERIFIED.] {claim.explanation}"
                )
                continue

            cited_ids = (
                claim.supporting_evidence_ids
                if claim.verdict == VerdictEnum.SUPPORTED
                else claim.contradicting_evidence_ids
            )

            all_confirmed = True
            failure_reasons = []

            for ev_id in cited_ids:
                verif = second_pass_map.get((claim.claim_id, ev_id))
                ev_item = evidence_manager.get(ev_id)

                if verif is None:
                    all_confirmed = False
                    failure_reasons.append(f"No second-pass response for evidence '{ev_id}'")
                    break

                if not verif.verdict_confirmed:
                    all_confirmed = False
                    failure_reasons.append(f"Evidence '{ev_id}' does not strictly substantiate verdict ({verif.reasoning})")
                    break

                # Literal substring quote verification
                if verif.quote:
                    clean_quote = verif.quote.strip()
                    if clean_quote and ev_item and clean_quote not in ev_item.text:
                        all_confirmed = False
                        failure_reasons.append(f"Cited quote was not found as a literal substring in evidence '{ev_id}'")
                        break

            if not all_confirmed:
                # Downgrade claim to UNVERIFIED
                claim.verdict = VerdictEnum.UNVERIFIED
                claim.confidence_score = 0.0
                claim.supporting_evidence_ids = []
                claim.contradicting_evidence_ids = []
                claim.explanation = (
                    f"[Semantic Audit: Second-pass reverification rejected evidence entailment: "
                    f"{'; '.join(failure_reasons)}. Degraded to UNVERIFIED.]"
                )
            else:
                claim.explanation = (
                    f"{claim.explanation} "
                    f"[Semantic Audit: Second-pass entailment confirmed against cited evidence; "
                    f"source quotes verified as literal substrings. Note: Substring matching alone does not prove universal entailment.]"
                )

        return audited_claims

    async def hunt(self, request: HuntRequest) -> HuntResponse:
        """Execute the end-to-end hallucination detection pipeline."""
        start_time = time.perf_counter()
        request_id = str(uuid.uuid4())

        # Step 1: Pre-flight input size validation
        self._validate_input_limits(request)

        # Step 2: Ingest and validate evidence
        evidence_mgr = EvidenceManager(request.evidence_items, self.settings)

        # Step 3: Extract atomic factual claims
        extracted_claims = await self.ai_engine.extract_claims(
            candidate_answer=request.candidate_answer,
            question=request.question,
        )

        # Step 4: Handle answers with no factual claims
        if not extracted_claims:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            summary = self._compute_summary([])
            return HuntResponse(
                request_id=request_id,
                claims=[],
                summary=summary,
                execution_time_ms=elapsed_ms,
            )

        # Step 5: Verify claims against evidence
        raw_evaluations = await self.ai_engine.verify_claims(
            claims=extracted_claims,
            evidence_manager=evidence_mgr,
        )

        # Step 6: Audit citations and prevent fabricated references
        audited_claims = self._audit_citations_and_align(
            extracted_claims=extracted_claims,
            raw_evaluations=raw_evaluations,
            evidence_manager=evidence_mgr,
        )

        # Step 7: Second-pass semantic entailment reverification
        reverified_claims = await self._reverify_semantic_entailment(
            audited_claims=audited_claims,
            evidence_manager=evidence_mgr,
        )

        # Step 8: Compute aggregate summary
        summary = self._compute_summary(reverified_claims)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return HuntResponse(
            request_id=request_id,
            claims=reverified_claims,
            summary=summary,
            execution_time_ms=elapsed_ms,
        )

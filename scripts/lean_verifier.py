"""Lean 4 proof verifier for open-instruct.

Adds a LeanVerifier that compiles Lean 4 proofs via a sandbox container
(NeMo-Skills compatible: /execute endpoint on configurable host:port).

To register, copy this file into open_instruct/ and import it from ground_truth_utils.py:
    from open_instruct.lean_verifier import LeanVerifier  # noqa: F401

The verifier will then be auto-discovered by build_all_verifiers().
"""

import asyncio
import json
import logging
import re
from typing import Any

import httpx

from open_instruct.ground_truth_utils import (
    CodeVerifierConfig,
    VerificationResult,
    VerifierFunction,
)

logger = logging.getLogger(__name__)


def _extract_lean_code(text: str) -> str:
    """Extract the last lean/lean4 code block from markdown output."""
    for lang in ["lean4", "lean3", "lean", ""]:
        matches = re.findall(rf"```{lang}\s*\n?(.*?)\n?```", text, re.DOTALL)
        if matches:
            return matches[-1].strip()
    return text.strip()


def _extract_proof_only(lean_code: str) -> str:
    """Strip theorem header, keeping only the proof body after ':='."""
    lines = lean_code.strip().splitlines()
    if not lines:
        return ""

    header_pat = re.compile(r"^\s*(theorem|example)\b")
    header_idx = next((i for i, l in enumerate(lines) if header_pat.match(l)), None)
    if header_idx is None:
        return lean_code.strip()

    assign_idx = next(
        (i for i in range(header_idx, len(lines)) if ":=" in lines[i]), None
    )
    if assign_idx is None:
        return lean_code.strip()

    _, after = lines[assign_idx].split(":=", 1)
    proof_first = after.strip()
    proof_lines = ([proof_first] if proof_first else []) + lines[assign_idx + 1 :]

    if proof_lines:
        first = proof_lines[0].lstrip()
        if first == "by":
            proof_lines = proof_lines[1:]
        elif first.startswith("by "):
            proof_lines[0] = first[3:]

    return "\n".join(proof_lines).rstrip()


class LeanVerifier(VerifierFunction):
    """Verify Lean 4 proofs by compiling them in a sandbox container.

    The ground truth (label) is a JSON dict with keys:
      - header: Lean 4 import/setup code
      - formal_statement: the theorem statement (ending with ':= by')

    The model output should contain Lean 4 proof tactics in a code block.
    Reward is 1.0 if compilation succeeds (no errors, no sorry), else 0.0.
    """

    _client: httpx.AsyncClient | None = None

    def __init__(self, verifier_config: CodeVerifierConfig) -> None:
        super().__init__("lean", verifier_config=verifier_config, weight=1.0)
        self._sandbox_url = verifier_config.code_api_url
        self._timeout = verifier_config.code_max_execution_time

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                limits=httpx.Limits(max_keepalive_connections=50, max_connections=100),
            )
        return self._client

    async def async_call(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: str,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        try:
            gt = json.loads(label) if isinstance(label, str) else label
        except (json.JSONDecodeError, TypeError):
            gt = {"header": "", "formal_statement": label}

        header = gt.get("header", gt.get("lean_header", ""))
        formal_statement = gt.get("formal_statement", "")

        cleaned = _extract_lean_code(prediction)
        proof_body = _extract_proof_only(cleaned)
        full_code = header + formal_statement + proof_body

        request_data = {
            "generated_code": full_code,
            "language": "lean4",
            "timeout": self._timeout,
        }

        try:
            client = self._get_client()
            resp = await asyncio.to_thread(
                lambda: httpx.post(
                    self._sandbox_url,
                    content=json.dumps(request_data),
                    timeout=self._timeout + 10.0,
                    headers={"Content-Type": "application/json"},
                )
            )
            result = resp.json()
        except Exception as e:
            logger.warning("Lean sandbox request failed: %s", e)
            return VerificationResult(score=0.0)

        process_status = result.get("process_status", "unknown")
        if process_status == "timeout":
            return VerificationResult(score=0.0)
        if process_status != "completed":
            return VerificationResult(score=0.0)

        stdout = result.get("stdout", "").lower()
        stderr = result.get("stderr", "").lower()
        combined = stdout + "\n" + stderr

        if re.search(r"\bsorry\b", combined):
            return VerificationResult(score=0.0)

        return VerificationResult(score=1.0)

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: str,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        return asyncio.run(
            self.async_call(tokenized_prediction, prediction, label, query, rollout_state)
        )

    @classmethod
    def get_config_class(cls) -> type:
        return CodeVerifierConfig

# Phase 3 report

## Implemented

- Failure extraction: retrieval / planning / evidence / reasoning / tool / verification / answer_synthesis
- Embedding KMeans clustering when an embedder is attached; type-based grouping otherwise
- Capability scores, uncertainty (model disagreement + error rate), regressions vs previous profile
- Frontier bands: mastered / learning / frontier / unknown / regressed
- Frontier score components: learnability, novelty, transfer_potential, uncertainty, cost

## Commands

```bash
pytest tests/unit/test_policies.py::test_diagnosis_from_real_experiences -q
```

Diagnosis over Phase 1 `experiences.jsonl` (after the 50+ run):

```python
from pathlib import Path
import orjson
from expolicy.core.experience import Experience
from expolicy.diagnosis.capability_profile import CapabilityDiagnoser
rows = [Experience.model_validate(orjson.loads(l)) for l in Path("outputs/phase1_static/experiences.jsonl").read_text().splitlines() if l.strip()]
print(CapabilityDiagnoser().diagnose(rows).model_dump())
```

## Next phase

Phase 4: passive experiential learning baselines (memory + weight internalization).

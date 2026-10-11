"""Deterministic Helion Archive: a closed, fully searchable research world.

Documents and tasks are generated from a hidden knowledge graph. Retrieval is
real BM25 over the materialized corpus; answers are never injected into the
agent. Held-out splits are assigned by hashing task ids.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from expolicy.core.types import CapabilityName, DataSplit

CAPABILITIES: tuple[str, ...] = tuple(item.value for item in CapabilityName)

STAGE_CAPABILITIES: dict[int, tuple[str, ...]] = {
    0: (CapabilityName.SINGLE_HOP.value,),
    1: (CapabilityName.MULTI_HOP.value, CapabilityName.QUERY_DECOMPOSITION.value),
    2: (CapabilityName.ENTITY_DISAMBIGUATION.value, CapabilityName.TEMPORAL.value),
    3: (CapabilityName.CONFLICTING_EVIDENCE.value, CapabilityName.SOURCE_RELIABILITY.value),
    4: (
        CapabilityName.LONG_CONTEXT.value,
        CapabilityName.CROSS_DOCUMENT.value,
        CapabilityName.FAILURE_RECOVERY.value,
    ),
}

CITIES = [
    "Aurel",
    "Nexora",
    "Kaldis",
    "Viremont",
    "Osthaven",
    "Brinwall",
    "Selwick",
    "Tornholm",
    "Greyfen",
    "Lakeshore",
    "Highmere",
    "Dunwick",
]

SECTORS = [
    "energy",
    "biotech",
    "shipping",
    "media",
    "computing",
    "materials",
    "finance",
    "aerospace",
]

FIRST = [
    "Lira",
    "Kael",
    "Mira",
    "Oren",
    "Sera",
    "Daven",
    "Nyx",
    "Tomas",
    "Elia",
    "Rook",
    "Ivo",
    "Neris",
    "Pax",
    "Yara",
    "Helo",
    "Quin",
]
LAST = [
    "Venn",
    "Calder",
    "Hale",
    "Ost",
    "Quill",
    "Marrow",
    "Fen",
    "Bright",
    "Solis",
    "Kade",
    "Rowe",
    "Ash",
    "Pell",
    "Wynn",
    "Tor",
    "Nash",
]


def _stable_split(task_id: str, seed: int) -> DataSplit:
    digest = hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 40:
        return DataSplit.DEPLOYMENT
    if bucket < 60:
        return DataSplit.ACQUISITION
    if bucket < 75:
        return DataSplit.VALIDATION
    if bucket < 90:
        return DataSplit.FUTURE
    return DataSplit.RETENTION


@dataclass
class Person:
    person_id: str
    name: str
    birth_year: int
    birth_city: str
    org_id: str
    role: str
    advisor_id: str | None
    specialty: str


@dataclass
class Org:
    org_id: str
    name: str
    hq_city: str
    founded: int
    sector: str
    ceo_id: str
    parent_id: str | None
    product: str


@dataclass
class Event:
    event_id: str
    name: str
    year: int
    month: int
    city: str
    org_id: str
    lead_id: str
    outcome: str


@dataclass
class Document:
    document_id: str
    title: str
    text: str
    source_type: str
    reliability: float
    year: int
    city: str
    fact_ids: list[str] = field(default_factory=list)
    is_distractor: bool = False


@dataclass
class Task:
    task_id: str
    question: str
    gold_answer: str
    capability: str
    stage: int
    gold_doc_ids: list[str]
    distractor_doc_ids: list[str]
    hops: int
    split: str
    aliases: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class World:
    seed: int
    people: dict[str, Person]
    orgs: dict[str, Org]
    events: dict[str, Event]
    documents: dict[str, Document]
    tasks: dict[str, Task]
    facts: dict[str, dict[str, Any]]

    def to_serializable(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "people": {k: asdict(v) for k, v in self.people.items()},
            "orgs": {k: asdict(v) for k, v in self.orgs.items()},
            "events": {k: asdict(v) for k, v in self.events.items()},
            "documents": {k: asdict(v) for k, v in self.documents.items()},
            "tasks": {k: asdict(v) for k, v in self.tasks.items()},
            "facts": self.facts,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_serializable()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> World:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            seed=payload["seed"],
            people={k: Person(**v) for k, v in payload["people"].items()},
            orgs={k: Org(**v) for k, v in payload["orgs"].items()},
            events={k: Event(**v) for k, v in payload["events"].items()},
            documents={k: Document(**v) for k, v in payload["documents"].items()},
            tasks={k: Task(**v) for k, v in payload["tasks"].items()},
            facts=payload["facts"],
        )


def _name_at(i: int) -> str:
    return f"{FIRST[i % len(FIRST)]} {LAST[(i * 7) % len(LAST)]}"


def _unique_names(n: int) -> list[str]:
    names: list[str] = []
    for i in range(n):
        names.append(f"{FIRST[i % len(FIRST)]} {LAST[i // len(FIRST) % len(LAST)]}")
    if len(set(names)) != n:
        mids = "ABCDEFGHJKLMNPQRSTUVWXYZ"
        names = [
            f"{FIRST[i % len(FIRST)]} {mids[i % len(mids)]}. {LAST[i // len(FIRST) % len(LAST)]}"
            for i in range(n)
        ]
    if len(set(names)) != n:
        raise ValueError("name generator produced collisions")
    return names


def build_world(seed: int = 13, *, n_people: int = 72, n_orgs: int = 24, n_events: int = 36) -> World:
    rng = random.Random(seed)
    names = _unique_names(n_people)
    people: dict[str, Person] = {}
    orgs: dict[str, Org] = {}
    events: dict[str, Event] = {}
    facts: dict[str, dict[str, Any]] = {}

    org_names = [
        f"{city} {suffix}"
        for city, suffix in zip(
            CITIES + CITIES,
            [
                "Institute",
                "Dynamics",
                "Press",
                "University",
                "Shipping",
                "Energy",
                "Labs",
                "Capital",
                "Foundry",
                "Systems",
                "Archive",
                "Holdings",
                "Observatory",
                "Works",
                "Consortium",
                "Bureau",
                "Relay",
                "Clinic",
                "Yard",
                "Exchange",
                "Forge",
                "Ledger",
                "Atlas",
                "Circuit",
            ],
        )
    ][:n_orgs]

    for i, org_name in enumerate(org_names):
        oid = f"org_{i:03d}"
        orgs[oid] = Org(
            org_id=oid,
            name=org_name,
            hq_city=CITIES[i % len(CITIES)],
            founded=1988 + (i * 3) % 30,
            sector=SECTORS[i % len(SECTORS)],
            ceo_id="",
            parent_id=f"org_{i - 5:03d}" if i >= 5 and i % 4 == 0 else None,
            product=f"{org_name.split()[0]}-{['Helix', 'Nimbus', 'Keel', 'Quartz', 'Ion', 'Loom'][i % 6]}-{100 + i}",
        )

    roles = ["researcher", "engineer", "archivist", "director", "correspondent", "pilot"]
    specialties = [
        "ion propulsion",
        "coral genomics",
        "archive law",
        "grid storage",
        "optical computing",
        "glacier mapping",
        "port logistics",
        "plasma ceramics",
    ]
    for i, name in enumerate(names):
        pid = f"per_{i:03d}"
        org_id = f"org_{i % n_orgs:03d}"
        advisor = f"per_{max(0, i - 11):03d}" if i > 12 else None
        people[pid] = Person(
            person_id=pid,
            name=name,
            birth_year=1955 + (i * 5) % 45,
            birth_city=CITIES[(i * 3) % len(CITIES)],
            org_id=org_id,
            role=roles[i % len(roles)],
            advisor_id=advisor,
            specialty=specialties[i % len(specialties)],
        )

    org_list = list(orgs.values())
    person_list = list(people.values())
    for i, org in enumerate(org_list):
        org.ceo_id = person_list[i * 3 % len(person_list)].person_id

    outcomes = [
        "signed a five-year research compact",
        "opened a public data vault",
        "cancelled a coastal expansion",
        "announced a reactor retrofit",
        "released a corrected measurement series",
        "merged two field offices",
    ]
    for i in range(n_events):
        eid = f"evt_{i:03d}"
        org = org_list[i % len(org_list)]
        lead = person_list[(i * 5) % len(person_list)]
        events[eid] = Event(
            event_id=eid,
            name=f"{org.hq_city} {['Accord', 'Summit', 'Survey', 'Retrofit', 'Inquiry'][i % 5]} {2016 + i % 10}",
            year=2016 + i % 10,
            month=1 + (i * 2) % 12,
            city=CITIES[(i * 2) % len(CITIES)],
            org_id=org.org_id,
            lead_id=lead.person_id,
            outcome=outcomes[i % len(outcomes)],
        )

    documents: dict[str, Document] = {}
    doc_n = 0

    def add_doc(
        title: str,
        paragraphs: list[str],
        *,
        source_type: str,
        reliability: float,
        year: int,
        city: str,
        fact_ids: list[str],
        distractor: bool = False,
    ) -> str:
        nonlocal doc_n
        did = f"doc_{doc_n:04d}"
        doc_n += 1
        filler = (
            f" Local observers in {city} recorded routine harbor traffic the same week. "
            f"The Helion Archive indexes this file under {source_type} holdings for {year}."
        )
        text = " ".join(paragraphs) + filler
        documents[did] = Document(
            document_id=did,
            title=title,
            text=text,
            source_type=source_type,
            reliability=reliability,
            year=year,
            city=city,
            fact_ids=fact_ids,
            is_distractor=distractor,
        )
        return did

    person_docs: dict[str, str] = {}
    org_docs: dict[str, str] = {}
    event_docs: dict[str, str] = {}
    rumor_docs: dict[str, str] = {}

    for person in people.values():
        org = orgs[person.org_id]
        fid = f"fact_{person.person_id}"
        facts[fid] = {
            "kind": "person",
            "person_id": person.person_id,
            "name": person.name,
            "org": org.name,
            "role": person.role,
            "city": org.hq_city,
            "birth_city": person.birth_city,
            "birth_year": person.birth_year,
            "specialty": person.specialty,
            "advisor": people[person.advisor_id].name if person.advisor_id else None,
        }
        did = add_doc(
            f"Biographical note: {person.name}",
            [
                f"{person.name} was born in {person.birth_year} in {person.birth_city}.",
                f"{person.name} works as a {person.role} at {org.name}.",
                f"{person.name}'s research specialty is {person.specialty}.",
                (
                    f"{person.name} was advised by {people[person.advisor_id].name}."
                    if person.advisor_id
                    else f"{person.name} has no recorded doctoral advisor in the archive."
                ),
            ],
            source_type="encyclopedia",
            reliability=0.95,
            year=2024,
            city=person.birth_city,
            fact_ids=[fid],
        )
        person_docs[person.person_id] = did

        if rng.random() < 0.35:
            wrong_city = CITIES[(CITIES.index(org.hq_city) + 4) % len(CITIES)]
            rid = add_doc(
                f"Unsigned blog: gossip about {person.name}",
                [
                    f"A travel blog claims {person.name} actually works from {wrong_city}, not {org.hq_city}.",
                    "The post cites no registry and mixes two similarly named contractors.",
                ],
                source_type="blog",
                reliability=0.25,
                year=2023,
                city=wrong_city,
                fact_ids=[],
                distractor=True,
            )
            rumor_docs[person.person_id] = rid

    for org in orgs.values():
        ceo = people[org.ceo_id]
        fid = f"fact_{org.org_id}"
        facts[fid] = {
            "kind": "org",
            "org_id": org.org_id,
            "name": org.name,
            "hq": org.hq_city,
            "founded": org.founded,
            "sector": org.sector,
            "ceo": ceo.name,
            "product": org.product,
            "parent": orgs[org.parent_id].name if org.parent_id else None,
        }
        did = add_doc(
            f"Registry entry: {org.name}",
            [
                f"{org.name} is a {org.sector} organization headquartered in {org.hq_city}.",
                f"The organization was founded in {org.founded}.",
                f"The current chief executive is {ceo.name}.",
                f"Its flagship product is {org.product}.",
                (
                    f"{org.name} is a subsidiary of {orgs[org.parent_id].name}."
                    if org.parent_id
                    else f"{org.name} is an independent organization with no parent company."
                ),
            ],
            source_type="official",
            reliability=0.98,
            year=2025,
            city=org.hq_city,
            fact_ids=[fid],
        )
        org_docs[org.org_id] = did
        add_doc(
            f"{org.name} campus notes",
            [
                f"Staff at the {org.hq_city} campus of {org.name} maintain the {org.product} documentation.",
                f"Visitors should request archive access through the {org.sector} desk.",
            ],
            source_type="news",
            reliability=0.7,
            year=2022,
            city=org.hq_city,
            fact_ids=[fid],
        )
        if rng.random() < 0.5:
            wrong_year = org.founded + 7
            add_doc(
                f"Forum post on {org.name} founding",
                [
                    f"An anonymous forum post says {org.name} was founded in {wrong_year}.",
                    "No primary filing is attached. The Helion commercial registry contradicts this date.",
                ],
                source_type="blog",
                reliability=0.2,
                year=2021,
                city=org.hq_city,
                fact_ids=[],
                distractor=True,
            )

        # Temporal CEO history
        predecessor = person_list[(person_list.index(ceo) + 9) % len(person_list)]
        start_year = org.founded + 10
        facts[f"fact_{org.org_id}_ceo_hist"] = {
            "kind": "ceo_history",
            "org": org.name,
            "old_ceo": predecessor.name,
            "old_until": start_year + 8,
            "new_ceo": ceo.name,
            "new_from": start_year + 8,
        }
        add_doc(
            f"Leadership timeline: {org.name}",
            [
                f"{predecessor.name} served as chief executive of {org.name} until {start_year + 8}.",
                f"In {start_year + 8}, {ceo.name} became chief executive of {org.name} and remains in office.",
            ],
            source_type="encyclopedia",
            reliability=0.93,
            year=2024,
            city=org.hq_city,
            fact_ids=[f"fact_{org.org_id}_ceo_hist"],
        )

    for event in events.values():
        org = orgs[event.org_id]
        lead = people[event.lead_id]
        fid = f"fact_{event.event_id}"
        facts[fid] = {
            "kind": "event",
            "name": event.name,
            "year": event.year,
            "month": event.month,
            "city": event.city,
            "org": org.name,
            "lead": lead.name,
            "outcome": event.outcome,
        }
        did = add_doc(
            f"Event file: {event.name}",
            [
                f"The {event.name} was held in {event.city} in {event.year}-{event.month:02d}.",
                f"It was organized by {org.name} and led by {lead.name}.",
                f"The recorded outcome: {event.outcome}.",
            ],
            source_type="official",
            reliability=0.94,
            year=event.year,
            city=event.city,
            fact_ids=[fid],
        )
        event_docs[event.event_id] = did
        add_doc(
            f"Wire story: {event.name}",
            [
                f"Reporters in {event.city} covered {event.name}, organized by {org.name}.",
                f"{lead.name} spoke at the closing session.",
            ],
            source_type="news",
            reliability=0.72,
            year=event.year,
            city=event.city,
            fact_ids=[fid],
        )

    # Confusable near-names for disambiguation (all shared given names)
    from collections import defaultdict

    by_first: dict[str, list[Person]] = defaultdict(list)
    for person in person_list:
        by_first[person.name.split()[0]].append(person)
    disambig_pairs: list[tuple[Person, Person]] = []
    for first, group in by_first.items():
        if len(group) < 2:
            continue
        a, b = group[0], group[1]
        disambig_pairs.append((a, b))
        fid = f"fact_disambig_{first.lower()}"
        facts[fid] = {
            "kind": "disambig",
            "name_a": a.name,
            "role_a": a.role,
            "org_a": orgs[a.org_id].name,
            "name_b": b.name,
            "role_b": b.role,
            "org_b": orgs[b.org_id].name,
            "shared_token": first,
        }
        add_doc(
            f"Name collision memo: {first}",
            [
                f"Two archive subjects share the given name {first}: "
                f"{a.name} ({a.role} at {orgs[a.org_id].name}) and "
                f"{b.name} ({b.role} at {orgs[b.org_id].name}).",
                "Do not merge their records.",
            ],
            source_type="official",
            reliability=0.96,
            year=2024,
            city="Aurel",
            fact_ids=[fid],
        )
    twin_a, twin_b = disambig_pairs[0] if disambig_pairs else (person_list[3], person_list[15])

    # Long-context chains (multiple crate series)
    chain_specs: list[dict[str, Any]] = []
    for chain_i in range(8):
        start = 8 + chain_i * 6
        chain_people = person_list[start : start + 5]
        if len(chain_people) < 5:
            continue
        prefix = f"C{chain_i + 1}"
        fid = f"fact_chain_{chain_i}"
        holder_idx = 2
        for idx, person in enumerate(chain_people[:-1]):
            nxt = chain_people[idx + 1]
            code = f"{prefix}-{idx + 1}"
            bit = f"{person.name} transferred a sealed crate code {code} to {nxt.name}."
            add_doc(
                f"Transfer slip {prefix}-{idx + 1}",
                [bit, f"The slip was filed at {orgs[person.org_id].name}."],
                source_type="official",
                reliability=0.9,
                year=2018 + chain_i,
                city=orgs[person.org_id].hq_city,
                fact_ids=[fid],
            )
        facts[fid] = {
            "kind": "chain",
            "prefix": prefix,
            "holder": chain_people[holder_idx].name,
            "code": f"{prefix}-{holder_idx}",
        }
        chain_specs.append(facts[fid] | {"fid": fid, "people": chain_people})

    tasks: dict[str, Task] = {}

    def add_task(
        tid: str,
        question: str,
        gold: str,
        capability: str,
        stage: int,
        gold_docs: list[str],
        hops: int,
        aliases: list[str] | None = None,
        distractors: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        split = _stable_split(tid, seed)
        tasks[tid] = Task(
            task_id=tid,
            question=question,
            gold_answer=str(gold),
            capability=capability,
            stage=stage,
            gold_doc_ids=gold_docs,
            distractor_doc_ids=distractors or [],
            hops=hops,
            split=split.value,
            aliases=aliases or [],
            metadata=metadata or {},
        )

    # Stage 0 single-hop
    for i, person in enumerate(person_list):
        org = orgs[person.org_id]
        add_task(
            f"t_sh_emp_{i:03d}",
            f"Which organization does {person.name} work for?",
            org.name,
            CapabilityName.SINGLE_HOP.value,
            0,
            [person_docs[person.person_id]],
            1,
            distractors=[rumor_docs[person.person_id]] if person.person_id in rumor_docs else [],
        )
        add_task(
            f"t_sh_birth_{i:03d}",
            f"In which city was {person.name} born?",
            person.birth_city,
            CapabilityName.SINGLE_HOP.value,
            0,
            [person_docs[person.person_id]],
            1,
        )
        add_task(
            f"t_sh_spec_{i:03d}",
            f"What is {person.name}'s research specialty?",
            person.specialty,
            CapabilityName.SINGLE_HOP.value,
            0,
            [person_docs[person.person_id]],
            1,
        )
    for i, org in enumerate(org_list):
        add_task(
            f"t_sh_hq_{i:03d}",
            f"Where is {org.name} headquartered?",
            org.hq_city,
            CapabilityName.SINGLE_HOP.value,
            0,
            [org_docs[org.org_id]],
            1,
        )
        add_task(
            f"t_sh_found_{i:03d}",
            f"In what year was {org.name} founded?",
            str(org.founded),
            CapabilityName.SINGLE_HOP.value,
            0,
            [org_docs[org.org_id]],
            1,
            aliases=[str(org.founded)],
        )
        add_task(
            f"t_sh_prod_{i:03d}",
            f"What is the flagship product of {org.name}?",
            org.product,
            CapabilityName.SINGLE_HOP.value,
            0,
            [org_docs[org.org_id]],
            1,
        )

    # Stage 1 multi-hop + decomposition
    for i, person in enumerate(person_list):
        org = orgs[person.org_id]
        add_task(
            f"t_mh_city_{i:03d}",
            f"In which city is the organization that employs {person.name} headquartered?",
            org.hq_city,
            CapabilityName.MULTI_HOP.value,
            1,
            [person_docs[person.person_id], org_docs[org.org_id]],
            2,
        )
        ceo = people[org.ceo_id]
        add_task(
            f"t_mh_ceo_{i:03d}",
            f"Who is the chief executive of the organization that employs {person.name}?",
            ceo.name,
            CapabilityName.MULTI_HOP.value,
            1,
            [person_docs[person.person_id], org_docs[org.org_id]],
            2,
        )
        if person.advisor_id:
            advisor = people[person.advisor_id]
            adv_org = orgs[advisor.org_id]
            add_task(
                f"t_qd_adv_{i:03d}",
                f"What sector does the employer of {person.name}'s advisor belong to?",
                adv_org.sector,
                CapabilityName.QUERY_DECOMPOSITION.value,
                1,
                [person_docs[person.person_id], person_docs[advisor.person_id], org_docs[adv_org.org_id]],
                3,
            )

    # Stage 2 disambiguation + temporal
    shared = twin_a.name.split()[0]
    add_task(
        "t_ed_role_a",
        f"Which {shared} works as a {twin_a.role} at {orgs[twin_a.org_id].name}?",
        twin_a.name,
        CapabilityName.ENTITY_DISAMBIGUATION.value,
        2,
        [person_docs[twin_a.person_id]],
        1,
    )
    add_task(
        "t_ed_role_b",
        f"Which {shared} works as a {twin_b.role} at {orgs[twin_b.org_id].name}?",
        twin_b.name,
        CapabilityName.ENTITY_DISAMBIGUATION.value,
        2,
        [person_docs[twin_b.person_id]],
        1,
    )
    for i, org in enumerate(org_list):
        hist = facts[f"fact_{org.org_id}_ceo_hist"]
        query_year = int(hist["old_until"]) - 1
        add_task(
            f"t_tm_old_{i:03d}",
            f"Who was the chief executive of {org.name} in {query_year}?",
            hist["old_ceo"],
            CapabilityName.TEMPORAL.value,
            2,
            [d.document_id for d in documents.values() if f"fact_{org.org_id}_ceo_hist" in d.fact_ids],
            1,
        )
        add_task(
            f"t_tm_new_{i:03d}",
            f"Who became chief executive of {org.name} in {hist['new_from']}?",
            hist["new_ceo"],
            CapabilityName.TEMPORAL.value,
            2,
            [d.document_id for d in documents.values() if f"fact_{org.org_id}_ceo_hist" in d.fact_ids],
            1,
        )

    # Stage 3 conflicting evidence + source reliability
    for i, org in enumerate(org_list):
        add_task(
            f"t_cf_found_{i:03d}",
            f"According to the most reliable primary registry, in what year was {org.name} founded?",
            str(org.founded),
            CapabilityName.CONFLICTING_EVIDENCE.value,
            3,
            [org_docs[org.org_id]],
            1,
            metadata={"prefer_source": "official"},
        )
        add_task(
            f"t_sr_ceo_{i:03d}",
            f"Using official or encyclopedia sources rather than blogs, who is the current CEO of {org.name}?",
            people[org.ceo_id].name,
            CapabilityName.SOURCE_RELIABILITY.value,
            3,
            [org_docs[org.org_id]],
            1,
            metadata={"prefer_source": "official"},
        )

    # Stage 4 long context / synthesis / recovery
    add_task(
        "t_lc_c3",
        "Who received crate code C-3 according to the transfer slips?",
        facts["fact_chain"]["holder_of_c3"],
        CapabilityName.LONG_CONTEXT.value,
        4,
        [d.document_id for d in documents.values() if "fact_chain" in d.fact_ids],
        3,
    )
    for i, event in enumerate(list(events.values())[:24]):
        org = orgs[event.org_id]
        lead = people[event.lead_id]
        add_task(
            f"t_xd_evt_{i:03d}",
            f"Which organization organized {event.name}, and who led it?",
            f"{org.name}; {lead.name}",
            CapabilityName.CROSS_DOCUMENT.value,
            4,
            [event_docs[event.event_id], org_docs[org.org_id], person_docs[lead.person_id]],
            2,
            aliases=[org.name, lead.name],
        )
        add_task(
            f"t_fr_evt_{i:03d}",
            f"Ignore rumor blogs. In which city was {event.name} held?",
            event.city,
            CapabilityName.FAILURE_RECOVERY.value,
            4,
            [event_docs[event.event_id]],
            1,
            distractors=list(rumor_docs.values())[:3],
        )

    return World(
        seed=seed,
        people=people,
        orgs=orgs,
        events=events,
        documents=documents,
        tasks=tasks,
        facts=facts,
    )


def world_stats(world: World) -> dict[str, Any]:
    from collections import Counter

    caps = Counter(t.capability for t in world.tasks.values())
    splits = Counter(t.split for t in world.tasks.values())
    stages = Counter(t.stage for t in world.tasks.values())
    return {
        "n_people": len(world.people),
        "n_orgs": len(world.orgs),
        "n_events": len(world.events),
        "n_documents": len(world.documents),
        "n_tasks": len(world.tasks),
        "by_capability": dict(caps),
        "by_split": dict(splits),
        "by_stage": dict(stages),
        "seed": world.seed,
    }


import re
from collections import defaultdict

from rapidfuzz import fuzz


def normalize_name(name: str) -> str:
    """Normalize an entity name for comparison."""
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def resolve_entities(
    entities: list[dict],
    similarity_threshold: int = 90,
) -> tuple[list[dict], dict[str, str]]:
    """
    Deduplicate entities of the same type.

    Returns:
        unique_entities: Deduplicated entity records.
        aliases: Maps original entity names to canonical names.
    """
    grouped = defaultdict(list)

    for entity in entities:
        entity_type = entity["entity_type"]
        grouped[entity_type].append(entity)

    unique_entities = []
    aliases = {}

    for entity_type, group in grouped.items():
        canonical_entities = []

        for entity in group:
            name = entity["name"].strip()
            normalized = normalize_name(name)

            if not normalized:
                continue

            matched = None

            for canonical in canonical_entities:
                canonical_normalized = normalize_name(
                    canonical["name"]
                )

                score = fuzz.ratio(
                    normalized,
                    canonical_normalized,
                )

                token_score = fuzz.token_set_ratio(
                    normalized,
                    canonical_normalized,
                )

                if (
                    normalized == canonical_normalized
                    or score >= similarity_threshold
                    or token_score >= similarity_threshold
                ):
                    matched = canonical
                    break

            if matched:
                aliases[name] = matched["name"]

                # Preserve a description if the canonical one is empty.
                if (
                    not matched.get("description")
                    and entity.get("description")
                ):
                    matched["description"] = entity["description"]

            else:
                canonical_entities.append({
                    **entity,
                    "entity_type": entity_type,
                })
                aliases[name] = name

        unique_entities.extend(canonical_entities)

    return unique_entities, aliases

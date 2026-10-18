
from research_assistant.processing.entity_resolver import resolve_entities

test_entities = [
    {"name": "GeoState", "entity_type": "Method", "description": "World state representation"},
    {"name": "Geostate", "entity_type": "Method", "description": "State representation"},
    {"name": "Transformer", "entity_type": "Method", "description": "Neural network architecture"},
    {"name": "Transformer Model", "entity_type": "Method", "description": "Transformer architecture"},
    {"name": "ImageNet", "entity_type": "Dataset", "description": "Image dataset"},
    {"name": "ImageNet", "entity_type": "Concept", "description": "Same name, different type"},
]

unique_entities, aliases = resolve_entities(
    test_entities,
    similarity_threshold=85,
)

print("Original:", len(test_entities))
print("Unique:", len(unique_entities))
print("Aliases:", aliases)

assert len(unique_entities) == 4
assert aliases["Geostate"] == "GeoState"
assert aliases["Transformer Model"] == "Transformer"
assert aliases["ImageNet"] == "ImageNet"

print("All tests passed!")

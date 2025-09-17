# Add this to your existing script or run separately
import pickle, json

with open("/workspaces/graph_rag/kg.pkl", "rb") as f:
    data = pickle.load(f)

json_data = {
    "entities": {name: {
        "type": entity.get("type", "Unknown"),
        "description": entity.get("description", "No description")
    } for name, entity in data["entities"].items()},
    "relationships": data["relationships"]
}

with open("/workspaces/graph_rag/kg.json", "w") as f:
    json.dump(json_data, f, indent=2)

print(" JSON export complete!")
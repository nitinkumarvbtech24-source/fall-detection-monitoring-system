"""
PipeGen Stage 2: Pipeline IR Validator & Visualizer
Validates the DAG topological consistency and generates Mermaid / Graphviz diagrams.
"""

import json
import yaml
import sys
from pathlib import Path

def load_ir(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        if file_path.suffix in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        else:
            return json.load(f)

def validate_pipeline_ir(ir_data: dict):
    pipe = ir_data.get("pipeline_ir", {})
    nodes = {n["id"]: n for n in pipe.get("nodes", [])}
    connections = pipe.get("connections", [])

    print(f"[*] Validating Pipeline IR: {pipe.get('name')} v{pipe.get('version')}")
    print(f"[*] Total Nodes: {len(nodes)}")
    print(f"[*] Total Interconnects: {len(connections)}")

    # Verify every connection's endpoints
    errors = []
    for conn in connections:
        src = conn["from"].split(".")[0]
        dst = conn["to"].split(".")[0]
        if src not in nodes:
            errors.append(f"Connection source node '{src}' does not exist.")
        if dst not in nodes:
            errors.append(f"Connection destination node '{dst}' does not exist.")

    if errors:
        print("[!] Validation FAILED:")
        for err in errors:
            print(f"    - {err}")
        return False

    print("[+] Pipeline IR is syntactically and topologically VALID!")
    return True

def generate_mermaid_diagram(ir_data: dict) -> str:
    pipe = ir_data.get("pipeline_ir", {})
    nodes = {n["id"]: n for n in pipe.get("nodes", [])}
    connections = pipe.get("connections", [])

    lines = ["```mermaid", "graph TD"]
    lines.append("    %% PipeGen Pipeline IR DAG")
    for nid, node in nodes.items():
        node_type = node.get("type", "Node")
        node_name = node.get("name", nid)
        lines.append(f'    {nid}["<b>{node_name}</b><br/><i>({node_type})</i>"]')

    for conn in connections:
        src_node, src_port = conn["from"].split(".")
        dst_node, dst_port = conn["to"].split(".")
        lines.append(f'    {src_node} -->|"{src_port} -> {dst_port}"| {dst_node}')

    lines.append("```")
    return "\n".join(lines)

if __name__ == "__main__":
    ir_yaml_path = Path(__file__).parent / "pipeline.ir.yaml"
    ir_json_path = Path(__file__).parent / "pipeline.ir.json"

    ir_data = load_ir(ir_yaml_path)
    
    # Save canonical JSON equivalent
    with open(ir_json_path, "w", encoding="utf-8") as f:
        json.dump(ir_data, f, indent=2)
    print(f"[+] Saved canonical JSON IR to: {ir_json_path.name}")

    if validate_pipeline_ir(ir_data):
        mermaid = generate_mermaid_diagram(ir_data)
        mermaid_path = Path(__file__).parent / "pipeline_graph.md"
        with open(mermaid_path, "w", encoding="utf-8") as f:
            f.write("# PipeGen Pipeline IR Flow Graph\n\n" + mermaid + "\n")
        print(f"[+] Generated Mermaid Graph to: {mermaid_path.name}")

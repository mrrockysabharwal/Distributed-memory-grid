"""
Integration test for Week 4 deliverables:
  - Validates automated cluster demonstration execution (scripts/demo_cluster.py)
  - Validates documentation integrity (api_spec.md, deployment_and_demo.md, final_system_report.md)
  - Validates Docker Compose multi-service definitions
"""
import os
import pytest
from scripts.demo_cluster import run_full_cluster_demonstration


def test_full_cluster_demonstration_workflow():
    result = run_full_cluster_demonstration(verbose=False)
    assert result["status"] == "success"
    assert result["registered_workers"] == 3
    assert result["surviving_workers"] == 2
    assert result["tokens_generated"] == 15
    assert result["ttft_seconds"] >= 0.0
    assert result["throughput_tps"] > 0.0
    assert "worker-1" in result["rebalanced_partitions"]
    assert "worker-3" in result["rebalanced_partitions"]
    assert "worker-2" not in result["rebalanced_partitions"]


def test_documentation_and_spec_files_exist():
    required_docs = [
        "docs/api_spec.md",
        "docs/deployment_and_demo.md",
        "docs/final_system_report.md",
        "docs/benchmarks/report.md",
    ]
    for doc_path in required_docs:
        assert os.path.exists(doc_path), f"Required documentation file missing: {doc_path}"
        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert len(content) > 200, f"Documentation file {doc_path} appears empty or truncated"


def test_docker_compose_configuration():
    assert os.path.exists("docker-compose.yml"), "docker-compose.yml is missing"
    with open("docker-compose.yml", "r", encoding="utf-8") as f:
        compose_content = f.read()

    assert "dmg-master" in compose_content
    assert "dmg-worker-1" in compose_content
    assert "dmg-worker-2" in compose_content
    assert "dmg-frontend" in compose_content
    assert "dmg-network" in compose_content


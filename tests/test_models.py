from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from bora.formula_adverse_reaction.model import BORAFormulaADR, Config as FormulaConfig
from bora.drug_drug_interaction.model import BORADrugInteraction, ModelConfig as DDIConfig
from bora.drug_pair_adverse_reaction.model import BORAPairADR, ModelConfig as PairConfig
from bora.herb_drug_interaction.model import BORAHerbDrug, Config as HerbConfig
from bora.herb_drug_interaction.unseen_herb_expert import UnseenHerbExpert
from bora.herb_drug_interaction.reproduce import reproduce_all


def test_formula_forward() -> None:
    cfg = FormulaConfig(d_model=8, n_heads=2, n_layers=1, dropout=0.0,
                        mem_topk_f=2, mem_topk_a=2)
    model = BORAFormulaADR(8, 12, 6, cfg, n_endpoint=2, n_material=5).eval()
    formula_count, endpoint_count, slots = 3, 2, 4
    logits, _, _, _ = model(
        torch.randn(formula_count, slots, 8), torch.ones(formula_count, slots),
        torch.ones(formula_count, slots, dtype=torch.bool), torch.randn(endpoint_count, 12),
        torch.randn(formula_count, endpoint_count, 6), torch.zeros(formula_count, endpoint_count),
        torch.ones(formula_count, endpoint_count), hierarchy=torch.eye(endpoint_count),
        presence=torch.zeros(formula_count, 5), comp_dose=torch.zeros(formula_count, 5),
    )
    assert logits.shape == (formula_count, endpoint_count)


def _entity_features(n: int, cfg: DDIConfig) -> dict[str, torch.Tensor]:
    return {"semantic": torch.randn(n, cfg.semantic_dim),
            "graph": torch.randn(n, cfg.graph_dim),
            "structure": torch.randn(n, cfg.structure_dim)}


def test_drug_pair_symmetry() -> None:
    cfg = DDIConfig(semantic_dim=8, graph_dim=4, structure_dim=9, d_model=8,
                    n_heads=2, n_layers=1, mechanism_dim=6,
                    memory_topk_left=2, memory_topk_right=2)
    model = BORADrugInteraction("ddi_binary", n_entity=4, cfg=cfg).eval()
    token, state = model.encode_entities(_entity_features(4, cfg))
    labels, observed, mechanism = torch.zeros(4, 4), torch.zeros(4, 4), torch.randn(1, 6)
    ab = model.score_ddi_binary(token, state, torch.tensor([0]), torch.tensor([1]), mechanism,
                                labels, observed)
    ba = model.score_ddi_binary(token, state, torch.tensor([1]), torch.tensor([0]), mechanism,
                                labels, observed)
    torch.testing.assert_close(ab, ba, atol=1e-6, rtol=1e-6)


def test_drug_pair_adr_forward() -> None:
    cfg = PairConfig(semantic_dim=8, graph_dim=4, structure_dim=9,
                     endpoint_mechanism_dim=8, pair_mechanism_dim=6,
                     d_model=8, n_heads=2, n_layers=1,
                     memory_topk_drug=2, memory_topk_endpoint=2)
    model = BORAPairADR(n_endpoint=3, cfg=cfg).eval()
    drug = {"semantic": torch.randn(4, 8), "graph": torch.randn(4, 4),
            "structure": torch.randn(4, 9)}
    endpoint = {"semantic": torch.randn(3, 8), "graph": torch.randn(3, 4),
                "mechanism": torch.randn(3, 8)}
    drug_tokens, drug_state = model.encode_drugs(drug)
    endpoint_tokens, endpoint_state = model.encode_endpoints(endpoint)
    neighbors = model.neighbor_tables(drug_state, endpoint_state)
    logits = model.score_encoded(
        drug_tokens, drug_state, endpoint_tokens, endpoint_state,
        torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([0, 2]),
        torch.randn(2, 6), torch.zeros(4, 3), torch.ones(4, 3),
        torch.full((3,), 0.1), neighbors,
    )
    assert logits.shape == (2,)


def test_herb_drug_forward() -> None:
    cfg = HerbConfig(semantic_dim=8, graph_dim=4, structure_dim=9,
                     constituent_dim=9, pair_mechanism_dim=6,
                     d_model=8, n_heads=2, n_layers=1,
                     memory_topk_herb=2, memory_topk_drug=2, dropout=0.0)
    model = BORAHerbDrug(3, 4, cfg).eval()
    semantic_h, graph_h, structure_h = torch.randn(3, 8), torch.randn(3, 4), torch.randn(3, 9)
    semantic_d, graph_d, structure_d = torch.randn(4, 8), torch.randn(4, 4), torch.randn(4, 9)
    herb_tokens, _, herb_state = model.encode_herb(
        semantic_h, graph_h, structure_h, torch.randn(3, 5, 9), torch.ones(3, 5, dtype=torch.bool)
    )
    _, drug_state = model.encode_drug(semantic_d, graph_d, structure_d)
    memory = model.retrieve_relations(herb_state, drug_state, torch.zeros(3, 4), torch.ones(3, 4))
    logits = model.score(herb_tokens, torch.ones(3, 5, dtype=torch.bool), herb_state, drug_state,
                         torch.tensor([0, 1]), torch.tensor([1, 2]), torch.randn(2, 6), memory)
    assert logits.shape == (2,)


def test_herb_router_zero_alpha_returns_stability_guarded_core() -> None:
    cfg = HerbConfig(semantic_dim=8, graph_dim=4, structure_dim=9,
                     constituent_dim=9, pair_mechanism_dim=6,
                     d_model=8, n_heads=2, n_layers=1,
                     memory_topk_herb=2, memory_topk_drug=2, dropout=0.0)
    model = BORAHerbDrug(3, 4, cfg).eval()
    herb_tokens, _, herb_state = model.encode_herb(
        torch.randn(3, 8), torch.randn(3, 4), torch.randn(3, 9),
        torch.randn(3, 5, 9), torch.ones(3, 5, dtype=torch.bool))
    _, drug_state = model.encode_drug(
        torch.randn(4, 8), torch.randn(4, 4), torch.randn(4, 9))
    model.retrieve_relations(herb_state, drug_state, torch.zeros(3, 4),
                       torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0],
                                     [1, 1, 1, 1]], dtype=torch.float32))
    core_logit = torch.tensor([-100.0, 0.75])
    expert = UnseenHerbExpert(21).eval()
    routed = model.route_unseen_herb(
        core_logit, expert, torch.tensor([1, 2]), torch.tensor([0, 1]), 0.0)
    core_probability = torch.sigmoid(core_logit)
    clipped_core = core_probability.to(torch.float64).clamp(1e-7, 1 - 1e-7)
    assert routed[0].item() == clipped_core[0].item()
    assert routed[1].item() == core_probability[1].item()
    assert bool(model.last_route.tolist()[0])
    assert not bool(model.last_route.tolist()[1])


def test_herb_router_leaves_supported_cells_unchanged() -> None:
    cfg = HerbConfig(semantic_dim=8, graph_dim=4, structure_dim=9,
                     constituent_dim=9, pair_mechanism_dim=6,
                     d_model=8, n_heads=2, n_layers=1,
                     memory_topk_herb=2, memory_topk_drug=2, dropout=0.0)
    model = BORAHerbDrug(3, 4, cfg).eval()
    _, _, herb_state = model.encode_herb(
        torch.randn(3, 8), torch.randn(3, 4), torch.randn(3, 9),
        torch.randn(3, 5, 9), torch.ones(3, 5, dtype=torch.bool))
    _, drug_state = model.encode_drug(
        torch.randn(4, 8), torch.randn(4, 4), torch.randn(4, 9))
    model.retrieve_relations(herb_state, drug_state, torch.zeros(3, 4),
                       torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0],
                                     [1, 1, 1, 1]], dtype=torch.float32))
    core_logit = torch.tensor([-1.25, 0.75])
    expert = UnseenHerbExpert(21).eval()
    routed = model.route_unseen_herb(
        core_logit, expert, torch.tensor([1, 2]), torch.tensor([0, 1]), 0.5)
    assert routed[1].item() == torch.sigmoid(core_logit)[1].item()
    assert bool(model.last_route.tolist()[0])
    assert not bool(model.last_route.tolist()[1])


@pytest.mark.skipif(
    "BORA_HERB_DRUG_DATA" not in os.environ,
    reason="set BORA_HERB_DRUG_DATA to run the public-data reproduction test")
def test_herb_drug_release_reproduces_reported_metrics() -> None:
    repository = Path(__file__).resolve().parents[1]
    report = reproduce_all(
        os.environ["BORA_HERB_DRUG_DATA"],
        repository / "parameters/herb_drug_interaction",
        os.environ.get("BORA_DEVICE", "cpu"),
        reported_decimals=3,
    )
    assert report["run_count"] == 30
    assert report["all_runs_match"]

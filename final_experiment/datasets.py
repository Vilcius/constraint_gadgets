"""Generate and audit the fixed 500-COP final experiment dataset."""
from collections import Counter, defaultdict
import ast
import json
from pathlib import Path
import pickle
import re

import numpy as np

from core import constraint_handler as ch
from core.penalty_heuristics import coefficient_penalties
from final_experiment.storage import ROOT, digest, write_json

HERE = ROOT / "final_experiment"
EXACT_CATEGORIES = (
    "cardinality_equality",
    "cardinality_inequality",
    "assignment",
    "flow",
    "independent_set",
)
EXPECTED_SIZES = {4: 15, 5: 30, 6: 70, 7: 93, 8: 97, 9: 102, 10: 93}
EXPECTED_PARTITIONS = {(2, 0): 200, (3, 0): 50, (1, 1): 160,
                       (2, 1): 74, (1, 2): 16}


def bitstrings(n):
    return [format(index, f"0{n}b") for index in range(2**n)]


def feasible_mask(constraints, n):
    parsed = ch.parse_constraints(constraints)
    return np.asarray([ch.check_feasibility(bits, parsed, n) for bits in bitstrings(n)])


def objective_reference(Q, constraints):
    n = len(Q)
    strings = bitstrings(n)
    bits = np.asarray([[int(value) for value in string] for string in strings])
    values = np.einsum("bi,ij,bj->b", bits, Q, bits)
    mask = feasible_mask(constraints, n)
    if not mask.any():
        raise ValueError("Infeasible COP")
    best = float(values[mask].min())
    return {
        "feasible_mask": mask.tolist(), "objective_values": values.tolist(),
        "optimal_indices": np.flatnonzero(mask & (values == best)).tolist(),
        "f_star": best, "f_max_feasible": float(values[mask].max()),
        "f_min_unconstrained": float(values.min()),
        "feasible_fraction": float(mask.mean()),
    }


def load_config(path=HERE / "config.json"):
    return json.loads(Path(path).read_text())


def _load_vcgs(config):
    database_path = Path(config["vcg_database"])
    manifest_path = Path(config["vcg_manifest"])
    if not database_path.is_absolute():
        database_path = ROOT / database_path
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    database = pickle.loads(database_path.read_bytes())
    source = json.loads(manifest_path.read_text())["instances"]
    rows = []
    for row in source:
        key = ch.normalize_constraint(row["constraint"])
        if key not in database:
            raise ValueError(f"Selected VCG database is missing {row['id']}: {key}")
        entry = database[key]
        if entry.get("loss_mode") != "fidelity" or float(entry["fidelity"]) < 0.999:
            raise ValueError(f"VCG {row['id']} is not a retained L_fid gadget")
        rows.append({
            "vcg_id": row["id"], "support": int(row["support"]),
            "vcg_family": row["family"], "canonical_constraint": key,
            "fidelity": float(entry["fidelity"]), "vcg_depth": int(entry["n_layers"]),
        })
    rows.sort(key=lambda item: (item["support"], item["vcg_id"]))
    if len(rows) != 50:
        raise ValueError(f"Expected 50 selected VCGs, found {len(rows)}")
    return rows


def _valid_assignment(groups, support):
    for disjoint, overlap in groups:
        if len(disjoint) != 5 or len(overlap) != 5:
            return False
        if {cell[1] for cell in overlap} != {1, 2}:
            return False
        combined = disjoint + overlap
        if not any(cell[0] == 2 for cell in combined):
            return False
        if not any(cell[0] == 3 for cell in combined):
            return False
    return True


def _assign_cells(config, vcgs, rng):
    assignments = []
    for support in range(4, 9):
        gadgets = [row for row in vcgs if row["support"] == support]
        target = config["cells"][str(support)]
        disjoint = []
        overlap = []
        for key, count in target.items():
            m, o = map(int, key.split("_"))
            (disjoint if o == 0 else overlap).extend([(m, o)] * int(count))
        if len(disjoint) != 50 or len(overlap) != 50:
            raise ValueError(f"Support {support} must have 50 disjoint and 50 overlapping cells")
        groups = None
        for _ in range(20000):
            rng.shuffle(disjoint)
            rng.shuffle(overlap)
            candidate = [
                (list(disjoint[5*i:5*i+5]), list(overlap[5*i:5*i+5]))
                for i in range(10)
            ]
            if _valid_assignment(candidate, support):
                groups = candidate
                break
        if groups is None:
            raise RuntimeError(f"Could not assign cells to support-{support} VCGs")
        for gadget, (left, right) in zip(gadgets, groups):
            for local_index, (m, o) in enumerate(left + right, 1):
                assignments.append({**gadget, "m": m, "o": o,
                                    "vcg_use": local_index})
    assignments.sort(key=lambda row: (row["support"], row["vcg_id"], row["vcg_use"]))
    return assignments


def _assign_topologies(skeletons, rng):
    topologies = ["star"] * 16 + ["chain"] * 16 + ["double"] * 15
    rng.shuffle(topologies)
    cursor = 0
    for row in skeletons:
        if row["m"] == 3 and row["o"] == 2:
            row["overlap_topology"] = topologies[cursor]
            cursor += 1
        elif row["o"] == 0:
            row["overlap_topology"] = "disjoint"
        elif row["m"] == 2:
            row["overlap_topology"] = f"single_constraint_overlap_{row['o']}"
        else:
            row["overlap_topology"] = "one_vcg_overlap"
    if cursor != 47:
        raise ValueError(f"Expected 47 three-constraint o=2 COPs, found {cursor}")


def _category_realizable(row, categories):
    """Whether some permitted placement leaves at least two feasible strings."""
    seed = int(digest([row["vcg_id"], row["m"], row["o"],
                       row["overlap_topology"], categories])[:8], 16)
    rng = np.random.default_rng(seed)
    for trial in range(40):
        n, vcg_wires, pairs = _supports(row, rng)
        constraints = [_remap_constraint(row["canonical_constraint"], vcg_wires)]
        constraints += [_exact_constraint(category, pair, trial + slot)
                        for slot, (category, pair) in enumerate(zip(categories, pairs))]
        parsed = ch.parse_constraints(constraints)
        if all(_is_exact(pc) for pc in parsed[1:]):
            mask = feasible_mask(constraints, n)
            if int(mask.sum()) >= 2 and not bool(mask.all()):
                return True
    return False


def _assign_exact_categories(skeletons, rng):
    for row in skeletons:
        choices = []
        if row["m"] == 2:
            candidates = [(name,) for name in EXACT_CATEGORIES]
        else:
            candidates = [(first, second) for first in EXACT_CATEGORIES
                          for second in EXACT_CATEGORIES
                          if not (first == second == "cardinality_inequality")]
        for categories in candidates:
            if _category_realizable(row, categories):
                choices.append(categories)
        if not choices:
            raise RuntimeError(f"No valid exact category for {row}")
        row["compatible_categories"] = choices

    # Constrained rows first. Random tie order gives multiple deterministic
    # attempts while keeping scarce category combinations available.
    for _ in range(1000):
        remaining = Counter({name: 128 for name in EXACT_CATEGORIES})
        strata = defaultdict(Counter)
        order = list(range(len(skeletons)))
        rng.shuffle(order)
        order.sort(key=lambda index: len(skeletons[index]["compatible_categories"]))
        assigned = {}
        failed = False
        for index in order:
            row = skeletons[index]
            candidates = []
            for categories in row["compatible_categories"]:
                needed = Counter(categories)
                if any(remaining[name] < count for name, count in needed.items()):
                    continue
                score = 0.0
                for category in categories:
                    score += strata[("support", row["support"])][category]
                    score += strata[("family", row["vcg_family"])][category]
                    score += strata[("m", row["m"])][category]
                    score += strata[("o", row["o"])][category]
                    score -= remaining[category] / 128
                candidates.append((score, float(rng.random()), categories))
            if not candidates:
                failed = True
                break
            _, _, categories = min(candidates)
            assigned[index] = categories
            remaining.subtract(categories)
            for category in categories:
                strata[("support", row["support"])][category] += 1
                strata[("family", row["vcg_family"])][category] += 1
                strata[("m", row["m"])][category] += 1
                strata[("o", row["o"])][category] += 1
        if not failed and not any(remaining.values()):
            for index, row in enumerate(skeletons):
                row["exact_categories"] = list(assigned[index])
                row.pop("compatible_categories", None)
            return
    raise RuntimeError("Could not satisfy compatibility and 128-per-category quotas")


def _remap_constraint(constraint, wires):
    return re.sub(r"x_(\d+)", lambda match: f"x_{wires[int(match.group(1))]}", constraint)


def _supports(row, rng):
    k, m, o = row["support"], row["m"], row["o"]
    n = k + 2 * (m - 1) - o
    # The existing VCG database normalizes variables by sorted wire index.
    # Preserve that internal order while varying which global wires it occupies.
    vcg = sorted(map(int, rng.choice(n, size=k, replace=False)))
    overlap_wires = list(map(int, rng.permutation(vcg)))
    extra = [wire for wire in map(int, rng.permutation(n)) if wire not in vcg]
    if m == 2:
        if o == 0:
            pairs = [(extra[0], extra[1])]
        elif o == 1:
            pairs = [(overlap_wires[0], extra[0])]
        else:
            pairs = [(overlap_wires[0], overlap_wires[1])]
    elif o == 0:
        pairs = [(extra[0], extra[1]), (extra[2], extra[3])]
    elif o == 1:
        pairs = [(overlap_wires[0], extra[0]), (extra[1], extra[2])]
    elif row["overlap_topology"] == "star":
        pairs = [(overlap_wires[0], extra[0]), (overlap_wires[1], extra[1])]
    elif row["overlap_topology"] == "chain":
        pairs = [(overlap_wires[0], extra[0]), (extra[0], extra[1])]
    elif row["overlap_topology"] == "double":
        pairs = [(overlap_wires[0], overlap_wires[1]), (extra[0], extra[1])]
    else:
        raise ValueError(f"Unknown topology {row['overlap_topology']}")
    return n, vcg, pairs


def _exact_constraint(category, pair, variant):
    i, j = pair
    if category == "cardinality_equality":
        rhs = variant % 3
        return f"1*x_{i} + 1*x_{j} == {rhs}"
    if category == "cardinality_inequality":
        return (f"1*x_{i} + 1*x_{j} <= 1" if variant % 4 else
                f"1*x_{i} + 1*x_{j} >= 2")
    if category == "assignment":
        return f"1*x_{i} + 1*x_{j} == 1"
    if category == "flow":
        return f"1*x_{i} - 1*x_{j} == 0"
    if category == "independent_set":
        return f"1*x_{i}*x_{j} == 0"
    raise ValueError(category)


def _is_exact(parsed):
    return any(check(parsed) for check in (
        ch.is_dicke_compatible, ch.is_cardinality_leq_compatible,
        ch.is_cardinality_geq_single_compatible,
        ch.is_independent_set_pair_compatible, ch.is_flow_compatible,
    ))


def _partition(parsed):
    structural = [0]
    penalty = []
    occupied = set(parsed[0].variables)
    for index in range(1, len(parsed)):
        if parsed[index].variables.isdisjoint(occupied):
            structural.append(index)
            occupied.update(parsed[index].variables)
        else:
            penalty.append(index)
    return structural, penalty


def _qubo_string(Q):
    terms = []
    for i in range(len(Q)):
        for j in range(i, len(Q)):
            value = int(Q[i, j])
            if not value:
                continue
            terms.append(f"{value}*x_{i}" if i == j else f"{value}*x_{i} x_{j}")
    return " + ".join(terms)


def _reserved_objective_hashes():
    hashes = set()
    original = ROOT / "data" / "qubos.csv"
    if original.exists():
        for line in original.read_text().splitlines():
            fields = line.split(";")
            if len(fields) >= 3:
                hashes.add(digest(np.asarray(ast.literal_eval(fields[2]), dtype=int).tolist()))
    for path in [ROOT / "tuning/results/full-b-five-methods/manifest_b.json",
                 ROOT / "tuning/results/full/manifest_b.json"]:
        if path.exists():
            for row in json.loads(path.read_text()).get("instances", []):
                hashes.add(digest(np.asarray(row["Q"], dtype=int).tolist()))
    return hashes


def _build_instance(row, index, config, rng, used_constraints, used_objectives,
                    reserved_objectives, category_variants):
    for attempt in range(10000):
        instance_seed = int(rng.integers(0, 2**32, dtype=np.uint32))
        local = np.random.default_rng(instance_seed)
        n, vcg_wires, exact_pairs = _supports(row, local)
        vcg_constraint = _remap_constraint(row["canonical_constraint"], vcg_wires)
        exact_constraints = []
        for category, pair in zip(row["exact_categories"], exact_pairs):
            variant = category_variants[category]
            category_variants[category] += 1
            exact_constraints.append(_exact_constraint(category, pair, variant))
        constraints = [vcg_constraint, *exact_constraints]
        parsed = ch.parse_constraints(constraints)
        if ch.normalize_constraint(vcg_constraint) != row["canonical_constraint"]:
            raise ValueError("Mapped VCG no longer matches its database key")
        if any(not _is_exact(pc) for pc in parsed[1:]):
            raise ValueError(f"Generated non-exact constraint: {constraints}")
        constraint_key = tuple(sorted(constraints))
        if constraint_key in used_constraints:
            continue
        mask = feasible_mask(constraints, n)
        if int(mask.sum()) < 2 or bool(mask.all()):
            continue
        structural, penalty = _partition(parsed)
        expected_overlap = sum(len(pc.variables) for pc in parsed) - len(set().union(
            *(pc.variables for pc in parsed)))
        if expected_overlap != row["o"]:
            raise ValueError("Generated support overlap does not match the assigned cell")
        penalty_slack = ch.determine_slack_variables(parsed, n)[1]
        pc_slack = ch.determine_slack_variables([parsed[i] for i in penalty], n)[1]
        if n + penalty_slack > config["max_total_qubits"]:
            continue
        objective_seed = int(rng.integers(0, 2**32, dtype=np.uint32))
        objective_rng = np.random.default_rng(objective_seed)
        Q = np.triu(objective_rng.integers(
            config["objective_coefficient_min"],
            config["objective_coefficient_max"] + 1,
            size=(n, n), dtype=int))
        objective_hash = digest(Q.tolist())
        if objective_hash in used_objectives or objective_hash in reserved_objectives:
            continue
        reference = objective_reference(Q, constraints)
        if reference["f_star"] == reference["f_max_feasible"]:
            continue
        structural_constraints = [constraints[i] for i in structural]
        structural_mask = feasible_mask(structural_constraints, n)
        strings = bitstrings(n)
        optimal_bitstrings = [strings[i] for i in reference["optimal_indices"]]
        feasible_bitstrings = [value for value, yes in zip(strings, mask) if yes]
        delta = coefficient_penalties(Q, parsed)["range"]
        used_constraints.add(constraint_key)
        used_objectives.add(objective_hash)
        return {
            "id": f"final_{index + 1:04d}",
            **row,
            "n_x": n,
            "regime": "disjoint" if row["o"] == 0 else "overlapping",
            "constraints": constraints,
            "families": [row["vcg_family"], *row["exact_categories"]],
            "vcg_constraint": vcg_constraint,
            "vcg_wires": vcg_wires,
            "exact_supports": [list(pair) for pair in exact_pairs],
            "structural_indices": structural,
            "penalty_indices": penalty,
            "n_structural": len(structural),
            "n_penalty": len(penalty),
            "penalty_rule": "range",
            "delta": delta,
            "Q": Q.tolist(),
            "qubo_string": _qubo_string(Q),
            "objective_seed": objective_seed,
            "objective_hash": objective_hash,
            "instance_seed": instance_seed,
            "constraint_hash": digest(constraint_key),
            "n_slack_penaltyqaoa": penalty_slack,
            "n_total_penaltyqaoa": n + penalty_slack,
            "n_slack_pcqaoa": pc_slack,
            "n_total_pcqaoa": n + pc_slack,
            "feasible_count": int(mask.sum()),
            "structural_feasible_count": int(structural_mask.sum()),
            "structural_feasible_fraction": float(structural_mask.mean()),
            "feasible_bitstrings": feasible_bitstrings,
            "optimal_bitstrings": optimal_bitstrings,
            **reference,
        }
    raise RuntimeError(f"Could not generate a valid instance for {row}")


def generate(config):
    rng = np.random.default_rng(config["dataset_seed"])
    vcgs = _load_vcgs(config)
    skeletons = _assign_cells(config, vcgs, rng)
    _assign_topologies(skeletons, rng)
    _assign_exact_categories(skeletons, rng)
    used_constraints, used_objectives = set(), set()
    reserved = _reserved_objective_hashes()
    variants = Counter()
    instances = [
        _build_instance(row, index, config, rng, used_constraints, used_objectives,
                        reserved, variants)
        for index, row in enumerate(skeletons)
    ]
    return instances


def audit(instances, config):
    objective_coefficients = [
        int(value)
        for row in instances
        for i, coefficients in enumerate(row["Q"])
        for value in coefficients[i:]
    ]
    reserved_objectives = _reserved_objective_hashes()
    summary = {
        "instances": len(instances),
        "unique_cops": len({(row["constraint_hash"], row["objective_hash"]) for row in instances}),
        "unique_objectives": len({row["objective_hash"] for row in instances}),
        "unique_constraint_collections": len({row["constraint_hash"] for row in instances}),
        "unique_vcgs": len({row["vcg_id"] for row in instances}),
        "vcg_uses": dict(sorted(Counter(row["vcg_id"] for row in instances).items())),
        "vcg_support": dict(sorted(Counter(row["support"] for row in instances).items())),
        "vcg_family": dict(sorted(Counter(row["vcg_family"] for row in instances).items())),
        "constraint_count": dict(sorted(Counter(row["m"] for row in instances).items())),
        "overlap_count": dict(sorted(Counter(row["o"] for row in instances).items())),
        "regime": dict(sorted(Counter(row["regime"] for row in instances).items())),
        "decision_size": dict(sorted(Counter(row["n_x"] for row in instances).items())),
        "exact_category": dict(sorted(Counter(
            category for row in instances for category in row["exact_categories"]).items())),
        "partition": {f"{key[0]}_structural_{key[1]}_penalized": value
                      for key, value in sorted(Counter(
                          (row["n_structural"], row["n_penalty"])
                          for row in instances).items())},
        "cell": {f"k{k}_m{m}_o{o}": value for (k, m, o), value in sorted(Counter(
            (row["support"], row["m"], row["o"]) for row in instances).items())},
        "topology": dict(sorted(Counter(row["overlap_topology"] for row in instances).items())),
        "constraint_occurrences": sum(row["m"] for row in instances),
        "vcg_occurrences": len(instances),
        "exact_constraint_occurrences": sum(row["m"] - 1 for row in instances),
        "max_penaltyqaoa_qubits": max(row["n_total_penaltyqaoa"] for row in instances),
        "max_pcqaoa_qubits": max(row["n_total_pcqaoa"] for row in instances),
        "penaltyqaoa_qubits": dict(sorted(Counter(
            row["n_total_penaltyqaoa"] for row in instances).items())),
        "pcqaoa_qubits": dict(sorted(Counter(
            row["n_total_pcqaoa"] for row in instances).items())),
        "feasible_count_min": min(row["feasible_count"] for row in instances),
        "feasible_count_max": max(row["feasible_count"] for row in instances),
        "vcg_fidelity_min": min(row["fidelity"] for row in instances),
        "objective_coefficient_min": min(objective_coefficients),
        "objective_coefficient_max": max(objective_coefficients),
        "reserved_objective_overlap": sum(
            row["objective_hash"] in reserved_objectives for row in instances
        ),
    }
    expected_cells = {
        f"k{k}_m{key.split('_')[0]}_o{key.split('_')[1]}": int(value)
        for k, cells in config["cells"].items() for key, value in cells.items() if value
    }
    checks = {
        "500_instances": summary["instances"] == 500,
        "500_unique_cops": summary["unique_cops"] == 500,
        "500_unique_objectives": summary["unique_objectives"] == 500,
        "500_unique_constraint_collections": summary["unique_constraint_collections"] == 500,
        "50_unique_vcgs": summary["unique_vcgs"] == 50,
        "ten_uses_per_vcg": set(summary["vcg_uses"].values()) == {10},
        "support_distribution": summary["vcg_support"] == {k: 100 for k in range(4, 9)},
        "vcg_family_distribution": summary["vcg_family"] == {
            "equality": 120, "mixed": 130, "positive": 130, "quadratic": 120},
        "constraint_count_distribution": summary["constraint_count"] == {2: 360, 3: 140},
        "overlap_distribution": summary["overlap_count"] == {0: 250, 1: 125, 2: 125},
        "regime_distribution": summary["regime"] == {"disjoint": 250, "overlapping": 250},
        "decision_size_distribution": summary["decision_size"] == EXPECTED_SIZES,
        "exact_category_distribution": set(summary["exact_category"].values()) == {128},
        "partition_distribution": summary["partition"] == {
            f"{s}_structural_{p}_penalized": count
            for (s, p), count in EXPECTED_PARTITIONS.items()},
        "cell_distribution": summary["cell"] == expected_cells,
        "constraint_occurrences": summary["constraint_occurrences"] == 1140,
        "register_limit": summary["max_penaltyqaoa_qubits"] <= config["max_total_qubits"],
        "at_least_two_feasible": summary["feasible_count_min"] >= 2,
        "selected_vcg_fidelity": summary["vcg_fidelity_min"] >= 0.999,
        "objective_coefficient_bounds": (
            summary["objective_coefficient_min"] >= config["objective_coefficient_min"]
            and summary["objective_coefficient_max"] <= config["objective_coefficient_max"]
        ),
        "new_objectives": summary["reserved_objective_overlap"] == 0,
    }
    summary["checks"] = checks
    summary["all_checks_pass"] = all(checks.values())
    if not summary["all_checks_pass"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Dataset audit failed: {failed}")
    return summary


def ensure_manifest(config, path=HERE / "artifacts/manifest.json"):
    path = Path(path)
    specification = {key: value for key, value in config.items()
                     if key not in ["training_seed", "sampling_seed", "checkpoint_steps",
                                    "shots", "max_depth", "restarts", "steps", "learning_rate",
                                    "profile"]}
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["specification"] != specification:
            raise ValueError("Dataset configuration changed; remove or rename the old manifest")
        if manifest["hash"] != digest(manifest["instances"]):
            raise ValueError("Dataset manifest hash mismatch")
        manifest["summary"] = audit(manifest["instances"], config)
        return manifest
    instances = generate(config)
    summary = audit(instances, config)
    manifest = {"specification": specification, "hash": digest(instances),
                "summary": summary, "instances": instances}
    write_json(path, manifest)
    write_json(path.with_name("dataset_summary.json"), summary)
    return manifest

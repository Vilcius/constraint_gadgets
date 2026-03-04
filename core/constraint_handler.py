"""
constraint_handler.py -- Constraint preprocessing, classification, and partitioning.

Provides a unified interface for:
  - Parsing constraint strings ("x_0 + x_1 <= 1", "2*x_0 + x_1*x_2 == 3")
  - Classifying constraint types (Dicke-compatible, general equality, inequality)
  - Determining variable sets and detecting disjointness
  - Partitioning constraints into structural vs. penalized groups
  - Slack variable allocation for inequalities
  - Feasibility checking of bitstrings
  - Constraint normalisation and matching (for pre-computed lookup)
"""

from __future__ import annotations

import re
import itertools as it
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ======================================================================
# Enums & data classes
# ======================================================================

class ConstraintType(Enum):
    """Classification of a single constraint."""
    DICKE = auto()          # sum x_i == b  (all +1 coefficients, equality)
    FLOW = auto()           # sum_in x_i - sum_out x_j == 0  (±1 coefficients, equality, rhs=0)
    WEIGHTED_SUM = auto()   # sum c_i x_i op b  (linear, possibly inequality)
    QUADRATIC = auto()      # contains x_i * x_j terms
    GENERAL = auto()        # fallback


class ConstraintOp(Enum):
    """Relational operator."""
    EQ  = "=="
    LEQ = "<="
    GEQ = ">="
    LT  = "<"
    GT  = ">"


# Map string representations to ConstraintOp
_OP_MAP = {
    "==": ConstraintOp.EQ,
    "<=": ConstraintOp.LEQ,
    ">=": ConstraintOp.GEQ,
    "=":  ConstraintOp.EQ,
    "<":  ConstraintOp.LT,
    ">":  ConstraintOp.GT,
}

# Parse order matters: check two-character operators before single-character
_OP_PARSE_ORDER = ["==", "<=", ">=", "=", "<", ">"]


@dataclass
class ParsedConstraint:
    """
    Fully parsed representation of a single constraint string.

    Attributes
    ----------
    raw : str
        Original constraint string.
    lhs_str : str
        Left-hand side as a string.
    rhs : float
        Right-hand side numeric value.
    op : ConstraintOp
        Relational operator.
    op_symbol : str
        Original operator symbol string (e.g. "<=").
    linear : dict[int, float]
        Mapping variable_index -> coefficient for linear terms.
    quadratic : dict[tuple[int,int], float]
        Mapping (var_i, var_j) -> coefficient for quadratic terms (sorted indices).
    constant : float
        Constant term on the LHS.
    variables : set[int]
        All variable indices appearing in this constraint.
    ctype : ConstraintType
        Classified type of this constraint.
    """
    raw: str
    lhs_str: str
    rhs: float
    op: ConstraintOp
    op_symbol: str
    linear: Dict[int, float] = field(default_factory=dict)
    quadratic: Dict[Tuple[int, int], float] = field(default_factory=dict)
    constant: float = 0.0
    variables: Set[int] = field(default_factory=set)
    ctype: ConstraintType = ConstraintType.GENERAL


@dataclass
class SlackInfo:
    """
    Slack variable allocation for a single penalised inequality constraint.

    Attributes
    ----------
    constraint_idx : int
        Index into the list of penalised constraints.
    n_slack : int
        Number of slack qubits needed.
    slack_start_wire : int or None
        First wire index for the slack variables.
    operator : str
        "eq", "leq", or "geq" (after converting strict inequalities).
    effective_rhs : float
        RHS after adjusting for strict inequalities.
    """
    constraint_idx: int
    n_slack: int
    slack_start_wire: Optional[int]
    operator: str
    effective_rhs: float


# ======================================================================
# Parsing
# ======================================================================

def parse_constraint(constraint: str) -> ParsedConstraint:
    """
    Parse a single constraint string into a ``ParsedConstraint``.

    Supports formats like:
      - "x_0 + x_1 + x_2 == 2"
      - "2*x_0 + 3*x_1 <= 5"
      - "x_0*x_1 + x_2 >= 1"
      - "-x_0 + x_1*x_2 == 0"

    Parameters
    ----------
    constraint : str
        The constraint string.

    Returns
    -------
    ParsedConstraint
    """
    raw = constraint.strip()

    # --- split on operator ---
    op_symbol = None
    lhs_str = rhs_str = None
    for sym in _OP_PARSE_ORDER:
        if sym in raw:
            parts = raw.split(sym, maxsplit=1)
            lhs_str = parts[0].strip()
            rhs_str = parts[1].strip()
            op_symbol = sym
            break

    if op_symbol is None:
        raise ValueError(f"No relational operator found in constraint: '{raw}'")

    op = _OP_MAP[op_symbol]

    try:
        rhs = float(rhs_str)
    except ValueError:
        raise ValueError(f"RHS '{rhs_str}' is not a valid number in constraint: '{raw}'")

    # --- parse LHS into linear / quadratic / constant ---
    linear, quadratic, constant = _parse_lhs(lhs_str)

    variables = set(linear.keys())
    for pair in quadratic:
        variables.update(pair)

    # --- classify ---
    ctype = _classify(linear, quadratic, constant, op, rhs)

    return ParsedConstraint(
        raw=raw,
        lhs_str=lhs_str,
        rhs=rhs,
        op=op,
        op_symbol=op_symbol,
        linear=linear,
        quadratic=quadratic,
        constant=constant,
        variables=variables,
        ctype=ctype,
    )


def parse_constraints(constraints: List[str]) -> List[ParsedConstraint]:
    """Parse a list of constraint strings."""
    return [parse_constraint(c) for c in constraints]


def _parse_lhs(lhs: str) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], float]:
    """
    Parse a LHS expression into (linear, quadratic, constant).

    Returns
    -------
    linear : dict[int, float]
    quadratic : dict[tuple[int,int], float]
    constant : float
    """
    lhs = lhs.replace(" ", "")
    linear: Dict[int, float] = {}
    quadratic: Dict[Tuple[int, int], float] = {}
    constant = 0.0

    # Split preserving sign
    terms = [t for t in re.split(r"(?=[+-])", lhs) if t]

    for term in terms:
        if "x_" not in term:
            try:
                constant += float(term)
            except ValueError:
                raise ValueError(f"Cannot parse '{term}' as constant")
            continue

        # Extract all variable indices
        variables = [int(v) for v in re.findall(r"x_(\d+)", term)]

        # Extract coefficient
        coeff_match = re.match(r"^([+-]?[\d.]*)[\*]?x_", term)
        if coeff_match:
            cs = coeff_match.group(1)
            if cs in ("", "+"):
                coefficient = 1.0
            elif cs == "-":
                coefficient = -1.0
            else:
                coefficient = float(cs)
        else:
            coefficient = -1.0 if term.startswith("-") else 1.0

        if len(variables) == 1:
            idx = variables[0]
            linear[idx] = linear.get(idx, 0.0) + coefficient
        elif len(variables) == 2:
            pair = tuple(sorted(variables))
            quadratic[pair] = quadratic.get(pair, 0.0) + coefficient
        else:
            raise ValueError(f"Terms with >2 variables not supported: '{term}'")

    return linear, quadratic, constant


# ======================================================================
# Classification
# ======================================================================

def _classify(
    linear: Dict[int, float],
    quadratic: Dict[Tuple[int, int], float],
    constant: float,
    op: ConstraintOp,
    rhs: float = 0.0,
) -> ConstraintType:
    """
    Classify a parsed constraint.

    - DICKE:        all +1 linear, no quadratic, no constant, equality (sum x_i == b).
    - FLOW:         all ±1 linear (both signs present), no quadratic, no constant,
                    equality with rhs=0 (sum_in x_i - sum_out x_j == 0).
    - WEIGHTED_SUM: linear only (possibly non-unit coefficients or inequality).
    - QUADRATIC:    has quadratic terms.
    - GENERAL:      fallback.
    """
    if quadratic:
        return ConstraintType.QUADRATIC

    if not linear:
        return ConstraintType.GENERAL

    no_constant = abs(constant) < 1e-12
    is_equality = (op == ConstraintOp.EQ)

    # Dicke: all +1 coefficients, equality, no constant
    all_unit_pos = all(abs(c - 1.0) < 1e-12 for c in linear.values())
    if all_unit_pos and no_constant and is_equality:
        return ConstraintType.DICKE

    # Flow: all ±1 coefficients, both signs present, equality, rhs=0, no constant
    all_unit_abs = all(abs(abs(c) - 1.0) < 1e-12 for c in linear.values())
    has_positive = any(c > 0 for c in linear.values())
    has_negative = any(c < 0 for c in linear.values())
    rhs_zero = abs(rhs) < 1e-12
    if all_unit_abs and has_positive and has_negative and no_constant and is_equality and rhs_zero:
        return ConstraintType.FLOW

    return ConstraintType.WEIGHTED_SUM


def is_dicke_compatible(pc: ParsedConstraint) -> bool:
    """Check if a parsed constraint can be enforced with Dicke state preparation."""
    return pc.ctype == ConstraintType.DICKE


def is_flow_compatible(pc: ParsedConstraint) -> bool:
    """Check if a parsed constraint can be enforced with FlowStatePrep.

    Flow-compatible constraints have the form sum_in x_i - sum_out x_j == 0
    (all ±1 linear coefficients, both signs present, equality, rhs=0).
    """
    return pc.ctype == ConstraintType.FLOW


# ======================================================================
# Variable analysis & disjointness
# ======================================================================

def get_variable_set(pc: ParsedConstraint) -> Set[int]:
    """Return the set of variable indices in a constraint."""
    return pc.variables


def are_disjoint(pc_a: ParsedConstraint, pc_b: ParsedConstraint) -> bool:
    """Check whether two constraints act on completely disjoint variable sets."""
    return pc_a.variables.isdisjoint(pc_b.variables)


def find_disjoint_groups(constraints: List[ParsedConstraint]) -> List[List[int]]:
    """
    Partition constraint indices into maximal groups of mutually disjoint constraints.

    Uses a simple greedy/union-find approach: two constraints are in the same
    group if they share any variable (transitively).

    Parameters
    ----------
    constraints : list[ParsedConstraint]

    Returns
    -------
    groups : list[list[int]]
        Each inner list contains indices into the input list. Constraints within
        a group share variables (directly or transitively). Constraints in
        different groups are variable-disjoint.
    """
    n = len(constraints)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Union constraints that share variables
    for i in range(n):
        for j in range(i + 1, n):
            if not constraints[i].variables.isdisjoint(constraints[j].variables):
                union(i, j)

    # Collect groups
    groups_map: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        groups_map.setdefault(root, []).append(i)

    return list(groups_map.values())


# ======================================================================
# Partitioning: structural vs. penalized
# ======================================================================

def partition_constraints(
    constraints: List[ParsedConstraint],
    strategy: str = "auto",
    max_structural_vars: int = 12,
) -> Tuple[List[int], List[int]]:
    """
    Partition constraints into structural (gadget-enforced) and penalized groups.

    Strategy options:
    - "auto":     Dicke-compatible and small disjoint groups go structural;
                  everything else is penalized.
    - "all_structural": All constraints are structural (original QAOA+ approach).
    - "all_penalty":    All constraints are penalized (standard penalty QAOA).
    - "dicke_only":     Only Dicke-compatible constraints are structural.

    Parameters
    ----------
    constraints : list[ParsedConstraint]
    strategy : str
        Partitioning strategy.
    max_structural_vars : int
        For "auto": maximum number of variables in a structural constraint group
        before it gets pushed to penalty (truth-table scales as 2^n).

    Returns
    -------
    structural_indices : list[int]
        Indices of constraints to enforce structurally.
    penalty_indices : list[int]
        Indices of constraints to enforce via penalty.
    """
    n = len(constraints)

    if strategy == "all_structural":
        return list(range(n)), []
    if strategy == "all_penalty":
        return [], list(range(n))
    if strategy == "dicke_only":
        structural = [i for i in range(n)
                      if is_dicke_compatible(constraints[i]) or is_flow_compatible(constraints[i])]
        penalty = [i for i in range(n) if i not in structural]
        return structural, penalty

    # --- "auto" strategy ---
    # Step 1: find disjoint groups
    groups = find_disjoint_groups(constraints)

    structural_indices: List[int] = []
    penalty_indices: List[int] = []

    for group_idxs in groups:
        # Variables in this group
        group_vars = set()
        for idx in group_idxs:
            group_vars |= constraints[idx].variables

        # Dicke and Flow constraints are always cheap to enforce structurally
        cheap_idxs = [i for i in group_idxs
                      if is_dicke_compatible(constraints[i]) or is_flow_compatible(constraints[i])]
        other_idxs = [i for i in group_idxs
                      if not is_dicke_compatible(constraints[i]) and not is_flow_compatible(constraints[i])]

        structural_indices.extend(cheap_idxs)

        # Non-Dicke/Flow: use structural (gadget) if the group is small enough
        other_vars = set()
        for idx in other_idxs:
            other_vars |= constraints[idx].variables

        if len(other_vars) <= max_structural_vars and other_idxs:
            structural_indices.extend(other_idxs)
        else:
            penalty_indices.extend(other_idxs)

    return sorted(structural_indices), sorted(penalty_indices)


# ======================================================================
# Slack variable determination
# ======================================================================

def determine_slack_variables(
    constraints: List[ParsedConstraint],
    slack_wire_offset: int,
) -> Tuple[List[SlackInfo], int]:
    """
    Determine slack variable count and wire allocation for penalised constraints.

    Inequality constraints are converted to equalities by adding slack variables:
      - sum(c_i x_i) + sum(s_j) == b   for <=
      - sum(c_i x_i) - sum(s_j) == b   for >=

    Parameters
    ----------
    constraints : list[ParsedConstraint]
        The penalised constraints.
    slack_wire_offset : int
        First available wire index for slack qubits.

    Returns
    -------
    slack_infos : list[SlackInfo]
        One entry per constraint.
    total_slack : int
        Total number of slack qubits allocated.
    """
    slack_infos = []
    current_wire = slack_wire_offset

    for i, pc in enumerate(constraints):
        # Compute LHS value range
        max_lhs = pc.constant + sum(c for c in pc.linear.values() if c > 0) \
                               + sum(c for c in pc.quadratic.values() if c > 0)
        min_lhs = pc.constant + sum(c for c in pc.linear.values() if c < 0) \
                               + sum(c for c in pc.quadratic.values() if c < 0)

        if pc.op in (ConstraintOp.LEQ, ConstraintOp.LT):
            eff_rhs = pc.rhs if pc.op == ConstraintOp.LEQ else pc.rhs - 1
            n_slack = max(0, int(np.ceil(eff_rhs - min_lhs)))
            slack_infos.append(SlackInfo(
                constraint_idx=i,
                n_slack=n_slack,
                slack_start_wire=current_wire,
                operator="leq",
                effective_rhs=eff_rhs,
            ))
            current_wire += n_slack

        elif pc.op in (ConstraintOp.GEQ, ConstraintOp.GT):
            eff_rhs = pc.rhs if pc.op == ConstraintOp.GEQ else pc.rhs + 1
            n_slack = max(0, int(np.ceil(max_lhs - eff_rhs)))
            slack_infos.append(SlackInfo(
                constraint_idx=i,
                n_slack=n_slack,
                slack_start_wire=current_wire,
                operator="geq",
                effective_rhs=eff_rhs,
            ))
            current_wire += n_slack

        else:  # EQ
            slack_infos.append(SlackInfo(
                constraint_idx=i,
                n_slack=0,
                slack_start_wire=None,
                operator="eq",
                effective_rhs=pc.rhs,
            ))

    total_slack = current_wire - slack_wire_offset
    return slack_infos, total_slack


# ======================================================================
# Feasibility checking
# ======================================================================

def check_feasibility(
    bitstring: str,
    constraints: List[ParsedConstraint],
    n_x: Optional[int] = None,
) -> bool:
    """
    Check whether a bitstring satisfies all constraints.

    Only the first n_x bits (decision variables) are used; flag/slack bits
    are ignored.

    Parameters
    ----------
    bitstring : str
        Binary string (e.g. "01101").
    constraints : list[ParsedConstraint]
    n_x : int or None
        Number of decision variable bits. If None, uses len(bitstring).

    Returns
    -------
    bool
    """
    if n_x is None:
        n_x = len(bitstring)
    x = bitstring[:n_x]

    for pc in constraints:
        lhs_val = pc.constant
        for idx, coeff in pc.linear.items():
            if idx < len(x):
                lhs_val += coeff * int(x[idx])
        for (i, j), coeff in pc.quadratic.items():
            if i < len(x) and j < len(x):
                lhs_val += coeff * int(x[i]) * int(x[j])

        if pc.op == ConstraintOp.EQ and not np.isclose(lhs_val, pc.rhs, atol=1e-6):
            return False
        elif pc.op == ConstraintOp.LEQ and lhs_val > pc.rhs + 1e-6:
            return False
        elif pc.op == ConstraintOp.LT and lhs_val >= pc.rhs - 1e-6:
            return False
        elif pc.op == ConstraintOp.GEQ and lhs_val < pc.rhs - 1e-6:
            return False
        elif pc.op == ConstraintOp.GT and lhs_val <= pc.rhs + 1e-6:
            return False

    return True


def evaluate_lhs(pc: ParsedConstraint, x_bits: str) -> float:
    """Evaluate the LHS of a constraint for a given binary assignment."""
    val = pc.constant
    for idx, coeff in pc.linear.items():
        if idx < len(x_bits):
            val += coeff * int(x_bits[idx])
    for (i, j), coeff in pc.quadratic.items():
        if i < len(x_bits) and j < len(x_bits):
            val += coeff * int(x_bits[i]) * int(x_bits[j])
    return val


# ======================================================================
# Constraint normalisation and matching (for pre-computed lookup)
# ======================================================================

class ConstraintMapper:
    """
    Map input constraints to a pre-existing set in a dataframe by checking
    all permutations of variable names and constraint ordering.

    This allows looking up pre-computed gadget data even when variable
    indices are permuted.
    """

    def __init__(self, constraints_in_df: List[List[str]]) -> None:
        self.constraints_in_df = constraints_in_df

    def normalize_constraint(self, constraint: str) -> str:
        """Normalize by removing spaces and sorting additive terms."""
        constraint = re.sub(r"\s+", "", constraint)
        op_match = re.search(r"(==|<=|>=|=|<|>)", constraint)
        if not op_match:
            return constraint
        operator = op_match.group(0)
        lhs, rhs = constraint.split(operator, maxsplit=1)
        terms = sorted(re.split(r"(\+)", lhs))
        return f"{''.join(terms)}{operator}{rhs}"

    def normalize_constraints(self, constraints: List[str]) -> List[str]:
        return [self.normalize_constraint(c) for c in constraints]

    def map_constraints(self, input_constraints: List[str]) -> Optional[List[str]]:
        """
        Find a matching constraint set in the dataframe, accounting for
        variable and constraint permutations.

        Returns the matched constraint list from the dataframe, or None.
        """
        normalized_input = self.normalize_constraints(input_constraints)
        for constraints in self.constraints_in_df:
            normalized_df = self.normalize_constraints(constraints)
            if self._match(normalized_input, normalized_df):
                return constraints
        return None

    def _match(
        self,
        input_constraints: List[str],
        df_constraints: List[str],
    ) -> bool:
        input_vars = sorted(set(re.findall(r"x_\d+", " ".join(input_constraints))))
        df_vars = sorted(set(re.findall(r"x_\d+", " ".join(df_constraints))))

        if len(input_vars) != len(df_vars):
            return False

        for perm in it.permutations(df_vars):
            var_map = {iv: perm[k] for k, iv in enumerate(input_vars)}
            mapped = [
                re.sub(r"x_\d+", lambda m: var_map[m.group(0)], c)
                for c in input_constraints
            ]
            for perm_c in it.permutations(mapped):
                if self._check_perm(perm_c, df_constraints):
                    return True
        return False

    @staticmethod
    def _check_perm(perm_constraints, df_constraints) -> bool:
        for pc in perm_constraints:
            op_match = re.search(r"(==|<=|>=|=|<|>)", pc)
            if not op_match:
                continue
            pc_op = op_match.group(0)
            for dc in df_constraints:
                dc_op_match = re.search(r"(==|<=|>=|=|<|>)", dc)
                if not dc_op_match:
                    continue
                dc_op = dc_op_match.group(0)
                lhs_pc = re.split(r"(\+)", pc.split(pc_op)[0])
                lhs_dc = re.split(r"(\+)", dc.split(dc_op)[0])
                rhs_pc = pc.split(pc_op)[1]
                rhs_dc = dc.split(dc_op)[1]
                if (sorted(lhs_pc) == sorted(lhs_dc)
                        and rhs_pc == rhs_dc
                        and pc_op == dc_op):
                    return True
        return False

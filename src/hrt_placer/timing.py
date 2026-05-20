"""Topology-derived net-criticality weighting.

The challenge gives us a netlist (nets + node lists) and I/O port positions
but no Liberty file or timing arcs, so a real STA is out of reach inside
the placer. Instead we estimate which nets are likely on long combinational
paths from netlist topology alone: nets deep in the graph (far from any
I/O port) are more likely to bound critical-path delay than nets adjacent
to I/O.

The result is a per-net weight tensor in [1, 1 + alpha] that callers can
assign to ``benchmark.net_weights`` -- ``engine.AnalyticalPlacer`` already
multiplies its HPWL term by ``benchmark.net_weights``, so heavier nets are
preferentially shortened.

This is a coarse surrogate, not a real timer. ``select.py`` keeps a
proxy-cost guard so the weighting is only adopted when it does not
regress unweighted proxy beyond a small budget.
"""

from __future__ import annotations

from collections import deque

import torch


def net_criticality_weights(
    benchmark,
    alpha: float = 1.0,
    max_pins: int = 64,
) -> torch.Tensor:
    """Per-net weights derived from logic depth from I/O.

    Builds an undirected graph over macros plus port nodes using nets of
    size <= ``max_pins`` (so clock / reset trees do not collapse depth to
    1). Runs a multi-source BFS from every port; each macro's depth is
    the shortest hop count to any port. A net's criticality is the mean
    normalised depth of its in-graph members, weight = 1 + alpha * crit.

    Falls back to uniform weights when net_nodes is unpopulated or no
    port-to-macro connectivity exists.
    """
    num_nets = int(benchmark.num_nets)
    weights = torch.ones(num_nets, dtype=torch.float32)
    if num_nets == 0:
        return weights

    net_nodes = benchmark.net_nodes
    if len(net_nodes) != num_nets:
        return weights

    n_macros = int(benchmark.num_macros)
    n_ports = int(benchmark.port_positions.shape[0])
    if n_ports == 0:
        return weights
    n_total = n_macros + n_ports

    adj: list[list[int]] = [[] for _ in range(n_total)]
    is_signal = [False] * num_nets
    nodes_per_net: list[list[int]] = [[] for _ in range(num_nets)]
    for i in range(num_nets):
        nodes = net_nodes[i].tolist()
        # Drop out-of-range indices defensively (port indices in net_nodes
        # are offset by num_macros in this codebase, but skip anything
        # outside the combined macro+port range).
        nodes = [v for v in nodes if 0 <= v < n_total]
        if not nodes or len(nodes) > max_pins:
            continue
        is_signal[i] = True
        nodes_per_net[i] = nodes
        # Star model: hub = first node, connect every other node to it.
        # Cheaper than a clique and gives identical BFS hop counts on the
        # downstream macro graph.
        hub = nodes[0]
        for v in nodes[1:]:
            adj[hub].append(v)
            adj[v].append(hub)

    depth = [-1] * n_total
    q: deque[int] = deque()
    for p in range(n_macros, n_total):
        depth[p] = 0
        q.append(p)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if depth[v] == -1:
                depth[v] = depth[u] + 1
                q.append(v)

    max_d = max(depth)
    if max_d <= 0:
        return weights

    for i in range(num_nets):
        if not is_signal[i]:
            continue
        ds = [depth[v] for v in nodes_per_net[i] if depth[v] > 0]
        if not ds:
            continue
        crit = (sum(ds) / len(ds)) / max_d
        weights[i] = 1.0 + alpha * crit

    return weights


def criticality_weighted_wirelength(
    placement: torch.Tensor, benchmark, weights: torch.Tensor
) -> float:
    """Sum of weight * HPWL over all nets, normalised by canvas.

    Wirelength surrogate that emphasises the same nets ``weights``
    emphasises. Used by ``select.py`` as a timing-side scoring signal.
    """
    num_nets = int(benchmark.num_nets)
    if num_nets == 0 or len(benchmark.net_nodes) != num_nets:
        return 0.0
    n_macros = int(benchmark.num_macros)
    n_ports = int(benchmark.port_positions.shape[0])
    pos = placement[:n_macros, :2].float()
    if n_ports > 0:
        all_pos = torch.cat([pos, benchmark.port_positions.float()], dim=0)
    else:
        all_pos = pos
    W = float(benchmark.canvas_width)
    H = float(benchmark.canvas_height)
    total = 0.0
    w_np = weights.float().tolist()
    for i in range(num_nets):
        nodes = benchmark.net_nodes[i].tolist()
        if not nodes:
            continue
        coords = all_pos[torch.tensor(nodes, dtype=torch.long)]
        bb = coords.max(dim=0).values - coords.min(dim=0).values
        total += w_np[i] * float(bb.sum())
    return total / ((W + H) * num_nets)

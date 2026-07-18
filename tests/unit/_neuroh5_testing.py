"""Shared fixture generators and comparison helpers for the neuroh5 pytest suite.

Imported both by pytest test modules (single-rank scenarios) and by the
standalone scripts in mpi_workers/ (multi-rank scenarios launched via
``mpirun``). Keeping generation logic here means a worker subprocess can
regenerate the exact same expected data a parent test built, from the same
(seed, count) parameters, without having to serialize it across the process
boundary.
"""

import numpy as np
import h5py


# ---------------------------------------------------------------------------
# Population / H5Types fixture setup
# ---------------------------------------------------------------------------

def create_populations_file(path, pop_defs, pop_combs=None):
    """Create the minimal /H5Types groups a NeuroH5 file needs before any
    graph/attribute/tree data can be written to it.

    pop_defs: list of (name, start, count, idx) tuples, idx ascending from 0.
    pop_combs: optional list of (src_idx, dst_idx) pairs describing valid
        projections; defaults to the full cross product of population
        indices (including self-projections).
    """
    grp_h5types = "H5Types"
    path_population_labels = "/%s/Population labels" % grp_h5types

    with h5py.File(path, "a") as h5:
        mapping = {name: idx for (name, start, count, idx) in pop_defs}
        dt_enum = h5py.special_dtype(enum=(np.uint16, mapping))
        h5[path_population_labels] = dt_enum

        g = h5.require_group(grp_h5types)
        dt_pop = np.dtype([
            ("Start", np.uint64),
            ("Count", np.uint32),
            ("Population", h5[path_population_labels].dtype),
        ])
        arr = np.zeros(len(pop_defs), dtype=dt_pop)
        for i, (name, start, count, idx) in enumerate(pop_defs):
            arr[i]["Start"] = start
            arr[i]["Count"] = count
            arr[i]["Population"] = idx
        dset = g.create_dataset("Populations", shape=(len(pop_defs),), dtype=dt_pop)
        dset[:] = arr

        if pop_combs is None:
            idxs = [idx for (_, _, _, idx) in pop_defs]
            pop_combs = [(s, d) for s in idxs for d in idxs]
        dt_comb = np.dtype([("Source", np.uint16), ("Destination", np.uint16)])
        comb_arr = np.zeros(len(pop_combs), dtype=dt_comb)
        for i, (s, d) in enumerate(pop_combs):
            comb_arr[i]["Source"] = s
            comb_arr[i]["Destination"] = d
        combset = g.create_dataset(
            "Valid population projections", shape=(len(pop_combs),), dtype=dt_comb
        )
        combset[:] = comb_arr


# ---------------------------------------------------------------------------
# Graph (DBS) fixtures
# ---------------------------------------------------------------------------

def make_edges(dst_start, dst_count, src_start, src_count, seed=0,
               min_deg=1, max_deg=4, with_attrs=False):
    """Deterministic synthetic edge dict in the format write_graph/append_graph
    expect: {dst_gid: (src_id_array_uint32, {namespace: {attr: array}})}.

    Degree per destination varies with gid (via the seeded RNG) so that
    per-rank edge counts are intentionally uneven when scattered.
    """
    rng = np.random.default_rng(seed)
    edges = {}
    for i in range(dst_count):
        dst = dst_start + i
        deg = int(rng.integers(min_deg, max_deg + 1))
        deg = min(deg, src_count)
        srcs = src_start + rng.choice(src_count, size=deg, replace=False)
        srcs = np.sort(srcs.astype(np.uint32))
        attrs = {}
        if with_attrs:
            attrs = {
                "Synapses": {
                    "weight": rng.random(deg).astype(np.float32),
                    "syn_id": (rng.integers(0, 1 << 16, size=deg)).astype(np.uint32),
                }
            }
        edges[int(dst)] = (srcs, attrs)
    return edges


def normalize_edge_attrs(edges, name_index, namespace="Synapses"):
    """Convert the read-side edge dict (attrs given as a flat, positionally
    ordered list per namespace) into the write-side shape (attrs given as a
    {name: array} dict), using the {name: position} mapping returned
    alongside the data (e.g. attr_info['PopB']['PopA']['Synapses']).
    """
    out = {}
    for dst, (src, attrs) in edges.items():
        ns_list = attrs.get(namespace, [])
        ns_dict = {name: ns_list[idx] for name, idx in name_index.items()}
        out[dst] = (src, {namespace: ns_dict})
    return out


def edges_equal(a, b, with_attrs=False):
    """Compare two edge dicts of the {dst: (src_array, {ns: {attr: arr}})} shape."""
    if set(a.keys()) != set(b.keys()):
        return False
    for dst in a:
        src_a, attrs_a = a[dst]
        src_b, attrs_b = b[dst]
        if not np.array_equal(np.asarray(src_a), np.asarray(src_b)):
            return False
        if with_attrs:
            ns = "Synapses"
            if ns not in attrs_b:
                return False
            for name in ("weight", "syn_id"):
                if name not in attrs_a[ns] or name not in attrs_b[ns]:
                    return False
                va = np.asarray(attrs_a[ns][name])
                vb = np.asarray(attrs_b[ns][name])
                if not np.allclose(va, vb):
                    return False
    return True


# ---------------------------------------------------------------------------
# Cell attribute fixtures
# ---------------------------------------------------------------------------

def make_cell_attrs(pop_start, pop_count, seed=0):
    """Deterministic synthetic per-cell attribute dict for
    write_cell_attributes/append_cell_attributes: {gid: {attr_name: array}}.
    """
    rng = np.random.default_rng(seed)
    attrs = {}
    for i in range(pop_count):
        gid = pop_start + i
        n = int(rng.integers(1, 4))
        attrs[int(gid)] = {
            "a": (rng.integers(0, 1000, size=n)).astype(np.uint32),
            "b": rng.random(1).astype(np.float32),
        }
    return attrs


def cell_attrs_equal(a, b):
    if set(a.keys()) != set(b.keys()):
        return False
    for gid in a:
        for name in a[gid]:
            if name not in b[gid]:
                return False
            va = np.asarray(a[gid][name])
            vb = np.asarray(b[gid][name])
            if va.dtype.kind == "f":
                if not np.allclose(va, vb):
                    return False
            elif not np.array_equal(va, vb):
                return False
    return True


# ---------------------------------------------------------------------------
# Tree fixtures
#
# The 'sections' array packs the section topology as required by
# neuroh5::cell::validate_tree: [num_sections, (num_nodes_in_section,
# 1-based node indices...) for each section]. Every point must belong to
# exactly one section, and the src/dst section-graph must have exactly one
# root (a section with no incoming edges). The simplest tree satisfying
# this has two sections (root + child) connected by a single src->dst edge.
# ---------------------------------------------------------------------------

def make_tree(n_pts, gid, seed=0):
    rng = np.random.default_rng(seed + gid)
    n_sec0 = max(1, n_pts // 2)
    n_sec1 = n_pts - n_sec0
    if n_sec1 > 0:
        sections = [2, n_sec0] + list(range(1, n_sec0 + 1))
        sections += [n_sec1] + list(range(n_sec0 + 1, n_pts + 1))
        src = np.array([0], dtype=np.uint16)
        dst = np.array([1], dtype=np.uint16)
    else:
        sections = [1, n_sec0] + list(range(1, n_sec0 + 1))
        src = np.array([0], dtype=np.uint16)
        dst = np.array([0], dtype=np.uint16)
    return {
        "x": rng.random(n_pts).astype(np.float32) + gid,
        "y": rng.random(n_pts).astype(np.float32),
        "z": rng.random(n_pts).astype(np.float32),
        "radius": np.full(n_pts, 1.0, dtype=np.float32),
        "layer": np.zeros(n_pts, dtype=np.int8),
        "parent": (np.arange(n_pts, dtype=np.int32) - 1),
        "swc_type": np.ones(n_pts, dtype=np.int8),
        "sections": np.array(sections, dtype=np.uint16),
        "src": src,
        "dst": dst,
    }


def make_trees(pop_start, pop_count, min_pts=4, max_pts=8, seed=0):
    rng = np.random.default_rng(seed)
    trees = {}
    for i in range(pop_count):
        gid = pop_start + i
        n_pts = int(rng.integers(min_pts, max_pts + 1))
        trees[int(gid)] = make_tree(n_pts, int(gid), seed=seed)
    return trees


def trees_equal(a, b):
    if set(a.keys()) != set(b.keys()):
        return False
    for gid in a:
        ta, tb = a[gid], b[gid]
        for name in ("x", "y", "z", "radius", "swc_type"):
            va = np.asarray(ta[name])
            vb = np.asarray(tb[name])
            if va.shape != vb.shape or not np.allclose(va, vb, atol=1e-5):
                return False
    return True

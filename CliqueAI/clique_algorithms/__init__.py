from .networkx_algorithm import networkx_algorithm


def __getattr__(name):
    # The GNN algorithm imports torch and loads a checkpoint at import time (~1 GB RAM, seconds of
    # startup), so only load it when someone actually asks for it.
    if name == "scattering_clique_algorithm":
        from .gnn_algorithm import scattering_clique_algorithm

        return scattering_clique_algorithm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

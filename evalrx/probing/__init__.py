"""Probes as tools: discover, route and call analyzers by name.

    import evalrx
    from evalrx.probing import ProbeToolbox

    box = ProbeToolbox(model, cases)
    box.describe()                                    # availability + reasons
    box.route(failure_modes=["visual_neglect"]).names # what to run first
    box.call("attention_knockout", {"queries": "last"})

See :mod:`evalrx.probing.catalog` for the manifest,
:mod:`evalrx.probing.router` for ranking, and
:mod:`evalrx.probing.toolbox` for execution.
"""

from evalrx.probing.catalog import (
    FAMILY_QUESTIONS,
    ProbeAvailability,
    ProbeParam,
    ProbeToolSpec,
    availability,
    describe_probes,
    family_of,
    probe_specs,
    spec_for,
)
from evalrx.probing.router import ProbeRoute, ProbeRouter, RoutedProbe
from evalrx.probing.toolbox import (
    ProbeCallError,
    ProbeToolbox,
    build_probe,
    check_params,
    summarize_result,
)

__all__ = [
    "FAMILY_QUESTIONS",
    "ProbeAvailability",
    "ProbeCallError",
    "ProbeParam",
    "ProbeRoute",
    "ProbeRouter",
    "ProbeToolSpec",
    "ProbeToolbox",
    "RoutedProbe",
    "availability",
    "build_probe",
    "check_params",
    "describe_probes",
    "family_of",
    "probe_specs",
    "spec_for",
    "summarize_result",
]

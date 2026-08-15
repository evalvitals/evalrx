"""Architecture-aware executors for published model-health repairs.

These modules are deliberately small and explicit.  A method is exposed here
only when its model access contract can be checked by the backend; an example
or benchmark runner must not silently relabel a prompt trick as the paper
method.
"""


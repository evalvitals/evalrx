"""Write-side white-box plumbing shared by the intervention analyzers.

The read-only analyzers consume a :class:`~evalrx.core.model.Trace`. The ones
here (attention knockout, causal tracing, gradient attribution) have to CHANGE
or DIFFERENTIATE the forward pass, which a ``Trace`` cannot express. They do it
by installing hooks on the live torch module for the duration of one ordinary
``model.forward`` call, so encoding, chat templates and the image-token map stay
exactly what every other analyzer sees.

    access     — the torch module behind a handle, the decoder stack, media positions
    targets    — the logit margin every intervention is scored on
    attn_tap   — a registered attention function that can edit/record attention
    residual   — read/replace the residual stream entering a decoder layer

Private package: torch is imported inside functions only, so importing an
analyzer that uses it still works on the light (API-only) install.
"""

'''Adam over fp32 MASTER copies of low-precision parameters.

Plain Adam stepping bf16 leaves directly loses every update smaller than
half an ULP at the weight's magnitude (measured on r2 exp004: all 28 relay
gates at -3.0, half-ULP there 0.0078, implied Adam steps ~4.5e-4 — each one
rounded to a no-op for the entire run). This optimizer keeps fp32 masters,
steps THEM, and copies back down, so sub-ULP progress accumulates in fp32
and surfaces in bf16 once it crosses the ULP.

Why client-side instead of ds_config bf16: this fork's train.py initializes
the DeepSpeed engine, then calls _configure_optimizer and hands
model_engine.optimizer to torch lr schedulers. Enabling DeepSpeed's own
fp16/bf16 modes makes the engine wrap the client optimizer
(FP16_UnfusedOptimizer), which torch schedulers reject ("... is not an
Optimizer"). This class IS a torch.optim.Adam — the master groups are its
param_groups, so schedulers mutate lr exactly as with plain Adam — and the
engine stays on the plain, exp004-proven no-wrapping path.

Resume caveat: state_dict() carries Adam moments but not the master VALUES
(torch never serializes params inside optimizer state); after a resume the
masters re-seed from the live bf16 weights, losing only the accumulated
sub-ULP residue (bounded by half an ULP per weight).
'''
import torch


class MasterWeightsAdam(torch.optim.Adam):
    def __init__(self, params, **kwargs):
        param_groups = list(params)
        if param_groups and not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]
        self._live_groups = []
        master_groups = []
        for g in param_groups:
            live = list(g['params'])
            masters = [p.detach().clone().float() for p in live]
            for m in masters:
                m.requires_grad_(False)      # grads are copied in manually
            self._live_groups.append(live)
            mg = {k: v for k, v in g.items() if k != 'params'}
            mg['params'] = masters
            master_groups.append(mg)
        super().__init__(master_groups, **kwargs)

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        for live in self._live_groups:
            for p in live:
                if set_to_none:
                    p.grad = None
                elif p.grad is not None:
                    p.grad.zero_()

    def step(self, closure=None):
        with torch.no_grad():
            for live, mg in zip(self._live_groups, self.param_groups):
                for p, m in zip(live, mg['params']):
                    m.grad = (p.grad.float() if p.grad is not None else None)
        loss = super().step(closure)
        with torch.no_grad():
            for live, mg in zip(self._live_groups, self.param_groups):
                for p, m in zip(live, mg['params']):
                    p.copy_(m)               # downcasts on copy
        return loss

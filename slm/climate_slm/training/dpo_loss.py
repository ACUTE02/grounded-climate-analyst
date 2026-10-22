"""The DPO (sigmoid) loss and its reporting numbers, as small pure functions. No torch needed: they work on plain
floats (laptop tests) and, with `logsigmoid=torch.nn.functional.logsigmoid`, on torch tensors (Colab trainer).

For one pair, with sequence log-probabilities (sum over the answer tokens) of the policy (pc, pr) and of the frozen
reference (rc, rr), for the chosen (c) and the rejected (r) answer:

    z      = beta * ((pc - rc) - (pr - rr))
    loss   = -log sigmoid(z)                     (no SFT term)
    reward = beta * (policy - reference) of each answer;  margin = reward_chosen - reward_rejected = z
    reward accuracy = share of pairs with margin > 0

At the start (policy == reference) z = 0, the loss is log 2 = 0.6931 and the accuracy is 0 (no pair has margin > 0)."""
import math
from dataclasses import dataclass


def logsigmoid(x: float) -> float:
    """log(1 / (1 + exp(-x))), stable for large |x|."""
    if x >= 0:
        return -math.log1p(math.exp(-x))
    return x - math.log1p(math.exp(x))


def dpo_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta: float, logsigmoid=logsigmoid):
    """-log sigmoid(beta * ((pc - rc) - (pr - rr))). Floats give a float, torch tensors give a tensor
    (pass torch's logsigmoid)."""
    z = beta * ((pol_chosen - ref_chosen) - (pol_rejected - ref_rejected))
    return -logsigmoid(z)


def dpo_terms(pol_chosen: float, pol_rejected: float, ref_chosen: float, ref_rejected: float, beta: float) -> dict:
    """The loss and the reward numbers of one pair, all plain floats."""
    rc, rr = beta * (pol_chosen - ref_chosen), beta * (pol_rejected - ref_rejected)
    margin = rc - rr
    return {"loss": dpo_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta), "reward_chosen": rc,
            "reward_rejected": rr, "margin": margin, "correct": margin > 0}


def nll_per_token(logp_sum, n_tokens: int):
    """Negative log-likelihood of a text per token: -(sum of token log-probs) / number of tokens."""
    return -logp_sum / n_tokens


def rpo_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta: float, alpha: float, n_chosen_tokens: int,
             logsigmoid=logsigmoid):
    """DPO v2 (RPO style): sigmoid DPO + alpha * (per-token NLL of the CHOSEN text under the policy).
    alpha = 0 is exactly dpo_loss. Floats give a float, torch tensors give a tensor."""
    return dpo_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta, logsigmoid)         + alpha * nll_per_token(pol_chosen, n_chosen_tokens)


def rpo_terms(pol_chosen: float, pol_rejected: float, ref_chosen: float, ref_rejected: float, beta: float, alpha: float,
              n_chosen_tokens: int) -> dict:
    """dpo_terms plus the NLL term: loss = dpo_loss + alpha * nll (all plain floats)."""
    t = dpo_terms(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta)
    nll = nll_per_token(pol_chosen, n_chosen_tokens)
    return {**t, "dpo_loss": t["loss"], "nll": nll, "loss": t["loss"] + alpha * nll}


@dataclass
class Meter:
    """Running mean of the per-pair numbers over a step or an epoch."""
    n: int = 0
    loss: float = 0.0
    margin: float = 0.0
    reward_chosen: float = 0.0
    reward_rejected: float = 0.0
    correct: int = 0
    dpo: float = 0.0
    nll: float = 0.0

    def add(self, terms: dict) -> None:
        self.n += 1
        self.loss += terms["loss"]
        self.margin += terms["margin"]
        self.reward_chosen += terms["reward_chosen"]
        self.reward_rejected += terms["reward_rejected"]
        self.correct += bool(terms["correct"])
        self.dpo += terms.get("dpo_loss", terms["loss"])         # v1 terms have no NLL part
        self.nll += terms.get("nll", 0.0)

    def summary(self) -> dict:
        if not self.n:
            return {"n_pairs": 0, "loss": None, "reward_accuracy": None, "mean_margin": None,
                    "mean_reward_chosen": None, "mean_reward_rejected": None, "dpo_loss": None, "nll": None}
        return {"n_pairs": self.n, "loss": self.loss / self.n, "reward_accuracy": self.correct / self.n,
                "mean_margin": self.margin / self.n, "mean_reward_chosen": self.reward_chosen / self.n,
                "mean_reward_rejected": self.reward_rejected / self.n, "dpo_loss": self.dpo / self.n, "nll": self.nll / self.n}

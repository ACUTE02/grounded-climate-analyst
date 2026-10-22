"""Plain-text fill-in sheet for the owner: do the refusal items really have no answer in their context?

The sheet shows the question and the full context only. No model output, no reference answer, no strategy,
no AI verdict. The labels must be judged from the context alone. Nothing in the data files is changed."""
import random

from climate_slm.config import REPO_ROOT
from climate_slm.data.sft import parse_example
from climate_slm.evaluation.scoring import kind_of, load_items

LINE = "-" * 70
BIG = "=" * 70

INTRO = """REFUSAL REVIEW - FILL KARNE KA DOCUMENT
======================================

PEHLE YE PADHO (2 minute)

Kya karna hai?
  Neeche {n} items hain ({n_test} test + {n_val} val). Har item mein ek QUESTION aur ek CONTEXT (source ka
  text) hai. Tumhe bas batana hai: kya is CONTEXT mein is QUESTION ka jawab hai ya nahi.

Kyun?
  Ye saare items "refusal" items hain. Matlab hum chahte hain ki model in sawalon par ye bole:
  "I cannot answer this from the provided context."
  Pichli baar spot-check mein refusal wale items par 'good' / 'bad' ka matlab saaf nahi tha. Isliye
  is baar sirf A, B ya C.

  Is sheet mein kisi model ka jawab NAHI hai, jaan-boojh ke. Sirf question aur context dekh kar judge karo.

Har item ke liye 3 cheezein karo:
  1. QUESTION padho.
  2. CONTEXT mein dekho (Ctrl+F se question ke keywords dhundo).
  3. Neeche "CHOICE:" ke baad sirf ek letter likho: A, B ya C.

Choice kaise likhna hai:
  A = context mein is question ka jawab sach mein NAHI hai. (Refusal sahi hai.)
  B = context mein question ka jawab hai (poora). Ye item refusal nahi hona chahiye.
  C = context mein jawab ka sirf kuch hissa hai, ya andaza lagana padega. (Adhoora jawab.)

Shak ho to C likho aur NOTE mein ek chhoti line likh do. NOTE optional hai, par B aur C mein likhna helpful hai.

Kaise fill karna hai:
  Har item ke neeche "CHOICE:" ke baad sirf A, B ya C likho. "NOTE:" ke baad kuch bhi (khaali bhi chalega).
  Kuch aur edit mat karo. Poori file bhar ke wapas bhej do.
  Is sheet se data apne aap nahi badlega. Jo items B ya C honge unka kya karna hai, wo reviewer baad mein decide karega.

Time: ek item 1-2 minute. Total lagbhag 30-40 minute. Break le sakte ho.
"""


def refusal_items(cfg: dict) -> list[dict]:
    """All refusal items of the configured splits (test + val), in a seeded shuffled order."""
    items = []
    for split in cfg["refusal_review"]["splits"]:
        items += [it for it in load_items(REPO_ROOT / cfg["paths"]["splits"][split]) if kind_of(it) == "refusal"]
    random.Random(cfg["refusal_review"]["seed"]).shuffle(items)
    return items


def build_sheet(cfg: dict) -> str:
    items = refusal_items(cfg)
    by_split = {s: sum(it["meta"]["split"] == s for it in items) for s in ("test", "val")}
    out = [INTRO.format(n=len(items), n_test=by_split["test"], n_val=by_split["val"]), BIG, ""]
    for k, it in enumerate(items, 1):
        context, question, _ = parse_example(it)
        out += [LINE, f"ITEM {k}/{len(items)}   ID: {it['meta']['id']}   SOURCE: {it['meta']['source_id']}", LINE,
                f"QUESTION: {question}", "", "CONTEXT:", context.strip(), "", "CHOICE: ", "NOTE: ", ""]
    return "\n".join(out)

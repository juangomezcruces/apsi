import os
"""The two dimension scripts must share a byte-identical engine, and neither
may mention the other's dimension. This is what makes 'same method, separate
dimensions' true rather than aspirational."""
import re, hashlib, importlib.util
MARK = "# ENGINE  --  byte-identical"
def split(path):
    s = open(path).read(); i = s.index(MARK)
    return s[:i], s[i:]
e_dim, e_eng = split("classify_economic.py")
p_dim, p_eng = split("classify_populism.py")

assert e_eng == p_eng, (
    "ENGINE BLOCKS HAVE DIVERGED -- a fix was applied to one script only.\n"
    f"  economic engine sha1 {hashlib.sha1(e_eng.encode()).hexdigest()[:12]}\n"
    f"  populism engine sha1 {hashlib.sha1(p_eng.encode()).hexdigest()[:12]}")

# the engine must be dimension-neutral
for bad in ["POPULISM_", "BANDS_POPULISM", "Strong Left", "Strong Pluralist",
            "not_economic", "not_political_authority", "CHES_LRECON_DE"]:
    assert bad not in e_eng, f"engine still names a dimension: {bad}"

# neither dimension block may reference the other's constants
assert "POPULISM" not in e_dim and "Pluralist" not in e_dim
assert "Strong Left" not in p_dim and "CHES_LRECON_DE" not in p_dim

def load(path, name):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); m.__name__ = name
    s.loader.exec_module(m); return m
E = load("classify_economic.py", "econ")
P = load("classify_populism.py", "pop")

# canonical names exist in both, with the right content
for m, n_batt, n_top, n_neg in ((E, 22, 18, 4), (P, 18, 29, 4)):
    assert len(m.BATTERY) == n_batt and len(m.TOPIC_HYPOTHESES) == n_top
    assert len(m.NEGATIVE_TOPIC_HYPOTHESES) == n_neg and len(m.BANDS) == 7
    assert m.AUTORUN is False

# the hashes that tie these files to the cached runs must not move
assert E.battery_hash() == "90a23fbedce3", E.battery_hash()
# the populism battery gained five people-centrism pairs and five gate probes,
# so its hash necessarily moved; the economic one must NOT have.
assert P.battery_hash() == "78de7ae5a864", P.battery_hash()
assert E.BANDS[0][1] == "Strong Left" and P.BANDS[0][1] == "Strong Pluralist"
assert E.BATTERY_PARAPHRASE is not None and P.BATTERY_PARAPHRASE is not None
# both dimensions now carry a paraphrase battery: same domains, order and weights
for m in (E, P):
    if m.BATTERY_PARAPHRASE is None:      # a dimension may ship without a second wording
        continue
    assert [d for d, *_ in m.BATTERY] == [d for d, *_ in m.BATTERY_PARAPHRASE]
    assert [w for _, w, *_ in m.BATTERY] == [w for _, w, *_ in m.BATTERY_PARAPHRASE]
    for (_, _, l1, r1), (_, _, l2, r2) in zip(m.BATTERY, m.BATTERY_PARAPHRASE):
        assert l1 != l2 and r1 != r2, "a paraphrase must actually reword"
assert not hasattr(P, "FRAME_HYPOTHESES")
for gone in ("AE_DOMAINS","PC_DOMAINS","SHARED_DOMAINS","combine_populism","score_populism"):
    assert not hasattr(P, gone), f"{gone} survived the return to one dimension"
assert E.CONFIG["average_paraphrase"] is True
assert P.CONFIG["average_paraphrase"] is True
assert P.OFF_TOPIC_LABEL == "not_political_authority" and E.OFF_TOPIC_LABEL == "not_economic"
texts=[h for _,_,l,r in P.BATTERY for h in (l,r)]
assert len(texts)==36 and len(set(texts))==35, "expected exactly one reused statement"
from collections import Counter
reused=[t for t,c in Counter(texts).items() if c>1]
assert len(reused)==1 and "even when unpopular" in reused[0], reused
assert all(t.startswith("This text") for t in texts), "supplied wording was altered"
SUPPLIED = [
 "This text supports the idea that political authority is legitimate only when it follows constitutional rules and legal procedures.",
 "This text portrays the direct will of the people as the primary source of political legitimacy.",
 "This text suggests that elected leaders lose legitimacy when they ignore popular demands.",
 "This text emphasizes the role of courts and independent agencies in validating political authority.",
 "This text portrays charismatic leadership as a sufficient basis for political legitimacy.",
 "This text portrays political institutions as protecting citizens from arbitrary and abusive power.",
 "This text depicts political institutions as outdated structures that primarily serve elite interests.",
 "This text supports reforming institutions without weakening or bypassing them.",
 "This text portrays unelected institutions as unjust obstacles to democratic decision-making.",
 "This text emphasizes that strong institutions are more important than strong leaders.",
 "This text supports political decision-making through negotiation and compromise.",
 "This text emphasizes decisive leadership over deliberation.",
 "This text promotes expert knowledge as a legitimate basis for policymaking, even when unpopular.",
 "This text portrays popular opinion as superior to expert or technocratic judgment.",
 "This text frames slow decision-making as a necessary feature of democracy.",
 "This text portrays political opponents as legitimate actors with valid interests.",
 "This text depicts elites as manipulating institutions for personal gain.",
 "This text frames criticism and dissent as strengthening democracy.",
 "This text portrays those who oppose the people’s will as enemies of democracy.",
 "This text frames political conflict as a struggle between ordinary citizens and corrupt elites.",
 "This text portrays society as composed of diverse groups with competing interests.",
 "This text portrays the people as a unified moral community with shared goals.",
 "This text supports limiting majority power in order to protect minority rights.",
 "This text portrays the will of the majority as something that should not be constrained by special interests.",
 "This text suggests that leaders can clearly identify and represent the true will of the people.",
]
REVISED = [   # reworded in round 2 and still in the battery
 "This text portrays the people as a unified moral community with shared goals.",
 "This text emphasizes decisive leadership over deliberation.",
]
DROPPED = [   # round 3: removed by domain_sweep, originals kept in-file
 "This text supports the idea that political authority is legitimate only when it follows constitutional rules and legal procedures.",
 "This text portrays the direct will of the people as the primary source of political legitimacy.",
 "This text supports limiting majority power in order to protect minority rights.",
 "This text portrays the will of the majority as something that should not be constrained by special interests.",
]
missing = [t for t in SUPPLIED if t not in texts and t not in REVISED and t not in DROPPED]
assert not missing, f"supplied statements lost or altered: {missing[:2]}"
assert len(SUPPLIED) == 25
kept = sum(t in texts for t in SUPPLIED)
rev  = sum(t in REVISED for t in SUPPLIED)
drop = sum(t in DROPPED for t in SUPPLIED)
assert kept == 19, f"19 supplied statements should remain verbatim, found {kept}"
assert kept + rev + drop == 25, f"every supplied statement must be accounted for: {kept}+{rev}+{drop}"
# the originals must stay recoverable, so a revision can always be undone
src = open(os.environ.get("SCORER","classify_populism.py")).read()
# the originals are recorded in a comment block and wrap across lines, so
# normalise comment markers and whitespace before looking for them
flat = " ".join(src.replace("#", " ").split())
for t in REVISED + DROPPED:
    assert " ".join(t.split()) in flat, \
        f"original wording not recorded for reversion: {t[:60]}"
sides=[(l,r) for _,_,l,r in P.BATTERY]
assert len(sides)==18
print("test_engine_parity: PASS  (engine sha1 "
      f"{hashlib.sha1(e_eng.encode()).hexdigest()[:12]}, {len(e_eng.splitlines())} lines)")

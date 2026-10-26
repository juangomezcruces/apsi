"""Is the centre biased by inactive cells? Mirrors the observed Kaggle pattern."""
import numpy as np
rng = np.random.default_rng(3)
N = 3000
# activation rates and template offsets taken from the real per-pair table
rates  = [0.655, 0.360, 0.345, 0.330, 0.225, 0.125, 0.095, 0.035, 0.015]
names  = ["privatization","planning","safety_net","state_role","min_wage",
          "env_reg","fin_reg","healthcare","education"]
TRUE_OFFSET = 1.5            # the genuine template bias we want to remove
print(f"{'pair':<15}{'act%':>7}{'centre(all)':>13}{'centre(active)':>16}"
      f"{'bias(all)':>11}{'bias(act)':>11}")
for name, a in zip(names, rates):
    act = rng.random(N) < a
    c = np.empty(N)
    # active: real signal around the template offset
    c[act] = TRUE_OFFSET + rng.normal(0, 3, act.sum())
    # inactive: difference of two large negative margins -> wide, off-centre noise
    c[~act] = rng.normal(-6, 6, (~act).sum())
    all_med, act_med = np.median(c), np.median(c[act])
    print(f"{name:<15}{100*a:6.1f}%{all_med:13.2f}{act_med:16.2f}"
          f"{all_med-TRUE_OFFSET:11.2f}{act_med-TRUE_OFFSET:11.2f}")
print("\nbias(all) tracks activation rate; bias(active) does not.")
print("That is exactly the pattern in the Kaggle per-pair table.")

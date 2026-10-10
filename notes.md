The repo with a simulator package and nine notebooks, then execute every notebook so each comparison against its library actually passes.

Note the gotcha: `d.T` is pandas' transpose. I'll name the binary treatment column `D` everywhere to avoid that trap.

Notebooks 01–04 pass all library checks. Now running 05–09.

Rebuilding overwrote notebooks 04 and 06–08 with unexecuted copies, so I'm re-running them.

The repo is in `causal-from-scratch.zip`, set up as a git repo with one commit.  
Every one of the nine notebooks runs, and each library comparison passes.  
Most match the library exactly; the synthetic control matches pysyncon to solver tolerance. `pytest` passes, and `scripts/run_notebooks.py` re-runs all nine checks in about 2.5 minutes. Most of that is notebook 05, double ML.

**What's in it:**
- **`causalsim` package.** It holds the data generators. Each returns the data, the true values, and the hidden variables, and has settings that break one assumption at a time. It includes `retail_pricing()`, which produces data shaped like your store CSV with a known 3×3 elasticity matrix and endogenous prices.
- **Notebooks 01–09**, in the order from my earlier answer: regression adjustment, IPW and matching, AIPW, 2SLS, double ML, DiD, interrupted time series and synthetic control, VAR and Granger, and PC. Each one simulates data, implements the method from scratch, compares it against a library with assertions, then deliberately breaks an assumption and ends with exercises.

**Library behavior worth knowing.** To get matching results I had to reproduce some library defaults, which are documented in the README:
- **DoWhy** clips propensity scores to [0.05, 0.95] by default, and its `ips_weight` option is the normalized (Hájek) estimator.
- **linearmodels**' clustered standard errors only match the Stata convention with `group_debias=True`.
- **statsmodels**' Granger test gives the same F statistic as my version but a different p-value, because it uses different degrees of freedom.
- **pandas 3** silently returns a wrong mean for an object column that mixes `NaN` and booleans. This caused a bogus 4% coverage figure in notebook 05 until I stored the values as floats.

To start:

```
pip install -r requirements.txt
pip install -e .
jupyter lab notebooks/
```

The DiD, interrupted time series, and IV exercises use `retail_pricing()`, 
which makes them a good practice run before you try the methods on your real CSV.


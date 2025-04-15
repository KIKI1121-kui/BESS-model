# BESS-model
1. Core Modules Implemented:

BESS Model: Includes SOC constraints, charge/discharge efficiency, power limits, and degradation cost modeling.

Data Processing: Supports real and synthetic data loading for weather, electricity price, load, and wind power.

Forecasting: Utilizes Random Forest algorithms for PV and wind generation prediction with uncertainty bounds.

AC Grid Modeling: 4-bus system with full AC power flow calculation and PTDF/VSF-based linearization.

Optimization Solver: Two-stage iterative solver with linear approximation and AC feasibility check.

Economic Evaluation: Calculates LCOS, NPV, IRR, and payback period using vectorized and cached operations.

Visualization Tools: SOC profiles, dispatch patterns, power flow visualization, and economic comparison plots.

Scenario Engine: Supports parallel multi-scenario analysis with automated result comparison.

2. Key Results So Far:

（The judgment of the economy is mainly based on real comparison rather than on achieving economic feasibility.）

Annual revenue across scenarios ranges from $36,000 to $38,000.

LCOS varies between $6.20 and $7.70/kWh (baseline: $6.72/kWh).

NPV remains negative under current assumptions, from -$120,000 to -$75,000.

Maximum IRR observed is 0.5%, indicating economic challenges under a pure arbitrage model.

Grid impact analysis shows optimized BESS can reduce average line losses to ~0.67 kW.

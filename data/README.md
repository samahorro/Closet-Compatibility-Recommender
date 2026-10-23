# Dataset contract (to be finalized with James)

Source: Polyvore Outfits (disjoint split).

Proposed outputs:
- item metadata keyed by item ID (category, title, original split)
- normalized item feature matrix keyed by item ID
- train/validation/test outfit membership lists

Never construct training pairs from validation/test outfits. Do not commit raw data or feature dumps to GitHub.

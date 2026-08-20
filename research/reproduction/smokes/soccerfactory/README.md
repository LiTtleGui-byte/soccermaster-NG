# SoccerFactory path smoke

This fixed SNGS-10004 diagnostic validates the post-migration paths through
Step 1, enrichment, coord-only Refiner, Refiner-preserving Step 3, conversion,
and one real SoccerMaster DataLoader batch. It is inference-only and does not
claim that role, team, jersey-number, track identity, or pitch coordinates are
accurate.

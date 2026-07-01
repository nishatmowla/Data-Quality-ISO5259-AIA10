# Contributing

## Adding a New Quality Characteristic

1. Add its enum value to `QualityCharacteristic` in `models/iso5259.py`
2. Add its sub-dimensions to `CHARACTERISTIC_DIMENSIONS`
3. Create `agents/evaluators/<characteristic>.py` following the pattern of existing evaluators
4. Wire it into `main.py` `run_evaluation()` and both UI files

## Adding a New Syntactic Rule Type

Add the case to `_apply_syntactic_rule()` in `tools/rule_checker.py` and update the rule type `enum` in `agents/rule_generator.py` and both UI editors.

## Changing the LLM Provider

Each agent function accepts a `client` argument and calls `client.chat.complete(...)`. Swap the client for any provider that exposes the same interface, or adapt the call inside each evaluator.

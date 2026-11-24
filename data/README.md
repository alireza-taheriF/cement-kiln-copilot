# Data

No plant data is stored in this repository: it belongs to the customer.

## Demo data (simulated)

`python -m src.sim.generate --out data/demo` writes 14 days of a simulated
3,000 t/d kiln line (5-minute grid) to `data/demo/` (git-ignored):

| file | content |
| ---- | ------- |
| `signals.csv` | long format `timestamp, tag_name, value`, canonical tags (`src/domain/tags.py`) |
| `lab.csv` | free-lime lab results every 2 h (`timestamp, parameter, value, sample_id`) |
| `fuel_events.json` | ground truth of the six gas curtailments and their ramp durations |
| `power_quota.csv` | the mill power quota per step |

`scripts/train_demo_model.sh` generates it, seeds the database and trains every
model. The simulator (`src/sim/plant.py`) has a known ground truth, which is
how the tests check that recommendations are safe when actually applied.

> **Simulated, not measured.** Coefficients are only in the right range for a
> dry-process line. Never use these numbers for engineering decisions.

## Plant data

Plant exports are read through `src/connectors` (CSV historian exports, Excel
lab sheets with Jalali dates, OPC UA history), mapped to canonical tags with
`config/connectors.yaml`, and stored in the database. Put local exports in
`data/plant_exports/`, which is git-ignored: never commit plant data.

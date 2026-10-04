# CDH Inventory DSS — Data Handling Rules

## Raw Data Principle

Files stored under `data/raw/` are original source files provided or exported by Calamba Doctors Hospital.

Raw files must never be manually modified, overwritten, cleaned, aggregated, or reformatted.

All cleaning, validation, transformation, feature engineering, and analytical processing must be performed programmatically.

## Data Zones

### Raw
`data/raw/`

Original client/source files.

### Validated
`data/validated/`

Records or datasets that have passed documented validation procedures.

### Processed
`data/processed/`

Analysis-ready tables derived programmatically from validated records.

### Quarantine
`data/quarantine/`

Files or records containing unresolved critical data-quality problems.

### Reference
`data/reference/`

Approved supporting lookup or reference data that are not primary historical observations.

## Data Granularity Rule

Completed annual inventory reports are treated as annual observations.

The project must not generate artificial monthly demand observations from annual totals for forecasting purposes.

Partial or year-to-date reports must be clearly separated from completed annual observations.

## Data Privacy Rule

Patient-identifying information must not be retained in the analytical database or decision-support system.

Confidential CDH raw data must not be committed to Git or uploaded to public repositories.

## Reproducibility Rule

Every derived analytical output must be traceable to:

- source file;
- source period;
- processing timestamp;
- transformation version;
- model version, when applicable; and
- scenario parameters, when applicable.
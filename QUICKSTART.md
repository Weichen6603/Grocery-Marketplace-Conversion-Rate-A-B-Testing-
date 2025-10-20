# Quick Start Guide

## One-Line Execution

```bash
python ab_test_analysis.py --csv application.csv --power
```

## 5-Minute Setup

### 1. Install dependencies
```bash
pip install numpy pandas matplotlib scipy statsmodels lifelines
```

### 2. Run analysis
```bash
cd /path/to/AB_test
python ab_test_analysis.py --csv application.csv --power
```

### 3. View results
- Charts: `outputs/conversion_ci.png`, `outputs/funnel.png`, `outputs/km_curves.png`
- Data: `outputs/group_funnel_summary.csv`, `outputs/applicant_level_wide.csv`

## Key Results

| Metric | Control | Treatment | Uplift |
|--------|---------|-----------|--------|
| Conversion Rate | 19.81% | 34.33% | +73.29% |
| Sample Size | 14,501 | 7,197 | - |
| Time to First Batch | 10 days | 7 days | -30% |
| Statistical Significance | - | **p ≈ 0** | **>99% confidence** |
| Statistical Power | - | **100%** | **>> 80% target** |

## Advanced Usage

### With PostgreSQL (Production)
```bash
pip install psycopg2-binary
python ab_test_analysis.py \
    --db-host analytics-db.example.com \
    --db-user analyst \
    --db-password your_password \
    --db-database analytics
```

### Custom Parameters
```bash
# Different LTV and cost per check
python ab_test_analysis.py \
    --csv application.csv \
    --cost_per_check 40 \
    --ltv 200 \
    --power
```

## Files Overview

| File | Purpose | Size |
|------|---------|------|
| `ab_test_analysis.py` | Main analysis script (CSV + PostgreSQL) | 24 KB |
| `db_pipeline.py` | PostgreSQL ETL module | 12 KB |
| `application.csv` | Sample event-level data | 7.7 MB |
| `README.md` | Full documentation | 16 KB |
| `QUICKSTART.md` | This file | - |

## Troubleshooting

### "Module not found" errors
```bash
# Install core dependencies
pip install numpy pandas matplotlib scipy statsmodels

# Optional: Survival analysis
pip install lifelines

# Optional: PostgreSQL
pip install psycopg2-binary
```

### "psycopg2 is not installed" error
**Option 1**: Install PostgreSQL support
```bash
pip install psycopg2-binary
```

**Option 2**: Use CSV instead
```bash
python ab_test_analysis.py --csv application.csv
```

## What the Scripts Do

### `ab_test_analysis.py`
- Reads event-level data (CSV or PostgreSQL)
- Pivots events into applicant-level records with funnel flags
- Performs two-proportion z-test for conversion rates
- Calculates Wilson confidence intervals
- Computes cost-effectiveness metrics (CPIC, ROI)
- Performs Kaplan-Meier survival analysis
- Calculates post-hoc statistical power
- Generates publication-quality visualizations
- Exports results as CSV and PNG files

### `db_pipeline.py`
- Connects to PostgreSQL databases
- Manages connection pooling for scalability
- Extracts event data with date/experiment filtering
- Validates data quality (5-point scoring system)
- Introspects database schemas
- Handles errors gracefully with helpful messages

## Resume Impact

This project demonstrates:
- ✅ **Statistical Analysis**: Two-proportion testing, hypothesis testing, power analysis
- ✅ **A/B Testing Framework**: Complete testing methodology, funnel analysis
- ✅ **Data Pipeline Engineering**: PostgreSQL integration, ETL, connection pooling
- ✅ **Python Development**: Production-grade code, error handling, CLI design
- ✅ **Data Visualization**: Publication-quality charts (300 dpi)
- ✅ **Business Impact**: $127.79 ROI per conversion, 73% conversion lift

## Performance Metrics

- **Total Records**: 21,698 applicants
- **Event-Level Data**: 21,698 events (1:1 ratio)
- **Statistical Power**: 100% (vs 80% required)
- **Confidence Level**: >99% (vs 99% required)
- **Execution Time**: ~5 seconds for full analysis
- **Output Size**: 3.3 MB (mostly CSV data)

---

For detailed documentation, see `README.md`

# Shopper Hiring A/B Test Analysis

## Project Overview

This repository contains a **production-ready A/B testing framework** with automated data pipelines (PostgreSQL + Python) to measure the impact of **early background check initiation** on shopper conversion rates in a hiring funnel.

### Key Capabilities
-  **Event-level data processing** with Pandas/NumPy
-  **PostgreSQL integration** with connection pooling
-  **Automated ETL pipeline** with data quality validation
-  **Statistical analysis** at 99%+ confidence & 100% power
-  **Cost-effectiveness modeling** with ROI analysis
-  **Publication-quality visualizations**

### Business Context

**Problem**: How to improve shopper applicant conversion through faster prerequisite completion?

**Hypothesis**: Initiating background checks earlier in the process will help shoppers maintain momentum and be more likely to complete their first batch (success metric).

**Test Design**: 
- **Control Group** (n=14,501): Standard process
- **Treatment Group** (n=7,197): Background check initiated earlier
- **Cost per Check**: $30
- **Success Metric**: First batch completion (`first_batch_completed_date`)

---

## Key Results

### Statistical Significance 
- **Conversion Rate Uplift**: 19.81% → 34.33% (**+73.29% relative lift**)
- **Absolute Difference**: +14.52 percentage points
- **Test Statistic**: z = 23.375, **p ≈ 0** (>> 0.01)
- **Confidence**: > **99%** (α = 0.01, two-tailed)
- **Statistical Power**: **100%** (>> 80% target)

### Cost-Effectiveness 
- **Incremental Cost**: ~1,819 additional background checks × \$30 = $54,569
- **Incremental Conversions**: **1,045** additional shoppers
- **CPIC** (Cost Per Incremental Conversion): **$52.21**
- **ROI**: $180 LTV - $52.21 CPIC = **$127.79 per incremental conversion → Positive ROI**

### Speed Improvement 
- **Median Time to First Batch**: 10 days → 7 days (**-30% faster**)
- **Kaplan-Meier Test**: Log-rank χ² = 3447.18, p ≈ 0 (highly significant)

---

## File Structure

```
AB_test/
├── ab_test_analysis.py          # Main analysis script (supports CSV & PostgreSQL)
├── db_pipeline.py               # PostgreSQL ETL pipeline module
├── application.csv              # Event-level sample data (21,698 records)
├── README.md                    # This file
└── outputs/                     # Generated outputs
    ├── conversion_ci.png        # Conversion rate comparison (95% CI)
    ├── funnel.png               # 5-stage funnel comparison
    ├── km_curves.png            # Kaplan-Meier survival curves
    ├── group_funnel_summary.csv # Funnel metrics by group
    └── applicant_level_wide.csv # Full applicant-level data (pivoted)
```

---

## Usage

### From CSV File (Quick Start)
```bash
python ab_test_analysis.py --csv application.csv --cost_per_check 30 --ltv 180
```

### From PostgreSQL Database (Production)
```bash
python ab_test_analysis.py \
    --db-host analytics-db.example.com \
    --db-port 5432 \
    --db-user analyst \
    --db-password your_password \
    --db-database analytics \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --cost_per_check 30 \
    --ltv 180 \
    --power
```

### With Statistical Power Analysis
```bash
python ab_test_analysis.py --csv application.csv --cost_per_check 30 --ltv 180 --power
```

### Arguments

#### Data Source
- `--csv` (optional): Path to event-level CSV file
- `--db-host` (optional): PostgreSQL hostname
- `--db-port` (default: 5432): PostgreSQL port
- `--db-user`: PostgreSQL username
- `--db-password`: PostgreSQL password
- `--db-database`: PostgreSQL database name
- `--start-date`: Start date for DB query (YYYY-MM-DD)
- `--end-date`: End date for DB query (YYYY-MM-DD)
- `--experiment-id` (optional): Filter by experiment ID

#### Analysis Parameters
- `--cost_per_check` (default: 30.0): Unit cost for background check
- `--ltv` (optional): Assumed lifetime value per converted shopper
- `--outdir` (default: outputs): Output directory for plots and tables
- `--power`: Enable post-hoc statistical power calculation

---

## Data Requirements

### Input: Event-Level CSV or PostgreSQL

#### CSV Format
Expected columns:
- `applicant_id`: Unique applicant identifier
- `channel`: Acquisition channel (e.g., "web-search-engine", "social-media")
- `group`: "control" or "treatment"
- `city`: Geographic location
- `event`: Event type (e.g., "application_date", "background_check_initiated_date")
- `event_date`: When the event occurred (YYYY-MM-DD format)

#### PostgreSQL Schema
```sql
CREATE TABLE events (
    event_id BIGSERIAL PRIMARY KEY,
    applicant_id INTEGER NOT NULL,
    channel VARCHAR(50),
    treatment_flag SMALLINT,  -- 0=control, 1=treatment
    city VARCHAR(100),
    event_type VARCHAR(50),
    event_timestamp TIMESTAMP,
    experiment_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_applicant (applicant_id),
    INDEX idx_timestamp (event_timestamp)
);
```

### Data Processing Pipeline

The script automatically:
1. **Extracts** events from CSV or PostgreSQL
2. **Validates** data quality (missing values, duplicates, data types)
3. **Transforms** event-level data to applicant-level wide table
4. **Derives** funnel flags (bg_initiated, bg_completed, card_activated, converted)
5. **Computes** time-to-first-batch (TTFB) in days
6. **Pivots** to wide format for group comparisons

---

---

## Data Pipelines & ETL Architecture

### PostgreSQL Pipeline (`db_pipeline.py`)

The `PostgreSQLPipeline` class provides production-grade data extraction with:

#### Features
- **Connection Pooling**: Reusable connection pool for high-concurrency scenarios
- **Automated Extraction**: Query builder with filtering and date range support
- **Data Validation**: Quality checks (null values, duplicates, data types)
- **Schema Introspection**: Automatic detection of table structure
- **Error Handling**: Retry logic and detailed error reporting
- **Batch Processing**: Support for large dataset extraction

#### Usage Example
```python
from db_pipeline import PostgreSQLPipeline

# Initialize pipeline
pipeline = PostgreSQLPipeline(
    host='analytics-db.company.com',
    database='events_db',
    user='analyst',
    password='secure_password'
)

# Extract event data
df = pipeline.fetch_events(
    start_date='2024-01-01',
    end_date='2024-12-31',
    experiment_id='bg_check_timing'
)

# Validate data quality
validation = pipeline.validate_data(df)
print(f"Data Quality Score: {validation['data_quality_score']:.1f}%")

# Get schema information
schema = pipeline.get_schema_info('events')
print(f"Columns: {len(schema)}")

# Close connections
pipeline.close()
```

#### Data Quality Validation
The pipeline performs 5 automated checks:
1. ✓ No null values in key columns
2. ✓ Valid group values (control/treatment)
3. ✓ Reasonable date ranges
4. ✓ Sufficient applicant volume
5. ✓ Minimal duplicate events

Quality score ranges from 0-100%.

#### Architecture Benefits
- **Scalability**: Connection pooling handles 1000s of concurrent queries
- **Reliability**: Automatic error handling and retry logic
- **Transparency**: Detailed logging of all operations
- **Maintainability**: Separation of concerns (DB layer vs analysis)
- **Testability**: Easy to mock for unit testing

---

### Primary Analysis: Two-Proportion Z-Test

**Test**: Difference in first-batch conversion rates between treatment and control

$$H_0: p_{treat} = p_{ctrl}$$
$$H_1: p_{treat} \neq p_{ctrl}$$

**Test Statistic**:
$$z = \frac{p_{treat} - p_{ctrl}}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_{treat}} + \frac{1}{n_{ctrl}}\right)}}$$

where $\hat{p} = \frac{x_{treat} + x_{ctrl}}{n_{treat} + n_{ctrl}}$

**Significance Level**: α = 0.01 (99% confidence, two-tailed)

### Confidence Intervals

95% and 99% CIs computed using **Wilson score method** (recommended for proportions):
$$CI = \frac{1}{1 + z^2/n}\left[\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}\right]$$

### Survival Analysis

**Kaplan-Meier Estimator** for time-to-first-batch:
- Event: First batch completed
- Censoring: Applicants not converted by end of observation window
- Test: Log-rank test for equality of survival curves

### Cost-Effectiveness

**CPIC** (Cost Per Incremental Conversion):
$$\text{CPIC} = \frac{\text{Incremental Cost}}{\text{Incremental Conversions}} = \frac{\Delta BG_{rate} \times n_{treat} \times \$30}{\Delta Conv_{rate} \times n_{treat}}$$

**ROI Decision**:
- Net = LTV - CPIC
- Positive if Net > 0

### Statistical Power

**Post-hoc power calculation** using non-centrality parameter:
$$\text{Power} = 1 - \Phi(z_\alpha - \sqrt{n_{eff}} \cdot h / \sqrt{2 \cdot p(1-p)})$$

where:
- $z_\alpha = 2.576$ (critical value for α = 0.01, two-tailed)
- $h = 2[\arcsin(\sqrt{p_{treat}}) - \arcsin(\sqrt{p_{ctrl}})]$ (Cohen's h)
- $n_{eff} = \frac{n_{ctrl} \times n_{treat}}{n_{ctrl} + n_{treat}}$ (effective sample size)

---

## Output Descriptions

### Visualizations

#### 1. `conversion_ci.png`
Bar chart comparing conversion rates with 95% confidence intervals:
- Control: 19.81% [19.17%, 20.47%]
- Treatment: 34.33% [33.25%, 35.44%]
- Professional color scheme (blue vs. green)
- 300 dpi, high-quality output

#### 2. `funnel.png`
Side-by-side funnel showing all 5 conversion stages:
1. Background Check Initiated
2. Background Check Completed
3. Card Activated
4. Orientation Completed (optional)
5. First Batch Completed (success)

Note: Orientation is optional; not required for conversion.

#### 3. `km_curves.png`
Kaplan-Meier survival curves showing time-to-first-batch:
- Treatment group reaches conversion faster (7-day median)
- Control group takes longer (10-day median)
- Curves are clean (no distracting confidence bands)

### Data Tables

#### `group_funnel_summary.csv`
Core metrics by group:
```
group,n,bg_initiated_rate,bg_completed_rate,card_activated_rate,orientation_rate,conversion_rate
control,14501,0.7473,0.6521,0.7473,0.4125,0.1981
treatment,7197,1.0000,0.8870,0.8183,0.4681,0.3433
```

#### `applicant_level_wide.csv`
Full applicant-level data (wide format):
- One row per applicant
- Columns: applicant_id, channel, group, city, [all events as timestamps], [derived metrics]
- Useful for further analysis, subgroup cuts, etc.

---

## Key Assumptions & Limitations

### Assumptions
1. **Random Assignment**: Control/Treatment groups are randomly assigned (no confounding)
2. **Independence**: Applicant conversions are independent
3. **Stable Effect**: Treatment effect is homogeneous across channels and time
4. **Complete Data**: No missing outcome data (MAR/MCAR assumptions)
5. **LTV**: Assumed $180 per converted shopper (can be adjusted with `--ltv`)

### Limitations
1. **Observational Heterogeneity**: Results may vary by channel/geography (use subgroup analysis)
2. **Time Window**: Observation window is fixed; later conversions not captured
3. **Causality**: Correlation ≠ causation (though randomization helps)
4. **External Validity**: Results specific to this applicant pool and time period

### Recommendations for Improvement
1. **Subgroup Analysis**: Break down by channel, city, time cohort
2. **Interaction Tests**: Check if treatment effect varies by channel
3. **Power Analysis Pre-Registration**: Plan sample size before next test
4. **Heterogeneous Effects**: Use ML/causal forest methods for personalized estimates

---

## Installation & Dependencies

### Core Requirements
```bash
pip install numpy pandas matplotlib scipy statsmodels
```

### Optional: PostgreSQL Support
```bash
pip install psycopg2-binary
```

**Note**: PostgreSQL support is optional. The module gracefully handles missing `psycopg2`:
- ✓ CSV workflow works without it
- ✓ Importing `db_pipeline` works without it
- ✓ Creating `PostgreSQLPipeline` instance raises helpful error if not installed
- ✓ No runtime errors or crashes

### Optional: Survival Analysis
```bash
pip install lifelines
```

### Full Installation
```bash
pip install numpy pandas matplotlib scipy statsmodels psycopg2-binary lifelines
```

### Python Version
- Python 3.8+

---

## Examples

### Example 1: Run Full Analysis with Power
```bash
python ab_test_analysis.py --csv application.csv --power
```

Output:
```
[Info] Reading application.csv ...
[Info] Applicants: 21,698

=== Core Funnel by Group ===
    group     n  bg_initiated_rate  conversion_rate
  control 14501             0.7473           0.1981
treatment  7197             1.0000           0.3433

=== Primary KPI: First-Batch Conversion ===
Control:   2873/14501 = 19.8124%  (95% CI [19.1717%, 20.4691%])
Treatment: 2471/7197 = 34.3338%  (95% CI [33.2454%, 35.4388%])
Δ (pp):    14.52 pp   | Lift: 73.29%
z = 23.375, p = 7.736e-121  --> Significant

✓ ACHIEVED > 80% statistical power
```

### Example 2: Custom LTV and Cost
```bash
python ab_test_analysis.py --csv application.csv --cost_per_check 40 --ltv 200
```

---

## Interpretation Guide

### For Decision-Makers
- **Bottom Line**: The treatment is highly effective and cost-positive. Recommend full rollout.
- **Confidence**: Results are based on large sample (21,698) with >99% confidence and 100% power.
- **ROI**: Expected $127.79 profit per incremental conversion (at $180 LTV).

### For Data Analysts
- **Method**: Two-proportion z-test, Wilson CI, Kaplan-Meier analysis
- **Effect Size**: Cohen's h = 0.3295 (small-to-medium practical significance)
- **Robustness**: Power analysis confirms we can reliably detect this effect

### For Product Managers
- **Speed Gain**: Shoppers convert 30% faster with treatment (7d vs 10d median)
- **Conversion Funnel**: Treatment improves BG completion (+23.5pp) and card activation (+7.1pp)
- **Next Steps**: Monitor for seasonality, test subgroups, measure long-term retention

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'lifelines'"
**Solution**: Install lifelines to enable survival analysis
```bash
pip install lifelines
```
Or run without `--power` flag; analysis will skip survival curves.

### Issue: "Missing columns: {'group'}"
**Solution**: Ensure CSV has required columns (applicant_id, channel, group, city, event, event_date)

### Issue: "No treatment group found"
**Solution**: Check that `group` column contains values like "treatment" (case-sensitive)

### Issue: "psycopg2 is not installed" when using PostgreSQL
**Solution 1** (Install psycopg2): `pip install psycopg2-binary`
**Solution 2** (Use CSV instead): `python ab_test_analysis.py --csv application.csv`
**Note**: This error only occurs when trying to create a PostgreSQL pipeline. CSV workflow always works.

---


## Appendix: Technical Details

### A. Two-Proportion Z-Test Formula

Given:
- $n_1, x_1$ = sample size and successes in group 1
- $n_2, x_2$ = sample size and successes in group 2
- $p_1 = x_1/n_1$, $p_2 = x_2/n_2$
- $\hat{p} = (x_1 + x_2)/(n_1 + n_2)$

Test statistic:
$$z = \frac{p_2 - p_1}{\sqrt{\hat{p}(1-\hat{p})(1/n_1 + 1/n_2)}}$$

P-value (two-tailed): $P(|Z| > |z|)$ where $Z \sim N(0,1)$

### B. Wilson Confidence Interval

For confidence level $1-\alpha$:
$$\text{CI}_\text{Wilson} = \frac{1}{1+z_{\alpha/2}^2/n}\left[\hat{p} + \frac{z_{\alpha/2}^2}{2n} \pm z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z_{\alpha/2}^2}{4n^2}}\right]$$

Advantage: Better coverage for extreme proportions (near 0 or 1)

### C. Cohen's h Effect Size

$$h = 2\left[\arcsin(\sqrt{p_2}) - \arcsin(\sqrt{p_1})\right]$$

Interpretation:
- $|h| < 0.2$: Small
- $0.2 \leq |h| < 0.5$: Small-to-medium
- $0.5 \leq |h| < 0.8$: Medium
- $|h| \geq 0.8$: Large

---

**Last Updated**: Mar 24, 2025

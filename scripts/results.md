# Comprehensive Format Comparison

## Field-Level Metrics Comparison

| Format | Precision | Recall | F1 | BERTScore |
|--------|-----------|--------|-----|-----------|
| json | 0.050 | 0.026 | 0.033 | 0.384 |
| toon | 0.041 | 0.024 | 0.027 | 0.486 |
| yaml | 0.050 | 0.027 | 0.034 | 0.490 |

## Entity-Level Metrics by Format

### Admission

| Format | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| json | 0.100 | 0.100 | 0.100 |
| toon | 0.111 | 0.111 | 0.111 |
| yaml | 0.100 | 0.100 | 0.100 |

### Diagnosis Tests

| Format | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| json | 0.045 | 0.333 | 0.062 |
| toon | 0.000 | 0.222 | 0.000 |
| yaml | 0.000 | 0.200 | 0.000 |

### Medical Examinations

| Format | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| json | 0.200 | 0.125 | 0.140 |
| toon | 0.111 | 0.111 | 0.111 |
| yaml | 0.100 | 0.125 | 0.040 |

### Surgeries

| Format | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| json | 0.100 | 0.200 | 0.100 |
| toon | 0.111 | 0.222 | 0.111 |
| yaml | 0.000 | 0.200 | 0.000 |

### Symptoms

| Format | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| json | 0.000 | 0.200 | 0.000 |
| toon | 0.000 | 0.222 | 0.000 |
| yaml | 0.000 | 0.200 | 0.000 |

### Treatments

| Format | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| json | 0.000 | 0.300 | 0.000 |
| toon | 0.000 | 0.333 | 0.000 |
| yaml | 0.000 | 0.300 | 0.000 |

## Schema Coverage Comparison

| Section | json | toon | yaml |
|---------|---------|---------|---------|
| admission | 20.0% | 33.3% | 20.0% |
| diagnosis tests | 100.0% | 77.8% | 100.0% |
| discharge | 60.0% | 44.4% | 60.0% |
| medical examinations | 90.0% | 66.7% | 100.0% |
| patient information | 100.0% | 77.8% | 100.0% |
| patient medical history | 90.0% | 77.8% | 100.0% |
| surgeries | 60.0% | 33.3% | 70.0% |
| symptoms | 100.0% | 77.8% | 100.0% |
| treatments | 100.0% | 77.8% | 100.0% |
| visit motivation | 100.0% | 100.0% | 100.0% |

## Format Reliability Metrics

| Format | Success % | Avg Retries | Avg Time(s) | Failed |
|--------|-----------|-------------|------------|--------|
| json | 100.0% | 0.7 | 43.6 | 0 |
| toon | 60.0% | 5.9 | 328.6 | 4 |
| yaml | 100.0% | 1.7 | 65.5 | 0 |

## Token Compression Comparison

| Format | Input Chars | Output Chars | Average Compression |
|--------|-------------|--------------|-------------------|
| json | 42,263 | 37,813 | 98.6% |
| toon | 39,772 | 24,516 | 72.2% |
| yaml | 42,263 | 27,240 | 74.2% |

---
layout: post
title: "Financial Knowledge Graph for Systemic risk"
date: 2026-01-03 09:00:00
tags: [research]
category: blog
hidden: false
---
# Financial Knowledge Graphs for Systemic Risk  
---

## Research Motivation 

Modern financial crises are not driven by isolated firm failures, but by **hidden interdependencies** across institutions, markets, and information channels. Accurately identifying and monitoring these connections is directly aligned with U.S. national interests in **financial stability, systemic-risk oversight, and macroprudential regulation**.

My research develops **financial knowledge graphs** that integrate market data, liquidity measures, and unstructured news using modern machine learning and natural language processing (NLP). This work potentially can contributes to regulatory risk monitoring frameworks relevant to the US and global regulatory agency.

---

## Technical Foundation: Graph Theory to Knowledge Graphs

Graph theory originates from Leonhard Euler’s 1736 solution to the *Seven Bridges of Königsberg* problem, demonstrating that **structure, not distance, determines feasibility**. This insight directly maps to systemic risk: contagion arises from network topology rather than individual balance sheets.

A **graph** consists of:

- **Nodes (vertices):** entities such as financial institutions
- **Edges:** relationships or dependencies
- **Degree:** number of connections per node
- **Density:** concentration of interconnections
- **Labels:** attributes such as sector, size, or risk classification

A **Knowledge Graph (KG)** extends this framework by embedding *semantic meaning* into edges and labels, enabling interpretable and extensible risk analysis.

![Seven Bridge Problem](assets/images/sevenbridge.png)



---

## Financial Knowledge Graph Construction

### Data Scope

- ~200 U.S. and global financial institutions
- Monthly observations from **2006–2013**
- Focus on **Top 25 firms by market capitalization**

### Graph Design

- **Nodes:** Systemically important financial institutions
- **Edges:** Partial correlations (conditional dependencies) of market-implied volatility
- **Edge weights:** Strength of dependency
- **Node labels:** KNN-based clustering using market capitalization

This methodology is published in *Journal of Financial Stability*:

> **From Liquidity Risk to Systemic Risk: A Use of Knowledge Graph**

---

## Systemic Risk Insights During the Global Financial Crisis

Visualizing the volatility-based KG during **2007–2009** reveals:

- Sharp increases in **network density** during 2008
- Central positioning of **CME Group** as a clearinghouse
- Strong conditional dependencies between clearing and dealer banks (e.g., JPMorgan)

These findings align with post-crisis regulatory emphasis on clearing, margining, and central counterparties (CCPs).

---

## Liquidity-Based Knowledge Graphs: Revealing Hidden Fragility

Volatility alone often over-connects firms during crises. To address this, I introduce **liquidity-discount-based KGs**:

- Liquidity discount measures price compression caused by funding stress
- Liquidity-based graphs reveal structural dependencies invisible to volatility
- Robust across crisis and non-crisis regimes

Empirical evidence shows liquidity metrics provide **earlier and more interpretable signals** of systemic fragility (related work in *Journal of Fixed Income*, 2012).

---

## Integrating NLP: News-Driven Financial Knowledge Graphs

Market data captures realized stress, but **news reflects expectations and sentiment**.

### Data and Methodology

- English-language financial news (LexisNexis)
- Text normalization, stop-word removal, and stemming
- Inclusion of distressed firms (Lehman Brothers, Bear Stearns, AIG)

### Two NLP-Based Graphs

1. **Frequency-based KG**  
   - Entity extraction via SpaCy  
   - Edge weight = co-mention frequency

2. **Embedding-based KG**  
   - OpenAI text embeddings (1536–3072 dimensions)  
   - Semantic similarity mapped to graph edges

---

## Empirical Findings from News-Based Networks

### Pre-Crisis (2007)

- Bankrupt firms (Lehman, Bear Stearns) show **few connections**
- Highly connected firms include Bank of America, Citi, Merrill Lynch
- Vulnerability is **not clearly identifiable ex ante**

### Crisis Period (2008)

- Network density increases sharply
- Media-driven co-mentions explode
- Graph complexity rises post-shock

**Key insight:** News-based networks react strongly *after* stress materializes.

---

## Embedding Geometry and Crisis Convergence

Using t-SNE on annualized news embeddings:

- **2007:** Distressed firms cluster separately
- **2008–2009:** Clusters collapse; firms converge semantically
- Major banks become indistinguishable from failed institutions

This reflects systemic narrative convergence during crises.

---

## Predictive Graphs from Machine Learning

I further construct graphs from supervised learning artifacts:

- RNNs trained on embedded news predict firm labels
- Confusion matrices serve as weighted adjacency matrices
- Misclassification probabilities measure institutional similarity

This approach bridges **deep learning interpretability** and network science.

---

## Can Knowledge Graphs Predict Systemic Crises?

Key questions evaluated:

- Is the network denser during crisis periods?
- Can Lehman or Bear Stearns be identified ex ante?
- Can AIG’s distress be forecast?

### Conclusion

While financial knowledge graphs **explain crisis propagation**, they do **not reliably predict crisis onset**. Systemic risk is a regime-shift phenomenon driven by macro shocks and policy responses.

---

## Broader Impact and Policy Relevance

Beyond crisis analysis, this framework supports:

- Market sentiment monitoring
- Bank network similarity analysis
- Return direction classification using embeddings

These tools align with U.S. priorities in **financial surveillance, stress testing, and AI-driven risk management**.

---

## Research Impact Summary (NIW Alignment)

This research:

- Advances **interpretable AI** for financial stability
- Integrates structured and unstructured data at scale
- Supports regulatory objectives under Basel III and FRTB
- Provides transferable tools for systemic-risk oversight

By improving transparency and analytical depth in monitoring interconnected financial systems, this work serves the **national interest of maintaining U.S. financial system resilience**.

---



# airbnb-sydney-classification
A machine learning project in R that predicts whether an Airbnb listing in Sydney is budget, mid range or luxury, and shows which features of a property matter most for its price.

This project was developed as part of an academic project for STAT5003 Computational Statistical Methods at the University of Sydney.

Full report: [View the rendered analysis](https://alfonsodelarosagomez.github.io/airbnb-sydney-classification/)

## 1. Overview

### Background and goal

Short term rentals such as Airbnb are a big part of Sydney's tourism market and a regular topic in the housing affordability debate. Travellers want to know whether a price is fair. Hosts want to know how to price a property. Councils and researchers want to understand how the market is structured.

This project asks one simple question: what makes an Airbnb listing in Sydney cheap, mid priced or expensive?

Instead of guessing an exact dollar amount, the model sorts each listing into one of three price bands.

| Band | Price per night (AUD) | Share of listings |
|---|---|---|
| Budget | Under $150 | 29% |
| Mid range | $150 to $349 | 48% |
| Luxury | $350 and above | 23% |

The band edges were chosen with reference to published average daily rates for Sydney hotels (CBRE, 2024) and high end short stay data (Airbtics, 2025).

### Technical summary

| Aspect | Description |
|---|---|
| Data | 18,187 Sydney listings and 79 variables, from Inside Airbnb (June 2025 snapshot). |
| Task | Multiclass classification of nightly price band. |
| Tools | R, tidyverse, tidymodels, missForest, themis (SMOTE), leaflet, R Markdown. |
| Methods | Data cleaning, imputation, PCA, class balancing, nested cross validation, five competing models. |
| Best model | Support Vector Machine with an RBF kernel (see the limitations in section 4 before relying on the scores). |

## 2. Stakeholder summary

This section assumes no background in data science.

### What does the Sydney market look like?

- Most listings sit in the middle: nearly half (48%) cost between $150 and $349 a night. A small number of very expensive properties stretch the top end, with some priced at several thousand dollars a night.
- Size drives price: the number of bathrooms, bedrooms and guests a property sleeps are the features most strongly linked to price. Median price rises step by step as the number of bathrooms increases.
- Type of stay matters: entire homes and apartments have the highest median price and the widest spread. Shared rooms are the cheapest.
- Location matters: listings cluster around the CBD, the Eastern Suburbs and the Lower North Shore. The most expensive listings gather in coastal and harbour side suburbs.

### How well can a model sort listings into price bands?

Five models were compared using a method designed to give a fair estimate of performance on listings the model has never seen.

| Model | Accuracy | ROC AUC |
|---|---|---|
| Support Vector Machine (RBF) | 0.935 | 0.991 |
| Elastic Net (regularised regression) | 0.907 | 0.981 |
| XGBoost | 0.879 | 0.970 |
| Random Forest | 0.811 | 0.939 |
| K Nearest Neighbours | 0.757 | 0.872 |

<img width="2112" height="960" alt="model_comparison" src="https://github.com/user-attachments/assets/a8bc22da-acec-460e-96a8-0aca449fcac8" />


The winning model was then tested once on a separate 20% of the data. It classified 3,288 of 3,637 listings correctly (about 90%).

<img width="1920" height="768" alt="confusion_matrix" src="https://github.com/user-attachments/assets/61ea4cd1-3015-473d-bd3e-0f15358aa50f" />

Reading the chart: each row is what the model predicted and each column is the true band. Almost all mistakes are between neighbouring bands, such as a high mid range listing predicted as luxury. Only 4 luxury listings were predicted as budget, and no budget listings were predicted as luxury.

### What does this mean?

- Price band can be predicted reliably from a listing's basic characteristics, which suggests the market is structured and not random.
- A pricing tool built on this approach could give hosts a sensible band to aim for, and give travellers a quick check on value.
- The hardest listings to classify sit near the band edges, which is where human judgement matters most.

## 3. Pipeline

The whole workflow lives in one reproducible R Markdown file, from raw CSV to final evaluation.

### Step 1. Data source

Inside Airbnb publishes public listing data for cities around the world. This project uses the Sydney listings snapshot from June 2025, which holds 18,187 rows and 79 columns of mixed types (text, numbers, dates and true or false flags). The data is stored in `data/listings.csv`.

### Step 2. Cleaning

- Removed columns that were not useful for modelling, such as web links, long text descriptions and scraping metadata.
- Standardised formats: converted price to a number, created the three price bands, parsed dates and turned yes or no fields into true or false.
- Removed unrealistic records, such as extreme price and bathroom outliers.

### Step 3. Exploration

Charts and a map of the data were used to understand price by room type, by suburb location, by number of bathrooms, and to find the numeric features most correlated with price. The price distribution is heavily skewed to the right, which is why classifying into bands is more stable than predicting an exact price.

### Step 4. Preprocessing

- Missing values were filled in with missForest, an imputation method that uses random forests to estimate each gap from the other columns.
- Numeric features were standardised, then reduced with Principal Component Analysis, keeping enough components to explain 95% of the variation.
- One hot encoding turned categories into numeric columns.
- SMOTE balanced the three classes by creating synthetic examples of the smaller groups, so that no band dominates training.

<img width="2688" height="1152" alt="class_balance_smote" src="https://github.com/user-attachments/assets/4292a035-887d-4b58-9b9d-29e652576a50" />


### Step 5. Modelling with nested cross validation

Five candidate models were compared: Elastic Net, Random Forest, XGBoost, K Nearest Neighbours and an RBF Support Vector Machine.

Nested cross validation keeps model tuning separate from model scoring. An inner loop (3 folds) chooses each model's settings. An outer loop (4 folds) then scores the tuned model on data it has not seen. This reduces the risk of an over optimistic score caused by tuning and testing on the same data.

### Step 6. Final evaluation

The best model (SVM with cost 100 and sigma 0.01) was fitted on a stratified 80% training split and tested once on the remaining 20%. The preprocessing recipe was estimated on the training data only and then applied to the test data.

## 4. Limitations and next steps

These were the drawbacks of the version presented in the final project for STAT5003. Now that this is a personal project, they are the next steps I plan to fix.

- About 13.5% of listings have no price, and these rows were kept and filled in by imputation.
- The column `estimated_revenue_l365d` was kept as a feature. In this dataset it equals price multiplied by estimated occupancy.
- Imputation was run on the full dataset before cross validation.

I expect the accuracy figures to fall once these are fixed, and the revised results will be published here.

## 5. Repository structure

```
airbnb-sydney-classification/
├── Airbnb_sydney_classification.Rmd   Full analysis (code and narrative).
├── data/
│   └── listings.csv                   Inside Airbnb Sydney listings, June 2025.
├── docs/
│   └── index.html                     Rendered report.
├── LICENSE
└── README.md
```

### How to run it

1. Install R and RStudio.
2. Install the packages: `tidyverse`, `tidymodels`, `themis`, `missForest`, `leaflet`, `cowplot`, `patchwork`, `gridExtra`, `knitr`, `kableExtra`, `doParallel`.
3. Open `Airbnb_sydney_classification.Rmd` and click Knit. The nested cross validation is computationally heavy, so expect a long run time.

## 6. Data and credits

- Listings data: [Inside Airbnb](https://insideairbnb.com/get-the-data/), Sydney, June 2025. Please check their site for the current terms of use.
- Market context: CBRE (2024), Hotels Australia Overview and Outlook 2024. Airbtics (2025).

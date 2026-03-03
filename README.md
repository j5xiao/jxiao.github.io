# User satisfaction prediction analysis based on recipe nutrition and complexity
by Jinxin Xiao


## Overview
This project analyzes the nutritional data and preparation complexity of recipes, and uses machine learning to predict user satisfaction.

## Introduction
This study aims to explore the intrinsic logic between "taste preferences" and "healthy ingredients" in modern food culture by deeply analyzing the massive amount of recipes and accompanying review data on the Food.com platform. The project will focus on user behavior characteristics and recipe content characteristics: on the one hand, by analyzing the emotional tendencies expressed by users in their reviews, key taste factors (such as the salt-to-sweet ratio and fat content) leading to high or low scores will be identified, thus outlining the taste profiles of different user groups; on the other hand, by combining the number of cooking steps, time, and detailed nutritional composition (protein, fat, sugar, etc.) of the recipes, the correlation between healthy eating and cooking complexity will be assessed. Ultimately, this project hopes to establish a multi-dimensional evaluation system that can not only recommend dishes that match users' historical preferences but also provide scientific insights and suggestions for healthy dietary ratios and recipe improvements from a data perspective.
### Dataset Introduction: User Reviews
This dataset, represented as df_reviews, contains the feedback and interaction history from users regarding various recipes. It serves as the primary source for our sentiment analysis and target labels.

📊 Overview
Total Rows: 731,927
Total Columns: 5

📂 Feature List
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| **`user_id`** | int64 | The unique identifier for the user who posted the review. |
| **`recipe_id`** | int64 | The unique identifier for the recipe being reviewed. |
| **`date`** | object | The date when the review was submitted. |
| **`rating`** | int64 | The numerical score provided by the user (1–5 scale). |
| **`review`** | object | The text content of the user's feedback and experience. |

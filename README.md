=============================
<!--Created Anchor links to navigate read me better-->

- [Project Description](#project-description)
- [Project Goal](#project-goal)
- [Initial Thoughts](#initial-thoughts)
- [Plan](#the-plan)
- [Data Dictionary](#data-dictionary)
- [Steps to Reproduce](#steps-to-reproduce) 
- [Conclusions](#conclusions)
	- [Takeaway and Key Findings](#takeaways-and-key-findings)
	- [Reccomendations](#recommendations)
	- [Next Steps](#next-steps)

----------------------------------

# Project Zillow

Predict the tax assessed property value of Zillow Single Family Residential properties with transaction date in 2017

### Project Description

As the most-visited real estate website in the United States, Zillow and its affiliates offer customers an on-demand experience for selling, buying, renting and financing with transparency and nearly seamless end-to-end service. I have decided to look into the different elements that determine tax assessed property value.

### Project Goal

* Discover drivers of property value
* Use drivers to develop a machine learning model to predict property value
* This information could be used to further our understanding of how Single Family properties are tax assessed for their property value

### Initial Thoughts

My initial hypothesis is that drivers of tax assessed property value will be the elements like number of rooms, square feet, and location.

## The Plan

* Acquire data from Codeup MySQL DB
* Prepare data
  * Create Engineered columns from existing data
* Explore data in search of drivers of property value
  * Answer the following initial questions
    * Is there a correlation between area and property value?
    * Is there a correlation between age and property value?
    * Is there a correlation between the room count and property value?
    * Is there a difference in average property value between counties?
* Develop a Model to predict property value
  * Use drivers identified in explore to help build predictive models of different types
  * Evaluate models on train and validate data
  * Select the best model based on $RMSE$ and $R^2$
  * Evaluate the best model on test data
* Draw conclusions

## Data Dictionary

|**Feature**|**Values**|**Definition**|
|:--------------------:|:---------------------------:|:-------------------------------------------------------- |
| *customer_id*           | Alpha-numeric               | Unique ID for each customer                              |
| *gender*                | Female/Male                 | Gender of customer                                       |
| *senior_citizen*        | True=1/False=0              | Whether customer is a senior citizen or not              |
| *partner*               | True=1/False=0              | True=1/False=0, whether customer has a partner or not    |
| *dependents*            | True=1/False=0              | True=1/False=0, whether customer has dependents or not   |
| *phone_service*         | True=1/False=0              | True=1/False=0, whether customer has phone service or not|
| *multiple_lines*        | Yes/No/No phone service     | Whether customer has multiple lines or not               |
| *internet_service_type* | None/DSL/Fiber Optic        | Which internet service customer has                      |
| *online_security*       | Yes/No/No internet service  | Whether customer has online_security                     |
| *online_backup*         | Yes/No/No internet service  | Whether customer has online_backup                       |
| *device_protection*     | Yes/No/No internet service  | Whether customer has device_protection                   |
| *tech_support*          | Yes/No/No internet service  | Whether customer has tech_support                        |
| *streaming_tv*          | Yes/No/No internet service  | Whether customer has streaming_tv                        |
| *streaming_movies*      | Yes/No/No internet service  | Whether customer has streaming_movies                    |
| *contract_type*         | 3 options                   | Month-to-Month/One-year/Two-year, term of contract       |
| *payment_type*          | 4 options (2 auto)          | Customer payment method                                  |
| *paperless_billing*     | True=1/False=0              | Whether a customer has paperless billing enabled         |
| *monthly_charges*       | Numeric USD                 | Amount customer is charged monthly                       |
| *total_charges*         | Numeric USD                 | Total amount customer has been charged                   |
| *tenure*                | Numeric                     | Number of months customer has stayed                     |
| *churn* (target)        | True=1/False=0              | Whether or not the customer has churned                  |
| *Additional Features*   | True=1/False=0              | Encoded values for categorical data                      |

FIPS County Codes:

* 06037 = LA County, CA
* 06059 = Orange County, CA
* 06111 = Ventura County, CA

## Steps to Reproduce

1) Clone this repo
2) If you have access to the Codeup MySQL DB:
   - Save **env.py** in the repo w/ `user`, `password`, and `host` variables
   - Run notebook
3) If you don't have access:
   - Request access from Codeup
   - Do step 2

# Conclusions

#### Takeaways and Key Findings

* The younger the property the better for property value
* The bigger the living area the bigger the property value
* Location matters for property value
* Model still needs improvement

### Recommendations and Next Steps

* It would nice to have the data to check if the included appliances or the type of heating services (gas or electric) of the property would affect property value
* More time is needed to work on features to better improve the model
    - latitude and longitude could hopefully give insights into cities and neighborhoods with higher or lower property values
    - pools and garages could also be looked into

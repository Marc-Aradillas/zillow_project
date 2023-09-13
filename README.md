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

As the leading real estate platform in the United States, Zillow and its affiliated services provide customers with an accessible and user-friendly experience for selling, buying, renting, and financing properties. With a commitment to transparency and delivering nearly seamless end-to-end service, I have undertaken an investigation into the various factors that influence tax-assessed property values.

### Project Goal

* Identify drivers of property value.
* Use drivers to develop a machine learning model to predict Single Family Residential property value.
* Develop understanding of how Single Family properties are tax assessed for their property value.

### Initial Thoughts

My initial hypothesis is that drivers of tax assessed property value will be the elements like number of different room types, square feet of different property areas, and year built.

## The Plan

* Acquire data from Codeup MySQL DB
* Prepare data
  * Create Engineered columns from existing data
* Explore data in search of drivers of property value
  * Answer the following initial questions
	* Do bedrooms have a relationship with home value?
	* Does home value have a relationship with the year property was built?
	* Does home value have a correlation with area?  
	* Does home value have a correlation with the property lot area?  
* Develop a Model to predict property value
  * Use drivers identified in explore to help build predictive models of different types
  * Feature engineer data, no preprocess to include all values.
  * Evaluate models on train and validate data
  * Select the best model based on $RMSE$ and $R^2$
  * Evaluate the best model on test data
* Draw conclusions

## Data Dictionary

|**Feature**|**Data Type**|**Definition**|
|:--------------------:|:-------------------:|:-------------------------------------------------------- |
| *parcel_id*           | int64                | Unique parcel identifier                                   |
| *latitude*            | float64              | Latitude coordinate of the property                        |
| *longitude*           | float64              | Longitude coordinate of the property                       |
| *lot_area*            | int32                | Area of the property lot in square feet                    |
| *region_id_county*    | int32                | Identifier for the county where the property is located    |
| *region_id_zip*       | int32                | Identifier for the zip code where the property is located   |
| *property_county_landuse_code* | object     | Land use code for the county of the property          |
| *property_zoning_desc* | object             | Zoning description for the property                        |
| *bathrooms*           | float64              | Number of bathrooms in the property                        |
| *bedrooms*            | int32                | Number of bedrooms in the property                         |
| *calc_bath_count*     | float64              | Calculated bathroom count                                   |
| *area*                | int32                | Calculated area of the property in square feet              |
| *basement_sqft*       | float64              | Square footage of the basement                              |
| *finished_square_feet_12* | float64          | Square footage of finished living area                      |
| *finished_sqft_15*    | object               | Square footage of finished living area                      |
| *finishedsqft50*      | float64              | Square footage of finished living area (50th percentile)    |
| *fips*                | int32                | Federal Information Processing Standards (FIPS) code        |
| *rooms*               | int32                | Number of rooms in the property                             |
| *num_stories*         | float64              | Number of stories in the property                           |
| *year_built*          | int32                | Year the property was built                                 |
| *property_landuse_type_id* | int64           | Identifier for the property's land use type               |
| *ac_type_id*          | float64              | Identifier for the air conditioning type                   |
| *building_quality_type_id* | float64         | Identifier for the building quality type                  |
| *heating_or_system_type_id* | float64        | Identifier for the heating or system type                 |
| *deck_type_id*        | float64              | Identifier for the deck type                                |
| *unit_cnt*            | float64              | Count of units on the property                              |
| *garage_car_cnt*      | float64              | Count of garage cars                                       |
| *garage_total_sqft*   | float64              | Total square footage of the garage                         |
| *pool_cnt*            | float64              | Count of pools on the property                              |
| *pool_size_sum*       | float64              | Sum of pool sizes in square feet                            |
| *pool_type_id_2*      | float64              | Identifier for pool type 2                                  |
| *pool_type_id_7*      | float64              | Identifier for pool type 7                                  |
| *fire_place_cnt*      | float64              | Count of fireplaces in the property                         |
| *fire_place_flag*     | int64                | Flag indicating the presence of a fireplace                |
| *has_hot_tub_or_spa*  | float64              | Indicates if the property has a hot tub or spa              |
| *patio_sqft*          | float64              | Square footage of the patio                                 |
| *storage_sqft*        | float64              | Square footage of storage                                   |
| *home_value*          | int32                | Tax assessed property value                                 |
| *tax_delinquency_flag* | int64              | Flag indicating tax delinquency                             |
| *raw_census_tract_and_block* | float64      | Raw census tract and block identifier                     |
| *n-life*              | float64              | Numeric feature: n-life                                    |
| *n-living_area_error* | float64              | Numeric feature: n-living_area_error                       |
| *n-living_area_prop*  | float64              | Numeric feature: n-living_area_prop                        |
| *n-living_area_prop2* | object               | Numeric feature: n-living_area_prop2                       |
| *n-extra_space*       | float64              | Numeric feature: n-extra_space                             |
| *n-extra_space-2*     | object               | Numeric feature: n-extra_space-2                           |
| *n-total_rooms*       | float64              | Numeric feature: n-total_rooms                             |
| *n-av_room_size*      | float64              | Numeric feature: n-av_room_size                            |
| *n-extra_rooms*       | float64              | Numeric feature: n-extra_rooms                             |
| *n-gar_pool_ac*       | int32                | Numeric feature: n-gar_pool_ac                             |
| *n-location*          | float64              | Numeric feature: n-location                                |
| *n-location-2*        | float64              | Numeric feature: n-location-2                              |
| *n-location-2round*   | float64              | Numeric feature: n-location-2round                         |
| *n-latitude-round*    | float64              | Numeric feature: n-latitude-round                          |
| *n-longitude-round*   | float64              | Numeric feature: n-longitude-round                         |
| *n-zip_count*         | int64                | Numeric feature: n-zip_count                               |
| *n-county_count*      | int64                | Numeric feature: n-county_count                            |
| *n-ac_ind*            | int32                | Numeric feature: n-ac_ind                                  |
| *n-heat_ind*          | int32                | Numeric feature: n-heat_ind                                |
| *n-prop_type*         | object               | Numeric feature: n-prop_type                               |

Feature Engineered columns
* *property_size* | Categorical | Size category of the property |
* *Large* - Large-sized property
* *Medium* - Medium-sized property
* *Small* - Small-sized property



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

# Conclusion

## Takeaways and Key Findings

- Home details such as the calculated area of the home, lot area, number of bedrooms, number of bathrooms, and year built are significant drivers of home value.
- Calculated area is the most important feature across all factors.
- Higher-priced and larger single residential family properties appear to be concentrated in Ventura County, while smaller homes are spread across Orange and Los Angeles counties.
- Larger living areas correlate with higher property values.
- Location plays a crucial role in property value.


## Model Improvement
- The model still requires further improvement.

# Recommendations and Next Steps

- If the data contained detailed of hard appliances attached or amenities homebuyers seek in single family residential properties maybe this could affect property value.

- Given more time, the following actions could be considered:
  - Gather more data to improve model performance.
  - Feature engineer new variables to enhance model understanding.
      - trasnaction dates for value over time
  - Fine-tune model parameters for better performance.

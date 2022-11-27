# ML Trading Strategy Version 2

This repository is dedicated to present a machine learning approach to predict stock movements. 

# Results

### Model 1 ###

Description - 

Input variables - eps_actual, month, eps_surprise, eps_growth, volume, normalized_value, 20 day logistic regression, 10 day logistic regression, 3 day logistic regression and 5 day logistic regression

Output variables - probability of the data point being a buy or a sell

![coefficients](https://user-images.githubusercontent.com/85404022/203859474-dfdfe3e7-1a91-4cd3-9983-a68c0f3b0ada.jpg)


Using past eps, eps surprise & eps growth has little effect on the output.


### Model 2 ###

Description - 

Input variables - volume, month, normalized_value, 20 day logistic regression, 10 day logistic regression, 3 day logistic regression and 5 day logistic regression

Output variables - probability of the data point being a buy or a sell

![coefficients](https://user-images.githubusercontent.com/85404022/204144762-4f6fbeef-dcb1-45bb-95c8-afd395a6a767.jpg)

Sample prediction on Visa (V) stock.

![V](https://user-images.githubusercontent.com/85404022/204144822-8ecda33e-b301-4de5-8fe2-6966230ac30d.png)

### Model 2 - Rolling average probability in the last 3 days ###

![V](https://user-images.githubusercontent.com/85404022/204152236-63811069-785a-4949-874b-12768ffa3564.png)


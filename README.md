# StatsML-Toolkit-Probability-to-Regression-Analysis
"StatML-ProbToRegress" is a repository focusing on statistical modeling and machine learning. It covers probability in card games, defect analysis, and product testing, along with polynomial and logistic regression applications in various fields. The project also delves into parameter estimation techniques and handling imbalanced datasets.

Task 1.1
1.1.a The Theory that is used to solve the problem.

Based on the given problem that contains a set of cards with 4 suits, and after randomly drawing 5 cards from the deck and not replacing it. This involves probability. However, there are several theories including bayes Theorem. In this situation the ‘Combinatorics Theory’ can be used. 
1.1.b Describing the theory and the equations.

Combinatorics probability theory can be defined as the process of counting, arranging, and selecting objects without considering their order to calculate the possible combinations of cards that can be drawn from the deck.
For instance, based on the above given situation. When you extract a card there are four possible outcomes (hearts, diamonds, clubs, spades. Therefore, each outcome maybe assumed to be equally likely. 
To develop the equation, it is important to identify the contents of the card pack and the outcome. 
	Total number of cards (n=52)
	Randomly drawn cards (k=5)
The equation as follows to calculate the probability and the explanation are as follows. 

C(n,k) =  n!/((k!×(n - k)!))

	C(n,k) = The number of combinations of k items, chosen from n items.
	n! = factorial of n
	k! = factorial of k
1.1.c Probability that all 5 cards are of the same suit.

Based on the above-mentioned observation of the probability of the probability that all 5 cards are of the same suit, or which is know as a ‘flush’. 
To derive a flush, the drawn cards could be all hearts, all clubs, all diamonds, or all spades. As long as the drawn 5 cards of the same suit. 
	Identifying the total number of outcomes.
To determine the total possible outcomes, when drawing 5 cards form the deck of 52 cards and not replacing it. The total number of items (52) and the count of items we want to select is 5. The formula is as follows:
C(52,5)=52!/((5!×(52-5)!) )
=52!/((5!×47!))

=((52×51×50×49×48×47!)/((5!×47!) )
=2,598,960
	
Therefore, the total number of possible outcomes for drawing 5 cards are 2,598,960.
	Calculating the total number of favorable outcomes. 
As all the four suits have the same count of cards (13), therefore, the calculating the probability all 5 cards are of the same suit. Furthermore, the number of outcomes is multiplied by 4 for all four different types of suits. 
Calculation for number of favorable outcomes for one suit.
C(13,5)=13!/((5!8(13-5)!)
=1,287
Calculation for number of favorable outcomes for four suits.
1,287 ×4=5,148

	Calculation of the probability
After calculating the total number of outcomes and the total number of outcomes. The probability of all 5 cards drawn being the same suit is calculated. 
Probability=5,148/2,598,960
=0.00198
Based on the above probability calculation the probability of all 5 cards being in the same suit is 0.19%.
1.1.d Using python coding to demonstrate the same steps. 

Based on the below code, the factorial function is used to calculate the factorial of number as input. Thereafter the total number of favorable outcomes are calculated. Thereafter the total number of favorable outcomes are calculated and multiply by the four suits. Finally calculating the probability. 
 
Figure 1 Python coding demonstrate steps.
	
	
1.1.e Using ‘math’ library in python coding to demonstrate the same steps. 

To use the math module the library is imported. Total ways are calculated the total number of ways 5 cards are randomly drawn using the ‘comb’ function. The total number of favorable outcomes for 4 suits is calculated by multiplying the 4 types suits with the combinations when choosing 5 cards which are drawn from the set of the number of cards which is in one set. Finally calculating the probability, where total number of favorable outcomes for 4 suits divided by the total number of outcomes.
 
Figure 2 Math function coding demonstrate steps.



Task 1.2
1.2.a The probability that the entire batch of 10 bulbs is defective.

Based on the manufacturing plant, the observations are as follows prior to calculating the probability that the entire batch of 10 bulbs is defective. The data are as follows.
	Batch size 	10
	Defective bulbs        3
	Failure rate	    2%
Based on the above given data points the probability of a single light bult being non defective is 98%.
To calculate the entire batch of 10 bulbs is defective the binomial distribution method is used. 
P(x)=(_x^n)C  p^x q^(n-x)   
=(10¦10) 〖 0.02〗^10   〖0.98〗^(10-10)
=10!/(10!*0!) 〖 0.02〗^10   〖0.98〗^(10-10)
=1.024* 10^(-17)
=0.000000000000000001024




1.2.b.i SciPy library for calculations and defining the variables.

 
Figure 3 Defining the Variables
1.2.b.ii Using ‘pmf’ library for calculation the likelihood.

 
Figure 4 ‘pmf’ function to calculate the likelihood.
1.2.b.iii Calculate evidence or marginal likelihood.

 
Figure 5 Calculate evidence or marginal likelihood.
1.2.b.iv Calculate the posterior probability distribution.
 
Figure 6 Calculate the posterior probability distribution.
1.3 Two Sample test to compare the strength of new product vs old product.

Based on the in information that is provided the old and new product information is complied.
Old Product
Sample mean (x_1) = 85
Sample standard deviation (s_1) = 10
Sample size (n_1) = 30
New Product
Sample mean (x_2) = 90
Sample standard deviation (s_2) = 8
Sample size (n_2) = 30
As the second step generating the null hypothesis and the alternative hypothesis.
 H_0 : Old product sample mean 〖(μ〗_1) = New product sample mean (μ_2)
H_1 : Old product sample mean 〖(μ〗_1) > New product sample mean (μ_2)
The significance level is (α) = 0.05.
Since this a two sample test and the formula calculating pooled standard deviation.
s_p = sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2))
s_p = sqrt(((n₁-1)*s₁² + (n₂-1)*s₂²) / (n₁ + n₂ - 2))
s_p = sqrt(((30-1)10² + (30-1)8²) / (30 + 30 - 2))
s_p = sqrt((29000 + 1856) / 58)
s_p = sqrt(4756/58)
s_p = sqrt(82) 
s_p = 9.05
Calculating the test statistic.
t = (x₁ - x₂) / (s_p * sqrt(1/n₁ + 1/n₂))
t = (85 - 90) / (9.05 * sqrt(1/30 + 1/30))
t = -5 / (9.05 * sqrt(1/15))
t = -5 / (9.05 * 0.258)
t = -5 / 2.33
t = -2.14



Since the alternative hypothesis is a one tailed calculating the degree of freedom.
Degrees of freedom = n_1+ n_2-2
                                =30+30-2
                    =58
Based on the information provided the critical value for a one tailed, t test with 58 degrees of freedom at a significance level of 0.05 is 1.671.
The calculated t statistics value is -2.14 is less than the critical value of -1.67. As a result of this we fail to reject the null hypothesis and there is not enough evidence that the new product has a higher strength than the old product.
Task 2.1
2.1. Calculating the Error of Polynomial Regression Model
Based on the given regression equation, 
Y= 〖-0.001X〗^3+ 〖0.032X〗^2+ 0.004X+ + 0.209
Calculating the predicted values of Y and substituting the values of X to the equation. 
X = 0
Predicted Y= 〖-0.001(0)〗^3+ 〖0.032(0)〗^2+ 0.004(0)+ + 0.209
                    = 0.209
X = 1
Predicted Y= 〖-0.001(1)〗^3+ 〖0.032(1)〗^2+ 0.004(1)+ + 0.209
                    = 0.244
X = 2
Predicted Y= 〖-0.001(2)〗^3+ 〖0.032(2)〗^2+ 0.004(2)+ + 0.209
                    =0.337
X = 3
Predicted Y= 〖-0.001(3)〗^3+ 〖0.032(3)〗^2+ 0.004(3)+ + 0.209
                    = 0.482

X = 4
Predicted Y= 〖-0.001(4)〗^3+ 〖0.032(4)〗^2+ 0.004(4)+ + 0.209
                    = 0.673
X = 5
Predicted Y= 〖-0.001(5)〗^3+ 〖0.032(5)〗^2+ 0.004(5)+ + 0.209
                    =0.904
X = 6
Predicted Y= 〖-0.001(6)〗^3+ 〖0.032(6)〗^2+ 0.004(6)+ + 0.209
                    =1.169
X = 7
Predicted Y= 〖-0.001(7)〗^3+ 〖0.032(7)〗^2+ 0.004(7)+ + 0.209
                    =1.462
X = 8
Predicted Y= 〖-0.001(8)〗^3+ 〖0.032(8)〗^2+ 0.004(8)+ + 0.209
                    =1.777
X = 9
Predicted Y= 〖-0.001(9)〗^3+ 〖0.032(9)〗^2+ 0.004(8)+ + 0.209
                    =2.108
Thereafter, the residuals are calculated for the data points. 
Residual = Observed Y – Predicted Y
X = 0
Residuals = 0.2 – 0.209
                = -0.009
X = 1
Residuals = 0.6 - 0.244
                = 0.356
X = 2
Residuals = 1.1 - 0.337
                = 0.763
X = 3
Residuals = 1.5 - 0.482
                = 1.018
X = 4
Residuals = 1.8 - 0.673
                = 1.127
X = 5
Residuals = 2.0 - 0.904
                = 1.096
X = 6
Residuals = 2.1 - 1.169
                = 0.931
X = 7
Residuals = 2.2 - 1.462
                = 0.738
X = 8
Residuals = 2.3 - 1.777
                = 0.523
X = 9
Residuals = 2.4 - 2.108
                = 0.292
X	Y	Predicted Y	Residual
0	0.2	0.209	-0.009
1	0.6	0.244	0.356
2	1.1	0.337	0.763
3	1.5	0.482	1.018
4	1.8	0.673	1.127
5	2.0	0.904	1.096
6	2.1	1.169	0.931
7	2.2	1.462	0.738
8	2.3	1.777	0.523
9	2.4	2.108	0.292



Based on the above table the predicted Y values and residual values are calculated. 
To find the error of the regression for these data points as a sum, the following mean squared error is used.
MSE = ((〖-0.009〗^2+〖0.356〗^2+〖0.763〗^2+〖1.018〗^2+〖1.127〗^2+〖1.096〗^2+〖0.931〗^2+〖0.738〗^2+〖0.523〗^2+〖0.292〗^2))/10
MSE = 0.5986853
2.2 Logistic regression model to predict Y based on X
	Predicted Y = 0	Predicted Y = 1
Actual Y = 0	300	100
Actual Y = 1	150	450

True Positive (TP) = 450
True Negative (TN) = 300
False Positive (FP) = 100
False Negative (FN) = 150





2.2.a Calculating the accuracy, precision, recall and F1 score for the logistics regression model.

Accuracy =((True Positives + True Negatives))/((True Positives + False Positives + True Negatives + False Negatives))
=((450+300))/((450+100+300+150))
=750/1000
=0.75
Accuracy=75%

Precision =(True Positives)/((True Positives + False Positives ))
=  450/((450+100))
=0.8182
Precision =81%
Recall =(True Positives)/((True Positives + False Negatives))
=  450/(450+150)
= 0.75
Recall = 75%

F1 Score=  ((2 ×Precision×Recall) )/((Precision+Recall))  
=  ((2 ×0.8182×0.75) )/((0.8182+0.75))  
0.78
F1 Score = 78
2.2.b Description of the given data set.

The given data set consists of observations and the given confusion matrix shows the true positive, true negative, false positive and false negative values. The data set gives a data set based on 1000 observations. Where, by using logistic regression to predict the value of Y based on the given value of X.
The accuracy that was calculated, to check how correct it is against the given model. Where the proportion of correct predictions against the total dataset. Based on the calculation the logistics regression model has a 75% accuracy rate and correctly predicts the outcome of Y for 75% of its observations in the given dataset. 
The precision signifies the accuracy when it makes a prediction. In other words, based on the precision value of 81%, the model is 81% correct in predicting that a instance belongs to the positive class. 
Recall is used to if the model can correctly identify positive instances. Based on the 75% recall value, it shows that the based on the true positives, identifies 75% of the actual positive instances. 
The F1 score is to evaluate based on the recall and precision. The score is given between 0 and 1, and one being the highest. Based on the model the F1 score is 0.78 and close to 1. The measurement provides a balance between the recall and precision based on the harmonic mean. 
2.2.c Real world applications of Logistic Regression and the practice. 

Logistics regression is a statistical method that is used to predict the probability of an event happening. It was first introduced in 1944 to predict the probability of patient developing a particular disease based on their previous medical history. Therefore, logistic regression was first practiced in health care and currently to many other fields. To understand how logistics regression has influenced and how some sectors are practicing it based on the industry it operates in. 
Heart attack prediction in Healthcare Industry.
Logistic regression is used to identify how exercise (hours) and weight (kg) may impact heart attacks. The response variable will include two outcomes, where if a heart attack occurs or does not occur. Based on the model the change in exercise and weight can provide if an individual may have a heart attack or not. Doctors can identify these patients and give more focus how to monitor them more and reducing the risk of hart attacks. 
Sentiment Analysis on social media.
In social media there are several social media tools that provide sentiment analysis. Where if a post is ‘Positive’, ‘Negative’ or ‘Neutral’. In logistic regression, the model will learn the relationship between sentiment labels and inputs. Where the model will be able to provide the sentiment of specific keywords. This will in return help companies to handle large volume of data to identify the sentiments of customer mentions on social media platforms. 

Customer Churn Prediction.
A television service provider can predict the likelihood of customers churning from their cable service to other cable and satellite services. Based on the customer details, the logistics model can predict the likely hood of customers moving out to competition. Where the current provider can see the customers who likely to move and provide special offers to retain the existing customers. 
Fraudulent Transactions in Banking and Finance. 
Many carry out fraudulent transactions online and offline. Banks need to know if customers are carrying out any of these illegal transactions. Based on the transaction amount and the credit score of a user, a bank predicts if it is a fraudulent transaction. This will help banks to monitor of its customers and identify if any illegal transactions are taking place. 
If a customer purchase or conversion will occur.
Many companies spend large amount of money for promoting their service and products and in many cases, companies are unable to understand if a customer will purchase their product or service. Using logistic regression, the company can predict the likelihood of a sale being closed. By doing this, the company can offer a discount to customers who have a low conversion, to convert the purchase. 



2.2.d Different parameter estimation techniques affect the performance and interpretation.

Using different parameter estimate techniques including the following techniques can affect the performance of logistic regression models. 
	 Maximum Likelihood Estimation (MLE)
MLE is used to estimate the parameter values to make the observed data as likely as possible. 
Performance	Interpretation
It offers parameter estimates that closely match the data, resulting in effective prediction.	Estimates of the parameters from MLE can be used to calculate the outcome variable's log-odds. They show how the predictors have an impact on the likelihood of the result.

	Bayesian Methods
Bayesian method uses a method to update the current beliefs by combining prior knowledge and evidence as new knowledge or information is learned. 



Performance	Interpretation
Using prior believes will improve performance with prior knowledge. Specially when data is limited.	This helps to understand the uncertainty and make credible conclusions based on credible intervals. 
	Penalized Methods
This method ads a penalty in model fitting to prevent it from overfitting. This is to balance the fit of the data to the model. 
Performance	Interpretation
Helps in preventing overfitting and ability to generalize to new data.	By emphasizing the most relevant variables and reducing unwanted variables.

	Iteratively Reweighted Least Squares (IRLS)
IRLS method is used to set parameters for data sets to fit the model. It keeps adjusting its values until it finds the best ones. This helps to fit the data to the model by setting parameters.  
Performance	Interpretation
Handles nonlinear relationships to fit the model better and for better outcomes and performance. 	Gives how much the predictors influence the likelihood of outcomes.

2.2.e Logistic regression model adapted to handle imbalanced datasets.

There are several approaches that are available to handle imbalanced data sets.
	Resampling Technique
This method uses either increasing through the minority class or decreasing through the majority class. This will help for a more balanced class in the data.
	Class Weighting
During the model training process, higher weight is given to the minority class and reduce the impact of the class imbalance. 
	Cost Sensitive Learning
This involves different misclassification costs are assigned to each respective class and correctly predicting the minority class is set high importance. 
	Resampling and Algorithm Combination
Using resampling techniques and using boosting and bagging methods to create multiple models and combined to make predictions. 
	Synthetic Minority Oversampling Technique (SMOTE)
This method uses new examples from the minority class by blending existing examples to create new examples. 

2.3 Multiple Linear Regression
2.3.a Linear Regression Model Fitting
 
Figure 7 Linear Regression Model Fitting



2.3.b Generating a random dataset with variables to a Linear Regression Model Fitting
 
Figure 8 Generating a random dataset with variables to a Linear Regression Model Fitting
2.3.c Fit a multiple linear regression model to the generated dataset.
 
Figure 9 Fit a multiple linear regression model to the generated dataset.

2.3.d Plot the residuals vs fitted graph using matplotlib.
 
Figure 10 Plot the residuals vs fitted graph using matplotlib.

 
Figure 11 Residuals vs fitted Scatter plot

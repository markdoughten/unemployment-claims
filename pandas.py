# import pandas library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
import os
import glob
from sklearn.model_selection import train_test_split

# read the local file by the path, and assign it to variable "PACaseloads"
PACaseloads = pd.read_csv("/Users/mdoughten/Desktop/Python/"
                        "Public_Assistance__PA__Caseloads_and_Expenditures__Beginning_2002.csv")

# sort the PACaseloads by Year and Month Code so it does in chronological order
PACaseloads.sort_values(["Year", "Month Code"], axis=0,
                      ascending=[False, False], inplace=True)

# delete 2019 from each data set and 2002-2005 to standardize the two datasets
indexNames = PACaseloads[(PACaseloads['Year'] == 2019)].index
PACaseloads.drop(indexNames, inplace=True)

indexNames = PACaseloads[(PACaseloads['Year'] <= 2005)].index
PACaseloads.drop(indexNames, inplace=True)

# Print the adjusted year column minimum and maximum
print("This is the maximum year:", PACaseloads['Year'].max())
print("This is the minimum year:", PACaseloads['Year'].min())

PACaseloads = PACaseloads.drop(columns=["Year", 'Month', 'Month Code', 'District Code', 'District'])

# debugger to verify that the columns were dropped
for column in ['Year', 'Month', 'Month Code', 'District Code', 'District']:
    assert column not in PACaseloads.columns

# export the sorted version to the file location
export_csv = PACaseloads.to_csv(r"/Users/mdoughten/Desktop/Python/Sorted Datasets/"
                              r"sortedPACaseLoad_dataframe.csv", index=None, header=True)

# read the local file by the path, and assign it to variable "PACasecloses"
PACasecloses = pd.read_csv("/Users/mdoughten/Desktop/Python/"
                         "Public_Assistance_Case_Closings_by_Reason_for_Closing___Beginning_2006.csv")

# sort the values similar to PACasecloses to standardize the data
PACasecloses.sort_values(["Year", "Month Code"], axis=0,
                       ascending=[False, False], inplace=True)

# delete 2019 from PACasecloses
indexNames = PACasecloses[PACasecloses['Year'] == 2019].index
PACasecloses.drop(indexNames, inplace=True)

# remove the duplication between August-July
PACasecloses = PACasecloses.drop_duplicates()

# Export PACasecloses to the local drive
export_csv = PACasecloses.to_csv(r"/Users/mdoughten/Desktop/Python/Sorted Datasets/"
                               r"/sortedPACloseLoad_dataframe.csv", index=None, header=True)

# combines the two files into one csv
os.chdir("/Users/mdoughten/Desktop/Python/Sorted Datasets")
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
PACaseloadclose = pd.concat([pd.read_csv(f) for f in all_filenames], axis=1)

PACaseloadclose['Total Family Assistance Case Closings'] = \
    PACaseloadclose['Family Assistance (FA) Case Closings - Client Request'] + \
    PACaseloadclose['FA Case Closings - Financial Issues'] +  \
    PACaseloadclose['FA Case Closings - Residence Issues'] +  \
    PACaseloadclose['FA Case Closings - Compliance Issues / Employment'] +   \
    PACaseloadclose['FA Case Closings - Compliance Issues / Other'] + \
    PACaseloadclose['FA Case Closings - Other']

# group by year to show the increase or decrease in public assistance cases
PACaseloadclose_year = PACaseloadclose.groupby('Year', as_index=False).sum()

PACaseloadclose_year = PACaseloadclose_year.drop(columns=['Month Code', 'District Code'])
for column in ['Month Code', 'District Code']:
    assert column not in PACaseloadclose_year.columns

for column in ['Total Family Assistance Case Closings']:
    assert column in PACaseloadclose_year.columns

# plot the total temporary assistance recipients against year and to see increase or decrease
plt.plot('Year', 'Total Temporary Assistance Recipients', label="Total Temporary Assistance Recipients",
         data=PACaseloadclose_year)
plt.plot('Year', 'Total Temporary Assistance Cases', "--", label="Total Temporary Assistance Cases",
         data=PACaseloadclose_year);
plt.legend(loc=0)
plt.show()
plt.close()

# as the cases increase the total recipients increases, justified by looking at the correlation and p-value between the
# between the two variables
pearson_coef, p_value = stats.pearsonr(PACaseloadclose_year['Total Temporary Assistance Recipients'],
                                       PACaseloadclose_year['Total Temporary Assistance Cases'])
print("The Pearson Correlation Coefficient is", pearson_coef, "with a P-value of P =", p_value,
      " for Total Temporary Assistance Recipients and Total Temporary Assistance Cases")

# the The Pearson Correlation Coefficient between total cases and total recipients is close to 1,
# the two variables have a positive linear correlation
# the P-value is less than 0.001, there is strong evidence that the correlation is significant

# a linear regression to predict the total number of recipients bases on the total number of cases
lm = LinearRegression()
X = PACaseloadclose_year[['Total Temporary Assistance Recipients']]
Y = PACaseloadclose_year['Total Temporary Assistance Cases']
lm.fit(X,Y)
Yhat=lm.predict(X)
Yhat[0:5]
print("This is the intercept:", lm.intercept_)
print("This is the value of the slope:", lm.coef_)

pearson_coef, p_value = stats.pearsonr(PACaseloadclose_year['Family Assistance Federally Participating Cases'],
                                       PACaseloadclose_year['Total Family Assistance Case Closings'])
print("The Pearson Correlation Coefficient is", pearson_coef, "with a P-value of P =", p_value,
      " for Family Assistance Federally Participating Cases and Total Family Assistance Case Closings")

# the The Pearson Correlation Coefficient between total cases and total recipients is close to 1,
# the two variables have a positive linear correlation
# the P-value is less than 0.001, there is strong evidence that the correlation is significant

# a linear regression to predict the total number of recipients bases on the total number of cases
lm = LinearRegression()
X = PACaseloadclose_year[['Family Assistance Federally Participating Cases']]
Y = PACaseloadclose_year['Total Family Assistance Case Closings']
lm.fit(X,Y)
Yhat=lm.predict(X)
Yhat[0:5]
print("This is the intercept:", lm.intercept_)
print("This is the value of the slope:", lm.coef_)

# scatter plot to visualize the correlation between Total Temporary Assistance Recipients
# and Total Temporary Assistance Cases

width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="Family Assistance Federally Participating Cases",
            y="Total Family Assistance Case Closings", data=PACaseloadclose_year)
plt.ylim(0,)
plt.title("Family Assistance Federally Participating Versuses Total Family Assistance Case Closings")
plt.show()
plt.close()

# set up the training data for the predictive model, seperating data into training and test
# attempting to perdict the family assistance federally participating cases

y_data = PACaseloadclose["Family Assistance Federally Participating Cases"]
x_data = PACaseloadclose.drop("Family Assistance Federally Participating Cases", axis =1)

x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size=0.4, random_state=0)
print("Number of test samples: ", x_test1.shape[0])
print("Number of training samples: ", x_train1.shape[0])

# plot a linear regression to help plot the trained and the tested datasets

lr = LinearRegression()
lr.fit(x_train1[['Total Family Assistance Case Closings', 'Total Temporary Assistance Recipients',
                'Safety Net Federally Non-Participating Expenditures', 'Safety Net Assistance Expenditures']], y_train1)

yhat_train = lr.predict(x_train1[['Total Family Assistance Case Closings', 'Total Temporary Assistance Recipients',
                'Safety Net Federally Non-Participating Expenditures', 'Safety Net Assistance Expenditures']])
yhat_train[0:5]

yhat_test = lr.predict(x_test1[['Total Family Assistance Case Closings', 'Total Temporary Assistance Recipients',
                'Safety Net Federally Non-Participating Expenditures', 'Safety Net Assistance Expenditures']])
yhat_test[0:5]

# create plots to help show the difference between the actual and the original

sns.kdeplot(y_train1, label="Actual Values (Train)", shade = True, color ="b")
sns.kdeplot(yhat_train, label="Predicted Values (Train)", shade = True, color = "r")
plt.legend();
plt.show()
plt.close()

sns.kdeplot(y_test1, label="Actual Values (Test)", shade = True, color = "b")
sns.kdeplot(yhat_test, label="Predicted Values (Test)", shade = True, color = "r")
plt.legend();
plt.show()
plt.close()

# the close approximatited

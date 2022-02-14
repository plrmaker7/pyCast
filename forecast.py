import pandas as pd # Fantastic for data analysis, especially time series!
import numpy as np # matrix and linear algebra stuff, similar to MATLAB
import errors

from matplotlib import pyplot as plt # plotting
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn import preprocessing as pre

gray_light = '#d4d4d2'
gray_med = '#737373'
red_orange = '#ff3700'

fontsize = 18

elec_and_weather = pd.read_csv('data/17707_usage_and_temp.csv', parse_dates=True, index_col=0)
print ('Start of electricity data: ', min(elec_and_weather.index))
print ('End of electricity data: ', max(elec_and_weather.index))
print(elec_and_weather.head())

fig = plt.figure(figsize=[14,8])

ax1 = elec_and_weather['usage'].plot(label='asdf', color='b', linewidth=0.5)
ax2 = elec_and_weather['temp'].plot(secondary_y = True, label='cew', color=red_orange, linewidth=0.5)
ax1.set_xlabel('')
ax1.set_ylabel('Usage (kilowatt-hours)', fontsize=fontsize, color = 'b')
ax2.set_ylabel('Outdoor Temperature (degrees F)', fontsize=fontsize, color=red_orange)
plt.show()

# fig = plt.figure(figsize=[14,8])
# ax1 = elec_and_weather['usage'].hist(bins=20)
# ax1.set_xlabel('Electricity Usage (kWh)', fontsize=fontsize)
# ax2.set_ylabel('Frequency', fontsize=fontsize)
# plt.show()

## Set weekends and holidays to 1, otherwise 0
elec_and_weather['Atypical_Day'] = np.zeros(len(elec_and_weather['usage']))

# Weekends
elec_and_weather['Atypical_Day'][(elec_and_weather.index.dayofweek==5)|(elec_and_weather.index.dayofweek==6)] = 1

# Holidays
holidays = ['2017-01-01','2017-12-25','2018-01-01','2018-12-25']

for i in range(len(holidays)):
    elec_and_weather['Atypical_Day'][elec_and_weather.index.date==np.datetime64(holidays[i])] = 1

 
print(elec_and_weather.head(3))

# Create new column for each hour of day, assign 1 if index.hour is corresponding hour of column, 0 otherwise

for i in range(0,24):
    elec_and_weather[i] = np.zeros(len(elec_and_weather['usage']))
    elec_and_weather[i][elec_and_weather.index.hour==i] = 1
    
# Example 3am
elec_and_weather[3][:6]

# Add historic usage to each X vector

# Set number of hours prediction is in advance
n_hours_advance = 1

# Set number of historic hours used
n_hours_window = 12


for k in range(n_hours_advance,n_hours_advance+n_hours_window):
    
    elec_and_weather['usage_t-%i'% k] = np.zeros(len(elec_and_weather['usage']))
    #elec_and_weather['tempF_t-%i'% k] = np.zeros(len(elec_and_weather['tempF']))
    #elec_and_weather['hum_t-%i'% k] = np.zeros(len(elec_and_weather['hum']))
    #elec_and_weather['wspdMPH_t-%i'% k] = np.zeros(len(elec_and_weather['wspdMPH']))
    
    
for i in range(n_hours_advance+n_hours_window,len(elec_and_weather['usage'])):
    
    for j in range(n_hours_advance,n_hours_advance+n_hours_window):
        
        elec_and_weather['usage_t-%i'% j][i] = elec_and_weather['usage'][i-j]
        #elec_and_weather['tempF_t-%i'% j][i] = elec_and_weather['tempF'][i-j]
        #elec_and_weather['wspdMPH_t-%i'% j][i] = elec_and_weather['wspdMPH'][i-j]
        #elec_and_weather['hum_t-%i'% j][i] = elec_and_weather['hum'][i-j]

elec_and_weather = elec_and_weather.ix[n_hours_advance+n_hours_window:]
        
#print(elec_and_weather.head(3))

# Define training and testing periods
train_start = '18-march-2017'
train_end = '24-march-2017'
test_start = '25-march-2017'
test_end = '31-march-2017'

# Split up into training and testing sets (still in Pandas dataframes)

X_train_df = elec_and_weather[train_start:train_end]
del X_train_df['usage']

y_train_df = elec_and_weather['usage'][train_start:train_end]

X_test_df = elec_and_weather[test_start:test_end]
del X_test_df['usage']


y_test_df = elec_and_weather['usage'][test_start:test_end]

X_train_df.to_csv('output/training_set.csv')

#N_train = len(X_train_df[0])
#print('Number of observations in the training set: ', N_train)

# Numpy arrays for sklearn
X_train = np.array(X_train_df)
X_test = np.array(X_test_df)
y_train = np.array(y_train_df)
y_test = np.array(y_test_df)

from sklearn import preprocessing as pre
scaler = pre.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

SVR_model = svm.SVR(kernel='rbf',C=100,gamma=.001).fit(X_train_scaled,y_train)
print ('Testing R^2 =', round(SVR_model.score(X_test_scaled,y_test),3))

# Use SVR model to calculate predicted next-hour usage
predict_y_array = SVR_model.predict(X_test_scaled)

# Put it in a Pandas dataframe for ease of use
predict_y = pd.DataFrame(predict_y_array,columns=['usage'])
predict_y.index = X_test_df.index

# Plot the predicted values and actual
import matplotlib.dates as dates

plot_start = test_start
plot_end = test_end

fig = plt.figure(figsize=[14,6])
ax = fig.add_subplot(111)
plt.plot(y_test_df.index,y_test_df,color='k',linewidth=2)
plt.plot(predict_y.index,predict_y,color=red_orange,linewidth=2)
plt.ylabel('Electricity Usage (kWh)', fontsize=fontsize)
plt.ylim([0,2])
plt.legend(['Actual','Predicted'],loc='best', fontsize=fontsize)
ax.xaxis.set_major_formatter(dates.DateFormatter('%b %d'))
fig.savefig('SVM_predict_TS.png')
plt.show()


### Plot daily total kWh over testing period
y_test_barplot_df = pd.DataFrame(y_test_df,columns=['usage'])
y_test_barplot_df['Predicted'] = predict_y['usage']

fig = plt.figure(figsize=[10,4])
ax = fig.add_subplot(111)
y_test_barplot_df.resample('d',how='sum').plot(kind='bar',ax=ax,color=[gray_light,red_orange])
ax.grid(False)
ax.set_ylabel('Total Daily Electricity Usage (kWh)', fontsize=fontsize)
ax.set_xlabel('')
# Pandas/Matplotlib bar graphs convert xaxis to floats, so need a hack to get datetimes back
ax.set_xticklabels([dt.strftime('%b %d') for dt in y_test_df.resample('d',how='sum').index.to_pydatetime()],rotation=0, fontsize=fontsize)
ax.legend(fontsize=fontsize)
plt.show()

fig.savefig('SVM_predict_DailyTotal.png')

#error measurements

N_test = len(y_test_df)
print ('\nNumber of observations (hours) in test dataset: ', N_test, '\n')

RMSE = errors.calc_RMSE(predict_y, y_test_df)
print ('\nRMSE =', round(RMSE, 2), '\n')

MAPE = errors.calc_MAPE(predict_y, y_test_df)
print ('MAPE =', round(MAPE, 2))

MBE = errors.calc_MBE(predict_y, y_test_df)
print ('\nMBE =', round(MBE, 2), '\n')

CV = errors.calc_CV(predict_y, y_test_df)
print ('\nCV =', round(CV, 2), '\n')


fig = plt.figure(figsize=(8,8))
fontsize = 16
plot = plt.plot(y_test_df,predict_y,color=red_orange,marker='.',linewidth=0,markersize=20,alpha=.4)
plot45 = plt.plot([0,2],[0,2],'k')
plt.xlim([0,2])
plt.ylim([0,2])
plt.xlabel('Actual Hourly Elec. Usage (kWh)', fontsize=fontsize)
plt.ylabel('Predicted Hourly Elec. Usage (kWh)', fontsize=fontsize)
plt.show()

fig.savefig('SVM_plot_errors.png')
















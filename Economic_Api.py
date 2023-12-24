import Economic_Analysis
"""
ID: Her veri serisinin benzersiz tanımlayıcısı (örn., TOTALSA, RSXFS).
Realtime Start & End: Veri serisinin analize dahil edildiği tarih aralığı.
Title: Veri serisinin başlığı veya konusu (örn., "Total Vehicle Sales", "Advance Retail Sales: Retail Trade").
Observation Start & End: Veri serisinin gözlemlenmeye başlandığı ve sonlandırıldığı tarih aralığı.
Frequency: Veri serisinin toplanma sıklığı (bu durumda, tüm seriler aylık olarak toplanmış).
Units: Verilerin ölçü birimi (örn., "Millions of Units", "Millions of Dollars").
Units Short: Ölçü birimlerinin kısaltması.
Seasonal Adjustment: Verinin mevsimsel etkilere göre düzeltilip düzeltilmediği (örn., "Seasonally Adjusted", "Not Seasonally Adjusted").
Seasonal Adjustment Short: Mevsimsel düzeltmenin kısaltması.
Last Updated: Veri serisinin en son güncellendiği tarih.
Popularity: Veri serisinin popülerliği veya kullanım sıklığı.
Notes: Veri serisi hakkında ek notlar veya açıklamalar..
"""

df = pd.DataFrame({
    'date': totalsa.index,
    'sales_amount': totalsa.values
})

# Genel Bakış
df.head()
df.info()
df.describe()
print("#################################################")

df.isnull().sum()

df.head(40)

# 'date' datetime çevirelim
df['date'] = pd.to_datetime(df['date'])

#'date' sütununu  index olarak ayarlayalım
df.set_index('date', inplace=True)

# Now, you can resample
y = df['sales_amount'].resample('MS').mean()
y = y.fillna(y.bfill())

train = y['2015-01-01':'2021-12-01']
test = y['2022-01-01':]

def sales_amount(train, test, y_pred, title):
    mae = mean_absolute_error(test, y_pred)
    train["2015":].plot(legend=True, label="TRAIN", title=f"{title}, MAE: {round(mae,2)}")
    test.plot(legend=True, label="TEST", figsize=(6, 4))
    y_pred.plot(legend=True, label="PREDICTION")
    plt.show()

##################################################
# SARIMA(p, d, q): (Seasonal Autoregressive Integrated Moving-Average)
##################################################

model = SARIMAX(train, order=(1, 0, 1), seasonal_order=(0, 0, 0, 12))

sarima_model = model.fit()

y_pred_test = sarima_model.get_forecast(steps=23)

y_pred = y_pred_test.predicted_mean

y_pred = pd.Series(y_pred, index=test.index)

sales_amount(train, test, y_pred, "SARIMA")


############################
# Hyperparameter Optimization (Model Derecelerini Belirleme)
############################

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]


def sarima_optimizer_aic(train, pdq, seasonal_pdq):
    best_aic, best_order, best_seasonal_order = float("inf"), None, None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                sarimax_model = SARIMAX(train, order=param, seasonal_order=param_seasonal)
                results = sarimax_model.fit(disp=0)
                aic = results.aic
                if aic < best_aic:
                    best_aic, best_order, best_seasonal_order = aic, param, param_seasonal
                print('SARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, aic))
            except:
                continue
    print('SARIMA{}x{}12 - AIC:{}'.format(best_order, best_seasonal_order, best_aic))
    return best_order, best_seasonal_order

best_order, best_seasonal_order = sarima_optimizer_aic(train, pdq, seasonal_pdq)



############################
# Final Model
############################

model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order)
sarima_final_model = model.fit(disp=0)

y_pred_test = sarima_final_model.get_forecast(steps=23)

y_pred = y_pred_test.predicted_mean
y_pred = pd.Series(y_pred, index=test.index)

sales_amount(train, test, y_pred, "SARIMA")

##################################################
# MAE'ye Göre SARIMA Optimizasyonu
##################################################

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]


def sarima_optimizer_mae(train, pdq, seasonal_pdq):
    best_mae, best_order, best_seasonal_order = float("inf"), None, None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                model = SARIMAX(train, order=param, seasonal_order=param_seasonal)
                sarima_model = model.fit(disp=0)
                y_pred_test = sarima_model.get_forecast(steps=23)
                y_pred = y_pred_test.predicted_mean
                mae = mean_absolute_error(test, y_pred)
                if mae < best_mae:
                    best_mae, best_order, best_seasonal_order = mae, param, param_seasonal
                print('SARIMA{}x{}12 - MAE:{}'.format(param, param_seasonal, mae))
            except:
                continue
    print('SARIMA{}x{}12 - MAE:{}'.format(best_order, best_seasonal_order, best_mae))
    return best_order, best_seasonal_order

best_order, best_seasonal_order = sarima_optimizer_mae(train, pdq, seasonal_pdq)

model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order)
sarima_final_model = model.fit()

y_pred_test = sarima_final_model.get_forecast(steps=23)
y_pred = y_pred_test.predicted_mean
y_pred = pd.Series(y_pred, index=test.index)

sales_amount(train, test, y_pred, "SARIMA")


############################
# Final Model
############################

model = SARIMAX(y, order=best_order, seasonal_order=best_seasonal_order)
sarima_final_model = model.fit()

feature_predict = sarima_final_model.get_forecast(steps=6)
feature_predict = feature_predict.predicted_mean

#Başka bir yöntemle bakalım
############################
# Veri Seti
############################

y = df['sales_amount'].resample('MS').mean()
y = y.fillna(y.bfill())

train = y['2015-01-01':'2021-12-01']
test = y['2022-01-01':]


##################################################
# Zaman Serisi Yapısal Analizi
##################################################

# Durağanlık Testi (Dickey-Fuller Testi)

def is_stationary(y):

    # "HO: Non-stationary"
    # "H1: Stationary"

    p_value = sm.tsa.stattools.adfuller(y)[1]
    if p_value < 0.05:
        print(F"Result: Stationary (H0: non-stationary, p-value: {round(p_value, 3)})")
    else:
        print(F"Result: Non-Stationary (H0: non-stationary, p-value: {round(p_value, 3)})")

is_stationary(y)

# Zaman Serisi Bileşenleri ve Durağanlık Testi
def ts_decompose(y, model="additive", stationary=False):
    result = seasonal_decompose(y, model=model)
    fig, axes = plt.subplots(4, 1, sharex=True, sharey=False)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    axes[0].set_title("Decomposition for " + model + " model")
    axes[0].plot(y, 'k', label='Original ' + model)
    axes[0].legend(loc='upper left')

    axes[1].plot(result.trend, label='Trend')
    axes[1].legend(loc='upper left')

    axes[2].plot(result.seasonal, 'g', label='Seasonality & Mean: ' + str(round(result.seasonal.mean(), 4)))
    axes[2].legend(loc='upper left')

    axes[3].plot(result.resid, 'r', label='Residuals & Mean: ' + str(round(result.resid.mean(), 4)))
    axes[3].legend(loc='upper left')
    plt.show(block=True)

    if stationary:
        is_stationary(y)

ts_decompose(y, stationary=True)


##################################################
# Single Exponential Smoothing
##################################################

# SES = Level

ses_model = SimpleExpSmoothing(train).fit(smoothing_level=0.5)

y_pred = ses_model.forecast(23)

mean_absolute_error(test, y_pred)

train.plot(title="Single Exponential Smoothing")
test.plot()
y_pred.plot()
plt.show()


train["2020-01-01":].plot(title="Single Exponential Smoothing")
test.plot()
y_pred.plot()
plt.show()


def sales_amount_ses(train, test, y_pred, title):
    mae = mean_absolute_error(test, y_pred)
    train["2020-01-01":].plot(legend=True, label="TRAIN", title=f"{title}, MAE: {round(mae,2)}")
    test.plot(legend=True, label="TEST", figsize=(6, 4))
    y_pred.plot(legend=True, label="PREDICTION")
    plt.show()

sales_amount_ses(train, test, y_pred, "Single Exponential Smoothing")

ses_model.params

############################
# Hyperparameter Optimization
############################

def ses_optimizer(train, alphas, step=23):

    best_alpha, best_mae = None, float("inf")

    for alpha in alphas:
        ses_model = SimpleExpSmoothing(train).fit(smoothing_level=alpha)
        y_pred = ses_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)

        if mae < best_mae:
            best_alpha, best_mae = alpha, mae

        print("alpha:", round(alpha, 2), "mae:", round(mae, 4))
    print("best_alpha:", round(best_alpha, 2), "best_mae:", round(best_mae, 4))
    return best_alpha, best_mae

alphas = np.arange(0.8, 1, 0.01)

# yt_sapka = a * yt-1 + (1-a)* (yt_-1)_sapka

ses_optimizer(train, alphas)

best_alpha, best_mae = ses_optimizer(train, alphas)

############################
# Final SES Model
############################

ses_model = SimpleExpSmoothing(train).fit(smoothing_level=best_alpha)
y_pred = ses_model.forecast(23)

sales_amount_ses(train, test, y_pred, "Single Exponential Smoothing")

##################################################
# Triple Exponential Smoothing (Holt-Winters)
##################################################


tes_model = ExponentialSmoothing(train,
                                 trend="add",
                                 seasonal="add",
                                 seasonal_periods=12).fit(smoothing_level=0.5,
                                                          smoothing_slope=0.5,
                                                          smoothing_seasonal=0.5)

y_pred = tes_model.forecast(23)
sales_amount_ses(train, test, y_pred, "Triple Exponential Smoothing")

############################
# Hyperparameter Optimization
############################

alphas = betas = gammas = np.arange(0.20, 1, 0.10)

abg = list(itertools.product(alphas, betas, gammas))


def tes_optimizer(train, abg, step=23):
    best_alpha, best_beta, best_gamma, best_mae = None, None, None, float("inf")
    for comb in abg:
        tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=comb[0], smoothing_slope=comb[1], smoothing_seasonal=comb[2])
        y_pred = tes_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)
        if mae < best_mae:
            best_alpha, best_beta, best_gamma, best_mae = comb[0], comb[1], comb[2], mae
        print([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mae, 2)])

    print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_gamma:", round(best_gamma, 2),
          "best_mae:", round(best_mae, 4))

    return best_alpha, best_beta, best_gamma, best_mae

best_alpha, best_beta, best_gamma, best_mae = tes_optimizer(train, abg)


############################
# Final TES Model
############################

final_tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=best_alpha, smoothing_trend=best_beta, smoothing_seasonal=best_gamma)

y_pred = final_tes_model.forecast(23)

sales_amount_ses(train, test, y_pred, "Triple Exponential Smoothing")



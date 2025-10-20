# Import relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, make_scorer, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, PowerTransformer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_classif
import optuna
import shap

#%%

random_seed = 42
metric = 'f1'

# Import data 
#data_raw = pd.read_csv('/Users/tracie/Desktop/academic/goethe/2-SoSe25/Seminar-AADS/turkishCF.csv', sep=';')
data_raw = pd.read_csv('/Users/marc/Dropbox/6. Semester/Advanced Applied Data Science/Data/turkishCF.csv', sep=';')

data = data_raw.copy()

print(data.duplicated().sum())
# No duplicated rows

# Rename feature names from turkish to english
data.rename(columns = {
    'platform_adi': 'platform',
    'kitle_fonlamasi_turu': 'crowdfunding_type',
    'kategori': 'category',
    'fon_sekli': 'funding_method',
    'proje_adi': 'project_name',
    'proje_sahibi': 'project_owner',
    'proje_sahibi_cinsiyet': 'project_owner_gender',
    'kac_proje_destekledi': 'number_of_backed_projects',
    'kac_projeye_abone': 'number_of_subscribed_projects',
    'kac_projenin_sahibi': 'number_of_projects_owners',
    'kac_proje_takiminda': 'number_teams_project_owner',
    'konum': 'location',
    'bolge': 'region',
    'yil': 'year',
    'proje_baslama_tarihi': 'start_date',
    'proje_bitis_tarihi': 'end_date',
    'gun_sayisi': 'duration',
    'tanitim_videosu': 'promotion_video',
    'video_uzunlugu': 'promotion_video_length',
    'gorsel_sayisi': 'number_of_images',
    'sss': 'faq',
    'guncellemeler': 'updates',
    'yorumlar': 'comments',
    'destekci_sayisi': 'number_of_supporters',
    'odul_sayisi': 'number_of_awards',
    'ekip_kisi_sayisi': 'number_of_project_team_members',
    'web_sitesi': 'website',
    'sosyal_medya': 'social_media',
    'sm_sayisi': 'number_of_social_media_accounts',
    'sm_takipci': 'social_media_followers',
    'etiket_sayisi': 'number_of_tags',
    'icerik_kelime_sayisi': 'number_of_words_in_description',
    'proje_aciklamasi': 'project_description',
    'hedef_miktari': 'project_target_amount',
    'toplanan_tutar': 'project_amount_collected',
    'destek_orani': 'percentage_amount_achieved',
    'basari_durumu': 'success'}, inplace = True)

print(data.head())
print(data.info())
print(data.isna().sum().sort_values(ascending=False))
print(data.describe(include='all').T)

#%% Categorical features

# Check distribution of binary features
categorical_cols = data.select_dtypes(include='object').columns
print(data[categorical_cols].nunique().sort_values(ascending=True))

binary_cols = data[categorical_cols].nunique()
binary_cols = binary_cols[binary_cols == 2].index.tolist()
binary_cols.remove('success')

# Crowdfunding type
print(data['crowdfunding_type'].value_counts())
# Delete because of no information 
data = data.drop(columns=['crowdfunding_type'])
binary_cols.remove('crowdfunding_type')

# Funding method
print(data['funding_method'].value_counts())

# Rename from turkish to english
data['funding_method'] = data['funding_method'].map({
    'ya hep ya hiç': 'all_or_nothing',   
    'hepsi kalsın': 'keep_it_all'})

data['funding_method_all_or_nothing'] = data['funding_method'].map({'all_or_nothing': 1, 'keep_it_all': 0})
data = data.drop(columns=['funding_method'])
binary_cols.append('funding_method_all_or_nothing')
binary_cols.remove('funding_method')

# Promotion video
print(data['promotion_video'].value_counts())
# Keep and translate
data['promotion_video'] = data['promotion_video'].map({'var': 1,'yok': 0})

# Website
print(data['website'].value_counts())
# Keep and translate
data['website'] = data['website'].map({'var': 1,'yok': 0})

# Social media
print(data['social_media'].value_counts())
# Keep and translate
data['social_media'] = data['social_media'].map({'var': 1,'yok': 0})

# Target 
print(data['success'].value_counts())
# We have an imbalanced data set -> need to address this 
data['success'] = data['success'].map({'başarılı': 1, 'başarısız': 0})

# New check
categorical_cols = data.select_dtypes(include='object').columns
print(data[categorical_cols].nunique().sort_values(ascending=True))

print(data['region'].value_counts())
print(data['location'].value_counts())
# Delete location because much more values, that leads to much more dummies
# that have perfect multicollinearity with region 

# Drop these columns because there are irrelevant or known after start
data = data.drop(columns=[
    'project_description', # irrelevant
    'project_name', # irrelevant
    'project_owner', # irrelevant
    'end_date', # we have project duration 
    'start_date', # we have project duration
    'percentage_amount_achieved', # leakage 
    'location']) # multicollinearity

# Category
print(data['category'].value_counts())
# Massive imbalance between instances, have to create larger groups that 
# models can work with information

# Translate from turkish to english
data['category'] = data['category'].map({
    'film-video-fotoğraf': 'film-video-photography',
    'teknoloji': 'technology',
    'kültür-sanat': 'culture-art',
    'eğitim': 'education',
    'diğer': 'other',
    'çevre': 'environment',
    'müzik': 'music',
    'sağlık-güzellik': 'health-beauty',
    'tasarım': 'design',
    'yayıncılık': 'publishing',
    'gıda-yeme-içme': 'food-eating-drinking',
    'spor': 'sports',
    'hayvanlar': 'animals',
    'moda': 'fashion',
    'sosyal sorumluluk': 'social_responsibility',
    'dans-performans': 'dance-performance',
    'turizm': 'tourism'})

# Group to reduce outliers
data['category'] = data['category'].map({
    'film-video-photography': 'creative',
    'culture-art': 'creative', 
    'design': 'creative',
    'music': 'creative',
    'fashion': 'creative',
    'dance-performance': 'creative',
    'publishing': 'creative', 
    'technology': 'technology',
    'other': 'other', 
    'education': 'social',
    'social_responsibility': 'social',
    'environment': 'social',
    'food-eating-drinking': 'lifestyle',
    'sports': 'lifestyle',
    'animals': 'lifestyle',
    'tourism': 'lifestyle',
    'health-beauty': 'lifestyle'})

print(data['category'].value_counts())
# Much better distribution 

# Gender 
print(data['project_owner_gender'].value_counts())
# Good and translate 
data['project_owner_gender'] = data['project_owner_gender'].map({
    'belirsiz': 'unknown',
    'kadın': 'female',
    'erkek':'male'})

# Region
print(data['region'].value_counts())
# Imbalanced distribution ->  group to other

data['region'] = data['region'].map({
    'marmara': 'marmara',
    'belirsiz': 'belirsiz',
    'iç anadolu': 'iç_anadolu',
    'ege': 'ege',
    'akdeniz': 'akdeniz',
    'genel': 'other',
    'karadeniz': 'other',
    'güneydoğu': 'other',
    'doğu': 'other'})

print(data['region'].value_counts())
# Better distribution 

# Platform 
print(data['platform'].value_counts())
# Imbalance, group to other

data['platform'] = data['platform'].map({
    'fongogo': 'fongogo',
    'crowdfon': 'crowdfon',
    'fonbulucu': 'fonbulucu',
    'arıkovanı': 'other',
    'buluşum': 'other',
    'ideanest': 'other'})

print(data['platform'].value_counts())

#%% Numerical data 
numerical_cols = data.select_dtypes(include=['number']).columns
print((data[numerical_cols] != 0).sum().sort_values(ascending=True))

# Check correlations 
correlations = data[numerical_cols].corrwith(data['success']).abs().sort_values(ascending=False)
print(correlations)
# No high correlations 

print(data.groupby('year')['success'].mean())
# Only year 2011 was very successfull -> imbalance and also 
# year has a very very low correlation with success and might result in many 
# dummies or implies a wrong order, therefore delete it 

data = data.drop(columns=[
    'number_of_subscribed_projects', # low number of values != 0
    'number_teams_project_owner', # low number of values != 0
    'id', # irrelevant
    'year', # correlation
    'faq', # leakage 
    'comments', # leakage
    'updates', # leakage 
    'number_of_supporters', # leakage
    'project_amount_collected', # leakage
    'social_media_followers' ]) # leakage

#%% Handle missing values
data.isna().sum().sort_values(ascending=False)

# Drop the row where the missing value in region is 
data.drop(data[pd.isnull(data.region)].index, 
          inplace=True) 

#%% Feature engineering 

# 80% training and 20% test data to have enough information in the train data 
# Split into train and test data 

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    data.drop(columns='success'), 
    data['success'], 
    test_size=0.2, 
    stratify=data['success'], 
    random_state=random_seed)
    
# Check distribution of target
print(data['success'].value_counts(normalize=True))
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))
# Distribution is the same

X_train = X_train_raw.copy()
X_test = X_test_raw.copy()

#%% Preprocesing 
numerical_cols = X_train.select_dtypes(include=['number']).columns
print(numerical_cols)

numerical_cols = pd.Index(numerical_cols).difference(binary_cols).tolist() 
# Do not scale binary features 

skewness_features = X_train[numerical_cols].skew().sort_values(ascending=False)
print(skewness_features)
# heavy skewed features
skewed_features = skewness_features[abs(skewness_features) > 1].index.tolist()

(X_train[skewed_features] == 0).sum()
# There are null values -> simple log does not work

# Adjust skewness 
pt = PowerTransformer(method='yeo-johnson')
X_train[skewed_features] = pt.fit_transform(X_train[skewed_features])
X_test[skewed_features] = pt.transform(X_test[skewed_features])

print(X_train[numerical_cols].skew().sort_values(ascending=False))
# Skewness is much better 

# Scale numerical continous features 
scaler = StandardScaler()

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train[numerical_cols]),
                              columns=numerical_cols,
                              index=X_train.index)

X_test_scaled = pd.DataFrame(scaler.transform(X_test[numerical_cols]), 
                             columns=numerical_cols,
                             index=X_test.index)

X_train_scaled.hist(bins=100, figsize=(20,15))
plt.show()

X_test_scaled.hist(bins=100, figsize=(20,15))       
plt.show()

# Check multicollinearity
vif = pd.DataFrame()                    
vif['feature'] = X_train_scaled.columns
vif['vif'] = [variance_inflation_factor(X_train_scaled.values, i)
              for i in range(X_train_scaled.shape[1])]

print(vif)
# No multicollinearity

#%% Dummy encoding 
categorical_cols = X_train.select_dtypes(include='object').columns

X_train_dummies = pd.get_dummies(X_train[categorical_cols],
                                 drop_first='True').astype(int)


X_test_dummies = pd.get_dummies(X_test[categorical_cols],
                                 drop_first='True').astype(int)

X_test_dummies = X_test_dummies.reindex(columns=X_train_dummies.columns,
                                        fill_value=0)

#%% Get final feature data after preprocessing
X_train_binary = X_train_raw[binary_cols]
X_test_binary = X_test_raw[binary_cols]

X_train = pd.concat([X_train_scaled, X_train_dummies, X_train_binary], 
                    axis=1)

X_test = pd.concat([X_test_scaled, X_test_dummies, X_test_binary], 
                    axis=1)

X_train = X_train.reindex(sorted(X_train.columns),
                          axis=1)

X_test = X_test.reindex(sorted(X_test.columns),
                          axis=1)

print(data.duplicated().mean())
print(X_train.duplicated().mean())
print(X_test.duplicated().mean())

# Before data handling there were no duplicates, 
# delete the duplicates in training data because it can lead 
# to overfitting besides in the raw data are no duplicates 
duplicated_rows =  X_train.duplicated()
X_train = X_train[~duplicated_rows]
y_train = y_train[~duplicated_rows]

#%% Checking correlation with target variable
print(X_train[numerical_cols].corrwith(y_train))

# Checking mutual information with target variable
mi = mutual_info_classif(X_train[numerical_cols], y_train)
mi_series = pd.Series(mi, index=numerical_cols)
print(mi_series.sort_values(ascending=False))

#%% Model builder
def model_build(ml_type, X_train=X_train, X_test=X_test, pred_score=metric): 
    
    random_seed = 42
    
    # Function to get the model
    # A function because we need this twice, one time in the tuning and then to fit 
    # Set class_weight = 'balanced' within the models where this is possible to address
    # imbalance and also a seed to make results reproductable
    def get_model(ml_type, params):
        
        random_seed = 42
        
        if ml_type == 'logreg': 
            return LogisticRegression(**params, 
                                      class_weight='balanced', 
                                      random_state=random_seed)
        
        if ml_type == 'svm': 
            return SVC(**params, 
                       probability=True,
                       class_weight='balanced', 
                       random_state=random_seed)
        
        if ml_type == 'dt': 
            return DecisionTreeClassifier(**params, 
                                          class_weight='balanced', 
                                          random_state=random_seed)
        
        if ml_type == 'rf': 
            return RandomForestClassifier(**params, 
                                          class_weight='balanced', 
                                          random_state=random_seed)
        
        if ml_type == 'xgb': 
            return XGBClassifier(**params, 
                                 random_state=random_seed)
        
    
    # Function for hyperparameter tuning 
    def objective(trial, ml_type, X_train, y_train, folds): 
        
        # Set parameters search spaces for each model
        if ml_type == 'logreg':
            params = {
                'C': trial.suggest_float('C', 0.01, 1),  
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),  
                'solver': trial.suggest_categorical('solver', ['liblinear']), 
                'max_iter': trial.suggest_int('max_iter', 1000, 5000)}
            
        if ml_type == 'svm': 
            params = {
                'C': trial.suggest_float('C', 0.01, 1),  
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'degree': trial.suggest_int('degree', 2, 4)}
            
            
            
        if ml_type == 'dt': 
            params = {
                'criterion': trial.suggest_categorical('criterion', ['entropy', 'gini']),
                'max_depth': trial.suggest_int('max_depth', 3, 5),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 20, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 30, 80)}
            
        if ml_type == 'rf': 
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 350, 500),
                'max_depth': trial.suggest_int('max_depth', 8, 12),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 8, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 15, 25),
                'max_features': trial.suggest_float('max_features', 0.3, 0.5),
                'max_samples': trial.suggest_float('max_samples', 0.8, 0.9),
                'bootstrap': True}
            
        if ml_type == 'xgb': 
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 150),
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.15),
                'max_depth': trial.suggest_int('max_depth', 3, 5),
                'min_child_weight': trial.suggest_int('min_child_weight', 10, 25),
                'subsample': trial.suggest_float('subsample', 0.5, 0.7),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.7),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 5, 20)}


        model = get_model(ml_type=ml_type, 
                          params=params)
        
        # Get F1 values of each fold
        score = cross_val_score(estimator=model,
                                X=X_train,
                                y=y_train,
                                cv=folds,
                                scoring='f1', 
                                n_jobs=-1)
        
        # Maximize the mean F1-score from the validation folds
        return score.mean()
    
    
    # Split train set into five splits means there is every time 80% training
    # and 20% validation, save the folds to have them in a later calculation 
    # for the mean score of the folds 
    k_folds = StratifiedKFold(n_splits=5,
                              shuffle=True, 
                              random_state=random_seed)
    folds = list(k_folds.split(X_train, y_train))
    
    # Optuna hyperparameter tuning
    sampler = optuna.samplers.TPESampler(seed=random_seed)
    
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    
    study = optuna.create_study(direction='maximize',
                                sampler=sampler, 
                                pruner=pruner)
    
    # High number of trials to find optimal values 
    study.optimize(lambda trial: objective(trial=trial, 
                                          ml_type=ml_type, 
                                          X_train=X_train,
                                          y_train=y_train, 
                                          folds=folds), 
                  n_trials=200, 
                  timeout=300)
    
    # Get optimal values
    optimal_params = study.best_params
    
    # If model is XGB set extra parameter to address imbalance 
    if ml_type == 'xgb':
        scale_pos_weight_value = (y_train == 0).sum() / (y_train == 1).sum()
        optimal_params['scale_pos_weight'] = scale_pos_weight_value
        
    # Tune model with best parameters
    tuned_model = get_model(ml_type = ml_type,
                            params=optimal_params)
    
    tuned_model.fit(X_train, y_train)
    
    y_pred_train = tuned_model.predict(X_train)
    train_score = f1_score(y_true=y_train,
                           y_pred=y_pred_train)
    
    
    cv_scores = cross_val_score(estimator=tuned_model,
                                X=X_train,
                                y=y_train,
                                cv=folds,
                                scoring=pred_score, 
                                n_jobs=-1)
    
    return tuned_model, cv_scores, folds, train_score

#%% Build predict funtion 
def model_pred(tuned_model, X_test=X_test): 
    
    y_pred = tuned_model.predict(X_test)
    y_prob = tuned_model.predict_proba(X_test)[:, 1]
    
    return y_pred, y_prob

#%%
def evaluate_model(fitted_model, y_test=y_test): 
    
    y_pred = fitted_model[0]
    y_prob = fitted_model[1]
    
    tn, fp, fn, tp = confusion_matrix(y_true=y_test,
                                      y_pred=y_pred).ravel()
    
    metrics = pd.DataFrame.from_dict(
        {'Accurracy': (tp+tn)/(tp+tn+fp+fn),
         'Specificity': tn/(tn+fp),
         'Recall/Sensitivity': tp/(tp+fn), 
         'Precision': tp/(tp+fp), 
         'F1': (2*tp)/(2*tp+fp+fn), 
         'AUC': roc_auc_score(y_test, y_prob)},
        orient='index', 
        columns=['value'])
    
    return metrics

#%% Build models and predict
logreg_model = model_build(ml_type='logreg')
dt_model = model_build(ml_type='dt')
rf_model = model_build(ml_type='rf')
xgb_model = model_build(ml_type='xgb')


svm_model = model_build(ml_type='svm')
logreg_pred = model_pred(tuned_model=logreg_model[0])
svm_pred = model_pred(tuned_model=svm_model[0])
dt_pred = model_pred(tuned_model=dt_model[0])
rf_pred = model_pred(tuned_model=rf_model[0])
xgb_pred = model_pred(tuned_model=xgb_model[0])
    
#%% Evaluations 
evaluation_logreg = evaluate_model(fitted_model=logreg_pred)
evaluation_svm = evaluate_model(fitted_model=svm_pred)    
evaluation_dt = evaluate_model(fitted_model=dt_pred)
evaluation_rf = evaluate_model(fitted_model=rf_pred)     
evaluation_xgb = evaluate_model(fitted_model=xgb_pred)  

evaluation_logreg.columns = ['LOGREG']
evaluation_svm.columns = ['SVM']
evaluation_dt.columns = ['DT']
evaluation_rf.columns = ['RF']
evaluation_xgb.columns = ['XGB']

evaluation_models = pd.concat([evaluation_logreg, 
                               evaluation_svm, 
                               evaluation_dt, 
                               evaluation_rf, 
                               evaluation_xgb], 
                              axis=1)

evaluation_models.iloc[0:] = evaluation_models.iloc[0:].round(2)

# Export to Excel
evaluation_models.to_excel('evaluation_models.xlsx', index=True)                        
                                                              
#%% Over-/ Underfitting check

score = 'F1'

logreg_model[3]
np.mean(logreg_model[1])
np.std(logreg_model[1])
evaluation_logreg.loc[score, 'LOGREG']
logreg_model[0].get_params()

svm_model[3]
np.mean(svm_model[1])
np.std(svm_model[1])
evaluation_svm.loc[score, 'SVM']
svm_model[0].get_params()

dt_model[3]
np.mean(dt_model[1])
np.std(dt_model[1])
evaluation_dt.loc[score, 'DT']
dt_model[0].get_params()

rf_model[3]
np.mean(rf_model[1])
np.std(rf_model[1])
evaluation_rf.loc[score, 'RF']
rf_model[0].get_params()

xgb_model[3]
np.mean(xgb_model[1])
np.std(xgb_model[1])
evaluation_xgb.loc[score, 'XGB']
xgb_model[0].get_params()
         
#%% ROC curve
model_dict = {
    'Logistic Regression': logreg_pred[1],
    'Support Vector Machine': svm_pred[1],
    'Decision Tree': dt_pred[1],
    'Random Forest': rf_pred[1],
    'XGBoost': xgb_pred[1]}

color_dict = {
    'Logistic Regression': 'orange',
    'Support Vector Machine': 'green',
    'Decision Tree': 'yellow',
    'Random Forest': 'red', 
    'XGBoost': 'darkblue'}

linestyle_dict = {
    'Logistic Regression': '-',
    'Support Vector Machine': '-',
    'Decision Tree': '-',
    'Random Forest': '-', 
    'XGBoost': '-'}

fig, ax = plt.subplots(figsize=(14, 8))

for name, y_prob in model_dict.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(
        fpr, tpr,
        label=f'{name} (AUC = {auc:.2f})',
        color=color_dict.get(name),              
        linestyle=linestyle_dict.get(name, '-'))
    
ax.plot([0, 1], [0, 1], linestyle='--', color='black')
ax.set_xlabel('False Positive Rate (FPR)', 
              fontsize=18)
ax.set_ylabel('True Positive Rate (TPR)', 
              fontsize=18)
ax.tick_params(axis='both', 
               labelsize=14)
ax.set_title('Comparison of ROC Curves', 
             fontsize=20)
ax.legend(loc='lower right', 
          fontsize=14)
ax.grid(False)

plt.savefig('roc_curve.png',
            dpi=500,
            bbox_inches='tight')
plt.show()

#%% Feature imporatnce
feature_importance = pd.Series(xgb_model[0].feature_importances_, 
                               index=X_train.columns).sort_values(ascending=False)
print(feature_importance)

#%% SHAP
explainer = shap.TreeExplainer(xgb_model[0])
shap_values = explainer(X_test)

shap.plots.beeswarm(shap_values, 
                    max_display=20, 
                    show = False)
    
fig = plt.gcf()
fig.set_size_inches(10, 6)

plt.savefig('shap_importance.png',
            dpi=500, 
            bbox_inches='tight')

plt.show()

#%% Statistical test 

# Less values 
success = X_test[y_test == 1]['duration']
fail = X_test[y_test == 0]['duration']    
t_value, p_value = ttest_ind(success, fail, equal_var=False, alternative='less')
print(t_value, p_value )
        
success = X_test[y_test == 1]['project_target_amount']
fail = X_test[y_test == 0]['project_target_amount']    
t_value, p_value = ttest_ind(success, fail, equal_var=False, alternative='less')
print(t_value, p_value )                
                
                
# Greater values 
success = X_test[y_test == 1]['promotion_video_length']
fail = X_test[y_test == 0]['promotion_video_length']    
t_value, p_value = ttest_ind(success, fail, equal_var=False, alternative='greater')
print(t_value, p_value )               

#%% Misclassifications 
pred = xgb_pred[0]              
                
# Searching for differences in shap values of correct and falsely predicted
misclassifications_test = (y_test!=pred).values

error_shap_comparison = pd.DataFrame({
    'feature': X_test.columns, 
    'correct': np.mean(shap_values[~misclassifications_test].values, axis=0), 
    'misclassified': np.mean(shap_values[misclassifications_test].values, axis=0)})

error_shap_comparison['diff'] = abs(error_shap_comparison['correct']) - abs(error_shap_comparison['misclassified'])
error_shap_comparison = error_shap_comparison.sort_values('diff', ascending=False)
print(error_shap_comparison)

#%% Confusion matrix and learning curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Confusion Matrix 
confusion_mat_data = confusion_matrix(y_true=y_test, 
                                      y_pred=xgb_pred[0])

sns.heatmap(confusion_mat_data, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Unsuccessful', 'Successful'], 
            yticklabels=['Unsuccessful', 'Successful'], 
            cbar=False,
            annot_kws={"size": 18},
            ax=ax1)

ax1.set_xlabel('Prediction', fontsize=20)
ax1.set_ylabel('True Value', fontsize=20)
ax1.tick_params(axis='both', which='major', labelsize=16)
ax1.set_title('Confusion Matrix XGBoost', fontsize=22)

model = xgb_model[0] 
folds = xgb_model[2]

# F1-Score Learning Curve
train_sizes_f1, train_scores_f1, validation_scores_f1 = learning_curve(
    estimator=model,
    X=X_train,
    y=y_train,
    cv=folds,
    scoring=make_scorer(f1_score),
    train_sizes=np.linspace(0.1, 1.0, 20),
    n_jobs=-1,
    random_state=random_seed)

validation_f1_mean = validation_scores_f1.mean(axis=1)
validation_f1_std = validation_scores_f1.std(axis=1)

# Recall Learning Curve
train_sizes_recall, train_scores_recall, validation_scores_recall = learning_curve(
    estimator=model,
    X=X_train,
    y=y_train,
    cv=folds,
    scoring=make_scorer(recall_score),
    train_sizes=np.linspace(0.1, 1.0, 20),
    n_jobs=-1,
    random_state=random_seed)

validation_recall_mean = validation_scores_recall.mean(axis=1)
validation_recall_std = validation_scores_recall.std(axis=1)

# Precision Learning Curve
train_sizes_precision, train_scores_precision, validation_scores_precision = learning_curve(
    estimator=model,
    X=X_train,
    y=y_train,
    cv=folds,
    scoring=make_scorer(precision_score),
    train_sizes=np.linspace(0.1, 1.0, 20),
    n_jobs=-1,
    random_state=random_seed)

validation_precision_mean = validation_scores_precision.mean(axis=1)
validation_precision_std = validation_scores_precision.std(axis=1)

# Learning curves
# F1
ax2.plot(train_sizes_f1, 
         validation_f1_mean, 
         color='blue', 
         label='F1-Score', 
         linewidth=2)

ax2.fill_between(train_sizes_f1,
                 validation_f1_mean - validation_f1_std,
                 validation_f1_mean + validation_f1_std,
                 color='blue',
                 alpha=0.2)

# Recall
ax2.plot(train_sizes_recall, 
         validation_recall_mean, 
         color='red', 
         label='Recall', 
         linewidth=2)

ax2.fill_between(train_sizes_recall,
                 validation_recall_mean - validation_recall_std,
                 validation_recall_mean + validation_recall_std,
                 color='red',
                 alpha=0.2)

# Precision
ax2.plot(train_sizes_precision, 
         validation_precision_mean, 
         color='green', 
         label='Precision', 
         linewidth=2)

ax2.fill_between(train_sizes_precision,
                 validation_precision_mean - validation_precision_std,
                 validation_precision_mean + validation_precision_std,
                 color='green',
                 alpha=0.2)

ax2.set_title('Learning Curves XGBoost', fontsize=22)
ax2.set_xlabel('Training Set Size', fontsize=20)
ax2.set_ylabel('Validation Score', fontsize=20)
ax2.tick_params(axis='both', which='major', labelsize=16)
ax2.grid(True)
ax2.legend(loc='lower right', fontsize=18)
ax2.set_ylim(0.3, 0.9)  

plt.tight_layout()
plt.savefig('confusion_matrix_and_learning_curve.png',
            dpi=500, 
            bbox_inches='tight')
plt.show()

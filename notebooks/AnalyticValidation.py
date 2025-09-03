# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ### Validating analytic workflows
#
# In this notebook we will show how to use simulations and create synthetic data to validate an analysis workflow.
#
#

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingGridSearchCV
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, SGDRegressor
from nilearn.maskers import NiftiMasker
from nilearn.image import smooth_img, index_img, resample_to_img
from nilearn.glm import threshold_stats_img
from nilearn.datasets import (
    fetch_atlas_difumo,
    load_mni152_brain_mask,
    load_mni152_template
)
import nilearn.plotting
import nibabel as nib
from ipywidgets import IntProgress
from IPython.display import display


# %% [markdown]
# ### Generating synthetic neuroimaging data
#
# It can be useful to benchmark analyses using synthetic data before ever applying them to real data.  In the best case, one would actually pre-register the analyses developed on synthetic data prior applying them to real data.
#
# Let's say that we wanted to test a new "biomarker" for brain age using task fMRI.  We could start by generating some random data (using the MNI305 mask as our starting point) and then apply our new technique to that, either using a completely random value for age, or creating age to have a specific relationship to the synthetic brain data.
#
# First, we can set up a NiftiMasker based on the MNI305 mask:

# %%
mnimask = load_mni152_brain_mask(resolution=3.)
mnitemplate = load_mni152_template(resolution=3.)

# %%
disp = nilearn.plotting.plot_anat(mnitemplate)
disp.add_overlay(mnimask, alpha=0.5)

# %%
masker = NiftiMasker()
maskdata = masker.fit_transform(mnimask)
maskdata.shape


# %% [markdown]
# We then generate random Gaussian data for each subject (let's say 100 subjects).
#

# %%
def generate_noise_img(maskimg, nsubs=100, seed=None, fwhm=8):
    masker = NiftiMasker()
    maskdata = masker.fit_transform(maskimg)
    
    rng = np.random.default_rng(seed)
    syndata = rng.normal(size=(nsubs, maskdata.shape[1]))
    synimg = masker.inverse_transform(syndata)
    if fwhm is not None:
        synimg = smooth_img(synimg, 8)
    return(synimg, masker)

synimg, masker = generate_noise_img(mnimask)

# %%
nilearn.plotting.plot_img(index_img(synimg, 1), cmap='gray')


# %% [markdown]
# ### Testing our biomarker
#
# Now let's test our biomarker.  Our innovation is to implement a feature selection procedure called Recursive Feature Elimination (RFE) to reduce the dimensionality of the data for the prediction model.  

# %%

def train_brainage_model(img, y, mask, nfeatures=50, n_splits=20, 
                         seed=None, shuffle_y=False, fsel='outside'):
    assert fsel is None or fsel in ('inside', 'outside')
    assert y.shape[0] == img.shape[-1]
    masker = NiftiMasker(mask)
    maskdata = masker.fit_transform(img)
    
    rng = np.random.RandomState(seed)
    if shuffle_y:
        y = y.copy()
        np.random.shuffle(y)

    estimator = LinearSVR()
    if fsel == 'outside':
        #  perform recursive feature elimination outside of CV
        selector = SelectKBest(f_regression, k=nfeatures)
        X = selector.fit_transform(maskdata, y)
    else:
        X = maskdata

    scores = []

    clf = LinearSVR()

    for i in range(n_splits):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=rng)
        if fsel == 'inside':
            # perform RFE inside CV loop
            selector = SelectKBest(f_regression, k=nfeatures)
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scores.append(r2_score(y_test, y_pred))
    return np.mean(scores)
    
rng = np.random.RandomState()
nsubs = 100
age = rng.normal(loc=40, scale=10, size=nsubs)

train_brainage_model(synimg, age, mnimask)


# %% [markdown]
# If we didn't know that there was no true relationship between the brain data and age, we wouldn't realize that this was weird.  We would likely want to know whether the observed $R^2$ is sufficiently larger than what one would expect under the null hypothesis of no predictive relationship (i.e. $R^2 = 0$).  To determine this, we can run the model repeatedly while shuffling the order of the y variable, which essentially breaks the relationship on average between the X and y variables.  In principle this should have a mean of zero (or less, given that crossvalidation using regression with smallish samples can often lead to negative $R^2$ values when the model fits the data worse than the mean; see [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7250718/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7250718/) for more).

# %%
scores = []
nsimruns = 50

progbar = IntProgress(min=0, max=nsimruns) # instantiate the bar
display(progbar) # display the bar
for i in range(nsimruns):
    progbar.value += 1
    scores.append(train_brainage_model(synimg, age, mnimask, shuffle_y=True))
print(np.mean(scores))
plt.hist(scores)

# %% [markdown]
# Wait, what?  The $R^2$ scores are all well above zero even when we shuffle the data, which means that there must be some kind of leaking of information from the test data into the training. In this case, it's due to the fact that our feature selection procedure was applied on the entire dataset prior to crossvalidation, rather than within the crossvalidation folks. If we move the feature selection step inside the crossvalidation loop, we should see that the predictive accuracy decreases.

# %%
train_brainage_model(synimg, age, mnimask, fsel='inside')


# %% [markdown]
# This example shows how injecting synthetic data into our analysis procedures can help ensure that they are not providing us with biased results.

# %% [markdown]
# ### Generating realistic signals for testing
#
# In the previous example we tested whether our procedure effectively controls for false positive results.  In other cases, we may also want to know the degree to which our analysis procedure can accurately detect signals when they exist (often known as "parameter recovery").  To assess this, we need to generate synthetic data that contains both realistic signal and realistic noise, and then apply our procedure to assess its effectiveness.
#
# Here we will inject some true signal relating activity to age into one of the regions of the brain, and then assess how well different methods can detect the signal.

# %%
difumo = fetch_atlas_difumo(resolution_mm=2, legacy_format=False)
difumo_maps = nib.load(difumo['maps'])
# use component 63, which includes bilateral caudate
# first resample the difumo image to the same space as the data (which is the 3mm mni space)
roi_img = resample_to_img(index_img(difumo_maps, 63), mnitemplate)
disp = nilearn.plotting.plot_stat_map(roi_img, bg_img=mnitemplate)


# %% [markdown]
# Now let's generate some data by injecting signal into those masked voxels relating their activity to age.
#
# Generate Gaussian noise, and then add signal within ROI voxels
#
#

# %%
def generate_synthetic_age_data(beta=10):
    synimg, masker = generate_noise_img(mnimask)
    syndata = masker.transform(synimg)
    roidata = np.repeat(masker.transform(roi_img), syndata.shape[0], 0)

    age = rng.normal(loc=40, scale=10, size=syndata.shape[0])
    agedata = np.repeat(age[:, np.newaxis], roidata.shape[1], 1)

    return syndata + roidata * agedata * beta, age

simdata, age = generate_synthetic_age_data(beta=20)

# %% [markdown]
# Fit linear regression to confirm that signal injection worked
#
#

# %%
lr = LinearRegression()
_ = lr.fit(simdata, age)
coef_img = masker.inverse_transform(lr.coef_)
cut_coords = [13, 14, 1]
nilearn.plotting.plot_stat_map(coef_img, bg_img=mnitemplate, threshold=.01, cut_coords=cut_coords)

# %% [markdown]
# Let's say that we are interested in trying several different predictive methods to assess their ability to accurately estimate age in a new sample, and to identify the relevant voxels in the brain that support this prediction.

# %%
# try halving grid search to find optimal parameters

models = {
          'svm': LinearSVR(),
          'enet': SGDRegressor(penalty='elasticnet'),
          'ridge': RidgeCV()
         }

param_grid = {}
param_grid['enet'] = {'alpha': np.arange(0.1, 0.9, 0.2), 
              'l1_ratio': np.arange(0.1, 0.95, 0.2)}
param_grid['svm'] = {"C": [0.1, 0.5, 1, 10, 25, 50, 100, 500, 1000, 10000]}

def run_cv(simdata, age, models, param_grid, seed=None, n_splits=20, shuffle_y=False):
    
    scores = {k: [] for k in models.keys()}
    coefs = {k: [] for k in models.keys()}
    best_params = {k: [] for k in models.keys()}

    y = age.copy()
    if shuffle_y:
        np.random.shuffle(y)
        
    rng = np.random.RandomState(seed)

    for model, clf in models.items():
        print(f'running {model}')
        progbar = IntProgress(min=0, max=n_splits) # instantiate the bar
        display(progbar) # display the bar


        for i in range(n_splits):
            progbar.value += 1
            X_train, X_test, y_train, y_test = train_test_split(
                simdata, y, test_size=0.25, random_state=rng)
            if model in param_grid:
                gsh = HalvingGridSearchCV(
                    estimator=clf, param_grid=param_grid[model], factor=2, random_state=rng
                )
                gsh.fit(X_train, y_train)
                clf.set_params(**gsh.best_params_)
                best_params[model].append(gsh.best_params_)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            scores[model].append(r2_score(y_test, y_pred))
            try:
                coefs[model].append(clf.coef_)
            except AttributeError:
                ...
        print(f'mean r-squared for {model}: {np.mean(scores[model])}\n')
    return(coefs, scores, best_params)

coefs, scores, best_params = run_cv(simdata, age, models, param_grid)


# %% [markdown]
# A simple way to look at the stability of the support for each classifier is to identify voxels with significant positive coefficient values across the crossvalidation folds.

# %%
t_img = {}
t = {}
meancoef = {}

for model, coeflist in coefs.items():
    if coeflist == []:
        print('no coefs for', model)
        continue
    meancoef[model] = np.mean(coeflist, axis=0)
    stderr = np.std(coeflist, axis=0, ddof=1) / np.sqrt(len(coeflist))
    t[model] = np.nan_to_num(meancoef[model] / stderr)
    t_img[model] = masker.inverse_transform(t[model])
    t_img_thresh, _ = threshold_stats_img(t_img[model], 
                                       height_control='fdr',
                                       cluster_threshold=20)
    # t_img[model].to_filename(f'{model}_tstat.nii.gz')
    nilearn.plotting.plot_stat_map(t_img_thresh, bg_img=mnitemplate, 
                                   cut_coords=cut_coords, threshold=5, vmax=20,
                                   title=model, symmetric_cbar=False)

# %% [markdown]
# We should also check that there is no leakage in our crossvalidation scheme, by testing the performance of the model when the outcome variable is shuffled.

# %%
_ = run_cv(simdata, age, models, param_grid, shuffle_y=True)


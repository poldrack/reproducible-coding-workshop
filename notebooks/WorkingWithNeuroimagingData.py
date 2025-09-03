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
# ### Working with neuroimaging data in Python
#
#

# %% tags=[]
import nibabel as nib
import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import nilearn
import nilearn.plotting
from nilearn.glm import threshold_stats_img
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd

# %% [markdown]
#
# #### Data setup
#
# For this exercise we will data from a study by Smeets et al. shared in [OpenNeuro](https://openneuro.org/datasets/ds000157/versions/00001).  The study is a blocked design in which subjects were shown pictures of food and non-food images. We will look at one run from one subject as an example.  We first need to download the relevant data files from OpenNeuro, obtaining them directly from Amazon Web Services using the `boto3` package.  These data are stored in [BIDS](http:/bids.neuroimaging.io) format, which makes it easy to identify which files we need for the analysis.

# %%
raw_dir = 'ds000157'
fmriprep_dir = raw_dir + '-fmriprep'
task = 'passiveimageviewing'
run = '' # '_run-1'
ses = '' #'_ses-test'
sub = 'sub-01'
space = '' # '_space-MNI152NLin2009cAsym_res-2'

images = {
        'mask': f"{sub}/func/{sub}{ses}_task-{task}{run}{space}_desc-brain_mask.nii.gz",
        'bold': f"{sub}/func/{sub}{ses}_task-{task}{run}{space}_desc-preproc_bold.nii.gz",
        'boldref': f"{sub}/func/{sub}{ses}_task-{task}{run}{space}_boldref.nii.gz",
        'confounds': f"{sub}/func/{sub}{ses}_task-{task}{run}{space}_desc-confounds_timeseries.tsv"
}

images = {k: os.path.join(fmriprep_dir, v) for k, v in images.items()}
# 
events = {'events': f"sub-01/{ses.replace('_', '')}func/sub-01{ses}_task-{task}{run}_events.tsv"}
# f'task-{task}{run}_events.tsv'}

events = {k: os.path.join(raw_dir, v) for k, v in events.items()}

def get_data(files, s3_bucket='openneuro-derivatives'):

    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    for label, file in files.items():
        if os.path.exists(file):
            print('using existing file:', file)
            continue
        outfile = file
        if 'derivatives' in s3_bucket:
            file = os.path.join('fmriprep', file)
        if not os.path.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))
        print(f'downloading {label}: {file} to {outfile}')
        s3.download_file(s3_bucket, file, outfile)

get_data(images)
get_data(events, s3_bucket='openneuro.org')



# %% [markdown]
# ### Displaying nifti images
#
# The `nilearn` packages provides a number of [plotting tools](https://nilearn.github.io/dev/plotting/index.html) for neuroimaging data. First we will plot the BOLD reference image using `nilearn.plotting.plot_img`.  There are many different options, but by default it plots three orthogonal sections through the image.

# %%

nilearn.plotting.plot_img(images['boldref'], cmap='gray')


# %% [markdown]
# ### Loading data from nifti images
#
# In many cases we would like to load the contents of a NIFTI image for further analysis.  We can do this using the `nibabel` package.  First, we can load the image and look at the information in the header.

# %%
img = nib.load(images['bold'])

print(img.header)

# %% [markdown]
# There are two ways that we can access the data within the image object.  First, we can access them through via the `dataobj` property, which provides an `array proxy` that points to the data:

# %%
img.dataobj.shape

# %% [markdown]
# In general it is prefered to load the data into a new variable, using the `get_fdata()` method of the image object:

# %%
data = img.get_fdata()
print(type(data))
print(data.shape)

# %% [markdown]
# Now we can work with the data as we would with any Numpy array. For example, let's plot the timecourse of one voxel:

# %%
# plot a timeseries from one voxel

tr = img.header.get_zooms()[3]

imgtimes = np.arange(0, img.shape[3] * tr, tr)
plt.plot(imgtimes, data[36, 14, 6, :])
plt.ylabel('BOLD signal')
plt.xlabel('seconds')

# %% [markdown]
# ### loading data from a set of voxels
#
# For many analyses, we would prefer to load a 2-dimensional matrix, with a subset of voxels on one axis and timepoints on the other axis. For example, we might want to run an analysis only on voxels that are within the brain mask.  We can extract data from a set of mask voxels using nilearn's `NiftiMasker`:

# %%
masker = nilearn.maskers.NiftiMasker(images['mask'], standardize=True)
maskdata = masker.fit_transform(images['bold'])

print(maskdata.shape)

# confirm that the number of columns matches number of nonzero voxels in the brain mask

assert maskdata.shape[1] == np.sum(nib.load(images['mask']).dataobj)

# %% [markdown]
# #### Create a "carpet plot"
#
# A "carpet plot" is a two-dimensional plot that presents voxel intensities as an image, with voxels on the Y axis and timepoints on the X axis ([Power, 2017](https://www.sciencedirect.com/science/article/abs/pii/S1053811916303871)).  They are a very useful way to visualize potential problems with an fMRI dataset.  Here we present a carpet plot for the fMRI dataset loaded above, along with a plot of mean global fMRI signal at each timepoint and framewise displacement (a measure of head motion).

# %%
fig, ax = plt.subplots(3, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1, 1]}, sharex=True)

# transpose the maskdata since we want timepoints on the X axis
ax[0].imshow(maskdata.T, aspect='auto', cmap='gray')
plt.tight_layout()
ax[0].set_ylabel('voxels')
_ = ax[0].set_xlabel('timepoints')

confound_df = pd.read_csv(images['confounds'], sep='\t')
ax[1].plot(confound_df.global_signal)
ax[1].set_ylabel('global mean signal')

ax[2].plot(confound_df.framewise_displacement)
ax[2].set_ylabel('framewise displacement')


# %% [markdown]
# Here we can see that head motion is sometimes associated with large whole-brain fluctuations in global signal, as described by Power and colleagues.

# %% [markdown]
# ### Fitting a linear model to the data
#
# In task fMRI we generally fit a linear model (based on the task, along with a set of confound regressors) to identify voxels that show a significant association with the task. The `nilearn` package has a set of functions for performing linear modeling analyses on fMRI data, which provide all of the functionality needed to analyze an fMRI dataset.  Here we provide a simple example by fitting the model to our example dataset from above.  To build the model, we need to load the file that specifies when the events happened during the scan.

# %%
# first set up the events file

events_df = pd.read_csv(events['events'], sep='\t')
if 'trial_type' not in events_df.columns:
    events_df['trial_type'] = task

# %% [markdown]
# Now we set up and estimate the model.

# %%
model = nilearn.glm.first_level.FirstLevelModel(t_r = tr,  smoothing_fwhm=5,
                                               mask_img=images['mask'],
                                               minimize_memory=False)
modelfit = model.fit(img, events_df[['onset', 'duration', 'trial_type']]) 

# extract the fitted response image
fitted_response = modelfit.predicted[0].get_fdata()




# %% [markdown]
# Having fit the model, we need to define a contrast in order to create the relevant statistical maps.  Here we will define a simple contrast that compares activity for both food and non-food images against a resting baseline.  The `generate_report()` method creates a report that provides various information about the contrast result. Here we correct for multiple comparisons using the false discovery rate (FDR) correction; this correction is generally not optimal for images ([Chumbley & Friston, 2009](https://pubmed.ncbi.nlm.nih.gov/18603449/)) but we use it here for convenience. We also impose a cluster size threshold of 30 voxels to remove small clusters.

# %%
conmtx = np.zeros(model.design_matrices_[0].shape[1])
conmtx[1:2] = 1  # set both food and nonfood to 1

modelfit.generate_report(conmtx, bg_img=images['boldref'],
                        cluster_threshold=30, height_control='fdr', alpha=.01)

# %% [markdown]
# In some cases we might want to work directly with the statistical images, which we can do by extracting them using the `compute_contrast()` method.

# %%
contrast_map = model.compute_contrast(conmtx, output_type='z_score')

_, z_threshold = threshold_stats_img(contrast_map, alpha=.01, height_control='fdr')
print('False Discovery rate = 0.05 threshold: %.3f' % z_threshold)

contrast_map_thresh = nilearn.image.threshold_img(contrast_map, threshold=z_threshold,
                                                  cluster_threshold=30, two_sided=False)

# %%
nilearn.plotting.plot_stat_map(contrast_map_thresh, threshold=z_threshold,
                               bg_img=images['boldref'], 
                               display_mode='z', cut_coords=np.arange(-10, 30, 5))

# %% [markdown]
# ### Moving between voxel coordinates and spatial coordinates
#
# There are two ways to refer to particular voxels in an image.  First, we can refer to their index along each of the dimensions of the image; for example, `[3, 5, 8]` would refer to the fourth voxel along the X axis (because indexing starts at zero), fifth voxel along the Y axis, and 8th voxel along the Z axis.  However, we can also refer to them in spatial coordinates, in which the location refers to the distance along each dimension from the *origin* of the image. In data that have been normalized to a standard space such as MNI space, this would refer to the origin (i.e. [0, 0, 0]) in that space; in unnormalized images the origin is usually the center of the image. 
#
# The NIFTI header contains a matrix (known as the *affine* matrix, obtained using the `affine` property) that defines the relationship between voxel coordinates and spatial coordinates.  The affine matrix provides a way to translate between voxel and spatial coordinates by matrix multiplication; see [here](https://nipy.org/nibabel/coordinate_systems.html) for more detail on the use of affine matrices and homogenous coordinates in neuroimaging.  In short, the first three elements in the diagonal of the affine matrix contain the voxel sizes that allow scaling of the coordinates, the first three elements in the fourth column define the origin which specifies the translation of the coordinates, and the off-diagonal elements in the top 3 X 3 matrix define the rotation of the coordinates.

# %%
print(img.affine)

# %%
xyzcoords = [36, 14, 6, 1]
print('voxel coords:', xyzcoords)

# to convert from voxel coords to spatial coords, use dot product of sform with voxel coords
spatialcoords = img.affine.dot(np.array(xyzcoords))
print('spatial coords:', spatialcoords)

# to convert back from spatial coords to voxel coords, use dot product of inverse sform with spatial coords
reconverted = np.linalg.inv(img.affine).dot(spatialcoords)
print('converted back to voxel coords:', reconverted)

# use an assertion test to ensure that this worked
assert np.allclose(xyzcoords, reconverted)

# %% [markdown]
# We can use this knoweldge to extract the data from a particular coordinate and plot it against its fitted response from the model.  

# %%
fig, ax = plt.subplots(1, 2, figsize=(20,6))

# nilearn expects spatial coordinates for its cut_coords argument
nilearn.plotting.plot_stat_map(contrast_map_thresh,  threshold=z_threshold,
                               bg_img=images['boldref'],  display_mode='ortho', axes=ax[0],
                              cut_coords = spatialcoords[:3])

# to extract the data, we need the voxel coords
voxelts = data[xyzcoords[0], xyzcoords[1], xyzcoords[2],  :]
voxelts = voxelts - np.mean(voxelts)
fittedts = fitted_response[xyzcoords[0], xyzcoords[1], xyzcoords[2],  :]

print(f'r-squared = {np.corrcoef(voxelts, fittedts)[0, 1] ** 2}')
ax[1].plot(imgtimes, voxelts)
hrfscale = 100 # scale for visualization
ax[1].plot(imgtimes,  fittedts * hrfscale)

# %%
maskdata.shape

# %%
from sdv.timeseries.deepecho import PAR

par = PAR()
maskdata_df = pd.DataFrame(maskdata)
par.fit(maskdata_df)

# %%

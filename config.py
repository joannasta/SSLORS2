import torch
import numpy as np

# Function to retrieve normalization tensors Hydro
def get_Hydro_means_and_stds():
    means = torch.tensor([
        340.76769064, 429.9430203, 614.21682446, 590.23569706,
        950.68368468, 1792.46290469, 2075.46795189, 2218.94553375,
        2266.46036911, 2246.0605464, 1594.42694882, 1009.32729131
    ], dtype=torch.float)

    stds = torch.tensor([
        554.81258967, 572.41639287, 582.87945694, 675.88746967,
        729.89827633, 1096.01480586, 1273.45393088, 1365.45589904,
        1356.13789355, 1302.3292881, 1079.19066363, 818.86747235
    ], dtype=torch.float)
    
    return means, stds

# Normalization Parameters for Depth MagicBathyNet
NORM_PARAM_DEPTH = {
    "agia_napa": -30.443,
    "puck_lagoon": -11.0
}

# Normalization File Paths MagicBathyNet
NORM_PARAM_PATHS = {
    "agia_napa": "/mnt/storagecube/joanna/MagicBathyNet/agia_napa/norm_param_s2_an.npy",
    "puck_lagoon": "/mnt/storagecube/joanna/MagicBathyNet/puck_lagoon/norm_param_s2_pl.npy"
}

# Common Model Parameters MagicBathyNet
MODEL_CONFIG = {
    "crop_size": 256,
    "window_size": (18, 18),
    "stride": 2
}

# Train and Test indices MagicBathyNet
train_images = ['359', '371', '377', '379', '360', '368', '419', '389', '420', '401', '408', '352', '388', '362', '421', '412', '351', '349', '390', '400', '378']
val_images = ['409', '418', '350', '399', '361', '430', '380']
test_images = ['411', '387', '410', '398', '370', '369', '397']

#MARIDA Normalization Parameter
def get_marida_means_and_stds():
    # Pixel-level number of negative/number of positive per class
    pos_weight = torch.Tensor([ 2.65263158, 27.91666667, 11.39285714, 18.82857143,  6.79775281,
            6.46236559,  0.60648148, 27.91666667, 22.13333333,  5.03478261,
        17.26315789, 29.17391304, 16.79487179, 12.88      ,  9.05797101])

    bands_mean = np.array([0.05197577, 0.04783991, 0.04056812, 0.03163572, 0.02972606, 0.03457443,
    0.03875053, 0.03436435, 0.0392113,  0.02358126, 0.01588816]).astype('float32')

    bands_std = np.array([0.04725893, 0.04743808, 0.04699043, 0.04967381, 0.04946782, 0.06458357,
    0.07594915, 0.07120246, 0.08251058, 0.05111466, 0.03524419]).astype('float32')
    return bands_mean, bands_std, pos_weight

#MARIDA Dataset Mapping
cat_mapping_marida = { 'Marine Debris': 1,
                'Dense Sargassum': 2,
                'Sparse Sargassum': 3,
                'Natural Organic Material': 4,
                'Ship': 5,
                'Clouds': 6,
                'Marine Water': 7,
                'Sediment-Laden Water': 8,
                'Foam': 9,
                'Turbid Water': 10,
                'Shallow Water': 11,
                'Waves': 12,
                'Cloud Shadows': 13,
                'Wakes': 14,
                'Mixed Water': 15}

labels_marida  = ['Marine Debris','Dense Sargassum','Sparse Sargassum',
          'Natural Organic Material','Ship','Clouds','Marine Water','Sediment-Laden Water',
          'Foam','Turbid Water','Shallow Water','Waves','Cloud Shadows','Wakes',
          'Mixed Water']

roi_mapping_marida = { '16PCC' : 'Motagua (16PCC)',
                '16PDC' : 'Ulua (16PDC)',
                '16PEC' : 'La Ceiba (16PEC)',
                '16QED' : 'Roatan (16QED)',
                '18QWF' : 'Haiti (18QWF)',
                '18QYF' : 'Haiti (18QYF)',
                '18QYG' : 'Haiti (18QYG)',
                '19QDA' : 'Santo Domingo (19QDA)',
                '30VWH' : 'Scotland (30VWH)',
                '36JUN' : 'Durban (36JUN)',
                '48MXU' : 'Jakarta (48MXU)',
                '48MYU' : 'Jakarta (48MYU)',
                '48PZC' : 'Danang (48PZC)',
                '50LLR' : 'Bali (50LLR)',
                '51RVQ' : 'Yangtze (51RVQ)',
                '52SDD' : 'Nakdong (52SDD)',
                '51PTS' : 'Manila (51PTS)'}

color_mapping_marida  ={'Marine Debris': 'red',
               'Dense Sargassum': 'green',
               'Sparse Sargassum': 'limegreen',
               'Marine Water': 'navy',
               'Foam': 'purple',
               'Clouds': 'silver',
               'Cloud Shadows': 'gray',
               'Natural Organic Material': 'brown',
               'Ship': 'orange',
               'Wakes': 'yellow', 
               'Shallow Water': 'darkturquoise', 
               'Turbid Water': 'darkkhaki', 
               'Sediment-Laden Water': 'gold', 
               'Waves': 'seashell',
               'Mixed Water': 'rosybrown'}

s2_mapping_marida  = {'nm440': 0,
              'nm490': 1,
              'nm560': 2,
              'nm665': 3,
              'nm705': 4,
              'nm740': 5,
              'nm783': 6,
              'nm842': 7,
              'nm865': 8,
              'nm1600': 9,
              'nm2200': 10,
              'Confidence': 11,
              'Class': 12}

indexes_mapping_marida  = {'NDVI': 0,
                   'FAI': 1,
                   'FDI': 2,
                   'SI': 3,
                   'NDWI': 4,
                   'NRD': 5,
                   'NDMI': 6,
                   'BSI': 7,
                   'Confidence': 8,
                   'Class': 9}

texture_mappin_marida  = {'CON': 0, 
                   'DIS': 1, 
                   'HOMO': 2, 
                   'ENER': 3, 
                   'COR': 4, 
                   'ASM': 5,
                   'Confidence': 6,
                   'Class': 7}

conf_mapping_marida  = {'High': 1,
                'Moderate': 2,
                'Low': 3}

report_mapping_marida  = {'Very close': 1,
                  'Away': 2,
                  'No': 3}

rf_features_marida  = ['nm440','nm490','nm560','nm665','nm705','nm740','nm783','nm842',
              'nm865','nm1600','nm2200','NDVI','FAI','FDI','SI','NDWI','NRD',
              'NDMI','BSI','CON','DIS','HOMO','ENER','COR','ASM']

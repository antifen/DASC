# Stain normalization image
PATH_NORM_IMAGE = "./local_data/color_norm/sample.tif"

# Path with datasets
PATH_DATASETS = r"/root/data/cls/"


# TUPAC16
PATH_TUPAC_Train_IMAGES = PATH_DATASETS + 'tupac/train/'
PATH_TUPAC_Test_IMAGES = PATH_DATASETS + 'tupac/test/'

# MIDOG21
# PATH_MIODG21_Train_IMAGES = PATH_DATASETS + '2021_hard/fp/train/'
# PATH_MIDOG21_Test_IMAGES = PATH_DATASETS + '2021_hard/test2/'
PATH_MIODG21_Train_IMAGES = PATH_DATASETS + '/2021_hard/train3/'
PATH_MIDOG21_Test_IMAGES = PATH_DATASETS + '/2021_hard/test2/'

# Path for results
PATH_RESULTS = "/root/cls/local_data/results/"
MIDOG21_ID_VAL = ['041', '042', '043', '044', '045', '091', '092', '093', '094', '095', '141', '142', '143', '144', '145']
MIDOG21_ID_TEST = ['046', '047', '048', '049', '050', '096', '097', '098', '099', '100', '146', '147', '148', '149', '150']

TUPAC16_ID_TRAIN = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                    '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '32', '33', '34', '35', '36', '39',
                    '40', '41', '42', '43', '46', '47', '48', '49', '50', '53', '54', '55', '56', '57', '60', '61', '62',
                    '63', '64', '67', '68', '69', '70', '71']
TUPAC16_ID_VAL = ['30', '37', '44', '51', '58', '65', '72']
TUPAC16_ID_TEST = ['31', '38', '45', '52', '59', '66', '73']  # This test set is Li et al. (2019) validation set.
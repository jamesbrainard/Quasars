# Classifying Special Quasars

This repository is home to a machine learning model built to classify Broad Absorption Line (BAL) and double-peaked quasars, aiding astrophycisists in the exploration of quasar features and their relation to cosmological phenomena like black hole and galaxy formation.


## Contents
- [Dataset](#dataset)
- [Feature Selection](#feature-selection)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Feature Importance](#feature-importance)
- [Replication (Setup & Usage)](#replication-setup--usage)


# Dataset
- **File:** 'data/Lyke2020.csv'
- **Source:** Lyke, B. W., Higley, A. N., McLane, J. N., Schurhammer, D. P., Myers, A. D., Ross, A. J., ... & Weaver, B. A. (2020). The Sloan Digital Sky Survey Quasar Catalog: Sixteenth Data Release. The Astrophysical Journal Supplement Series, 250(1), 8.
- The dataset contains hundreds of columns related to quasar identification, flux, magnitude, and other features.
- The target variable (BAL_PROB) indicates the BAL probability of each quasar.


# Preprocessing
- **Files:** 
-- preprocessing.ipynb
-- open_fits.py

Preprocessing complex cosmological data is no easy feat. However, the data has been previously processed by Lyke et. al. (2020), albiet for a different purpose.

The data was also originally supplied in a multidimensional .FITS file, which requires some work to convert to a much more usable .csv:
    
    import pandas
    from astropy.io import fits

    filename = 'DR16Q_v4.fits'
    with fits.open(filename) as hdul:
        # Accesses the first extension
        hdu = hdul[1] 
        data_dict = {}

        # Loops over each field in the binary table
        for name in hdu.columns.names:
            column_data = hdu.data[name]
        
            # Tests to see of the column is multi-dimensional
            # If true, flatten the column
            if column_data.ndim > 1:
                # Flattens each row's data and stores in the dictionary
                for i in range(column_data.shape[1]):
                    data_dict[f"{name}_{i}"] = column_data[:, i]
            else:
                # If data is not multi-dimensional, stores it as normal
                data_dict[name] = column_data

        df = pandas.DataFrame(data_dict)

        # Saves the DataFrame to a CSV file
        output_filename = 'output_table.csv'
        df.to_csv(output_filename, index=False)

A few thousand quasars in the dataset are, in fact, not quasars. These are removed rather simply:

    filtered_data = data[
        (data['IS_QSO_FINAL'] == 1)
    ]

Additionally, much of the data is unimportant for our purposes. Dozens upon dozens of columns are simply identifiers, and we don't need identifiers. Identifiers in this dataset all include the name 'DUPLICATE' in the column name. Thus, we remove all columns that contain that word.

    filtered_data = filtered_data.loc[:, ~filtered_data.columns.str.contains('DUPLICATE')]

For the purpose of creating a solid model, we want to remove any quasars that the scientists who created the original dataset were not absolutely certain were or were not BAL quasars.

    filtered_data = filtered_data[
        (data['BAL_PROB'] == 1) | (data['BAL_PROB'] == 0)
    ]

Lastly, we want to remove any dramatic outliers. These are inevitable in most huge datasets, and can introduce bias into models.

    cols = ['PSFFLUX_0', 'PSFFLUX_1', 'PSFFLUX_2', 'PSFFLUX_3', 'PSFFLUX_4',
            'FUV', 'NUV',
            'YFLUX', 'JFLUX', 'HFLUX', 'KFLUX',
            'W1_FLUX', 'W2_FLUX',
            'FIRST_FLUX',
            'XMM_SOFT_FLUX', 'XMM_HARD_FLUX',
            'GAIA_PARALLAX',
            'GAIA_G_FLUX_SNR', 'GAIA_BP_FLUX_SNR', 'GAIA_RP_FLUX_SNR']

    for col in cols:
        Q1 = filtered_data[col].quantile(0.25)
        Q3 = filtered_data[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        filtered_data[col] = numpy.where((filtered_data[col] < lower_bound) | (filtered_data[col] > upper_bound), numpy.nan, filtered_data[col])

# Feature Selection
After experimenting with different subsets of features, it was decided that including all flux data would fit best for the model's performance. In the future, flux data should be modified with relation to z (redshift) to resolve a number closer to a quasar's true energy output.

The following columns are included in the dataset:

    'PSFFLUX_0', 'PSFFLUX_1', 'PSFFLUX_2', 'PSFFLUX_3', 'PSFFLUX_4',
    'FUV', 'NUV', 'YFLUX', 'JFLUX', 'HFLUX', 'KFLUX', 
    'W1_FLUX', 'W2_FLUX', 'FIRST_FLUX', 'XMM_SOFT_FLUX', 'XMM_HARD_FLUX',
    'GAIA_PARALLAX', 'GAIA_G_FLUX_SNR', 'GAIA_BP_FLUX_SNR', 'GAIA_RP_FLUX_SNR'

# Model Training
The model training is conducted in two phases:
1. **K-Fold Cross-Validation:** 5-fold cross-validation is used to assess model consistency across subsets of the data - with such a large dataset, it's no surprise that cross-validation finds little to no difference in accuracy.
2. **Train/Validation/Test Split:** A further split ensures validation accuracy on unseen data. Data is split as follows:
    70% training
    20% testing
    10% validation

# Evaluation
The model is evaluated using the following statistical tools:
- **Accuracy Score**
- **Confusion Matrix**

# Feature Importance
Using the RandomForestClassifier's build-in feature importance functions, we can rank features by their importance to the model:

|          | Feature  | Importance |
|----------|----------|----------|
|     12     | W2_FLUX   | .120979   |
|     0     | PSFFLUX_0   | .095173   |
|     11     | W1_FLUX   | .094985   |
|     1     | PSFFLUX_1   | .090424   |
|     4     | PSFFLUX_4   | .088195   |
|     2     | PSFFLUX_2   | .085659   |
|     3     | PSFFLUX_3   | .084978   |
|     6     | NUV   | .074282   |
|     5     | FUV   | .069433   |
|     17     | GAIA_G_FLUX_SNR   | .055556   |
|     19     | GAIA_RP_FLUX_SNR   | .048642   |
|     18     | GAIA_BP_FLUX_SNR   | .048599   |
|     16     | GAIA_PARALLAX   | .043096   |
|     9     | HFLUX   | .000000   |
|     8     | JFLUX   | .000000   |
|     7     | YFLUX   | .000000   |
|     10     | KFLUX   | .000000   |
|     15     | XMM_HARD_FLUX   | .000000   |
|     13     | FIRST_FLUX   | .000000   |
|     14     | XMM_SOFT_FLUX   | .000000   |

# Replication (Setup & Usage)
1. Clone this repository:

        git clone https://github.com/jamesbrainard/quasars.git
        cd quasars

2. Install the necessary dependencies:

        pip install -r requirements.txt

3. Run preprocessing.ipynb if data/Clean_Quasar_Data.csv is not already downloaded

4. Use the random_forest.ipynb file to recreate the data.
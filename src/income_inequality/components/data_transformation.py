
import os
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from income_inequality.logging import logger 
from imblearn.over_sampling import SMOTE
import joblib 
import warnings
warnings.filterwarnings("ignore")


class DataTransformation:
    def __init__(self, config):
        self.config = config
        self.preprocessor = None 
        self.transformed_df = None 

    def get_data_transformation(self):
        try:
            # Load the dataset
            df = pd.read_csv(self.config.data_path)

            # Drop some columns which can cause data imbalance
            df.drop(columns=["ID", "is_hispanic", "country_of_birth_mother", "country_of_birth_own"], axis=1, inplace=True)

            # Drop columns having more than 70% of missing values
            missing = df.isna().sum().div(df.shape[0]).mul(100).to_frame().sort_values(by=0, ascending = False)
            dropcols = missing[missing[0]>70]
            df.drop(list(dropcols.index), axis=1, inplace=True)

            # Drop duplicates
            df = df.drop_duplicates()

            ## Feature Selection
            selected_columns = ["age", "stocks_status", "wage_per_hour", "industry_code", "gender", "employment_stat",
                                "citizenship", "tax_status", "country_of_birth_father", "mig_year", "income_above_limit"]

            # Drop columns that are not in selected_columns
            df = df[selected_columns]

            # Define the target column
            X = df.drop(columns=["income_above_limit"], axis=1)
            y = df["income_above_limit"]

            # Manually encode the target variable
            y.replace({'Below limit': 0, 'Above limit': 1}, inplace=True)

            # Check for missing values in the target variable
            if y.isnull().any():
                raise ValueError("Target variable 'income_above_limit' contains NaN values.")

            # Impute missing values in the target variable (y) with median
            y = SimpleImputer(strategy='median').fit_transform(y.values.reshape(-1, 1))
            y = y.ravel()

            # Define numerical and categorical features
            numerical_features = X.select_dtypes(exclude="object").columns 
            categorical_features = X.select_dtypes(include="object").columns 

            # Define the Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", RobustScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ordinalencoder", OrdinalEncoder()),
                ]
            )

            logger.info(f"Numerical Columns: {numerical_features}")
            logger.info(f"Categorical Columns: {categorical_features}")

            # Define the preprocessor / transformer
            preprocessor = ColumnTransformer(transformers=[
                    ("OrdinalEncoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features),
                    ("RobustScaler", RobustScaler(), numerical_features)
                ], remainder="passthrough")

            self.preprocessor = preprocessor  # store the preprocessor for later usage 

            # Transform the whole data using the preprocessor
            X_transformed = preprocessor.fit_transform(X)

            # Get the updated column names after encoding
            column_names = numerical_features.to_list() + categorical_features.to_list()

            # Combine X_transformed and y back into one dataframe
            self.transformed_df = pd.DataFrame(X_transformed, columns=column_names)
            self.transformed_df["income_above_limit"] = y 

            logger.info("Data preprocessing Completed")

        except Exception as e:
            raise e 

    def handle_data_imbalance(self):
        if self.transformed_df is None:
            raise ValueError("Data transformation not done. please call get_data_transformation")

        # Split the data into train and test
        train, test = train_test_split(self.transformed_df)

        # Separate the features and target 
        X_train = train.drop(columns=["income_above_limit"])
        y_train = train["income_above_limit"]

        # Handle data imbalance using SMOTE
        oversample = SMOTE()
        X_train, y_train = oversample.fit_resample(X_train, y_train)

        # Save the resampled train set in a CSV file
        train_resampled = pd.DataFrame(X_train, columns=X_train.columns)
        train_resampled["income_above_limit"] = y_train
        train_resampled.to_csv(os.path.join(self.config.root_dir, "train_resampled.csv"), index=False)

        logger.info("Handling data imbalance using SMOTE completed")

    def save_preprocessor(self):
        if self.preprocessor is not None:
            joblib.dump(self.preprocessor, self.config.preprocessor_path)
            logger.info(f"Preprocessor saved to: {self.config.preprocessor_path}")
        else:
            logger.warning("Preprocessor is not available. Please call get_data_transformation to create it")

    def train_test_split(self):
        if self.preprocessor is None:
            raise ValueError("Preprocessor is not available. Please call get_data_transformation.")

        # Split the data into train and test set
        train, test = train_test_split(self.transformed_df)

        # Save the encoded train and test sets in the form of CSV files
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info(f"Splitting the data into train and test set")
        logger.info(f"Shape of train data: {train.shape}")
        logger.info(f"Shape of test data: {test.shape}")



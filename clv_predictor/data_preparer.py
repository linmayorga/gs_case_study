import logging
import pandas as pd
from datetime import timedelta
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class DataPreparer():
    
    def __init__(
        self,
        base_path="",
        xs_path="",
        handle_outliers_flag=True):
        
        self.df_base = pd.read_csv(base_path)
        self.df_xs = pd.read_csv(xs_path)
        self.handle_outliers_flag = handle_outliers_flag

    def prepare_base_data(self):
        
        df_base = self.df_base
        
        #Create still_active column, transform date columns to datetime
        df_base["still_active"] = df_base["customer_churned_at"].isnull().astype(int)
        df_base["customer_started_at"] =pd.to_datetime(df_base["customer_started_at"])
        df_base["customer_churned_at"] =pd.to_datetime(df_base["customer_churned_at"])

        #Fill missing values in customer_churned_at with max_end_date
        max_end_date = df_base["customer_churned_at"].max()
        df_base["customer_churned_at"] = df_base.apply(lambda x: x.customer_churned_at  if pd.isnull(x.customer_churned_at)==False else max_end_date, axis = 1)

        #Create duration ("how_long_customer") column, and start month and day features
        df_base["how_long_customer"] =  df_base["customer_churned_at"] - df_base["customer_started_at"]
        df_base["month_start"] = df_base["customer_started_at"].dt.month
        df_base["day_start"] = df_base["customer_started_at"].dt.weekday

        #Fill operating_system NaNs with "undefined" 
        df_base.loc[df_base["operating_system"].isnull(), "operating_system"] = "undefined"

        #Extract observations with full information for CLV calculation
        df_base_clv = df_base[(df_base["how_long_customer"]>=timedelta(365)) | (df_base["still_active"]==0)].reset_index(drop=True)

        #Create capped duration (max 365 days = 12 months) column and convert to float
        df_base_clv["how_long_customer_capped"] = df_base_clv.apply(lambda x: min(x.how_long_customer, timedelta(365)), axis = 1)
        df_base_clv["how_long_customer_capped"] = df_base_clv["how_long_customer_capped"].dt.total_seconds().astype(float)/(3600*24)

        #Calculate clv_12
        df_base_clv["clv_12"] = df_base_clv.apply(lambda x: x.commission*(x.how_long_customer_capped/(365/12)), axis = 1)

        #Keep only data points with non negative clv_12
        df_base_clv_non_neg = df_base_clv[df_base_clv["clv_12"]>=0].reset_index(drop=True)

        self.df_base_clv_non_neg = df_base_clv_non_neg
        
    
    @staticmethod
    def extract_quartiles_and_outlier_bounds(df, column_list):

        outlier_bounds = {}
        quartiles = {}

        for col in column_list:
            Q1 = df[col].quantile(0.25)
            Q2 = df[col].quantile(0.5)
            Q3 = df[col].quantile(0.75)
            quartiles[col] = {}
            quartiles[col]["Q1"] = Q1
            quartiles[col]["Q2"] = Q2
            quartiles[col]["Q3"] = Q3

            # Calculate the IQR
            IQR = Q3 - Q1

            # Define the lower and upper bounds for outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_bounds[col] = {}
            outlier_bounds[col]["lower"] = lower_bound
            outlier_bounds[col]["upper"] = upper_bound

        return outlier_bounds, quartiles

    
    def handle_outliers_base(self, X_train, y_train):
        logger.warning("Handle outliers")
        
        df_base_clv_ml = X_train.join(y_train)
        outlier_bounds, quartiles = DataPreparer.extract_quartiles_and_outlier_bounds(df_base_clv_ml, ["commission", "clv_12"])
        
        #Refill  high outliers
        df_base_clv_ml.loc[(df_base_clv_ml["commission"]>outlier_bounds["commission"]["upper"]), "commission"] = outlier_bounds["commission"]["upper"]
        df_base_clv_ml.loc[(df_base_clv_ml["clv_12"]>outlier_bounds["clv_12"]["upper"]), "clv_12"] = outlier_bounds["clv_12"]["upper"]
        
        #Refill low outliers
        df_base_clv_ml.loc[(df_base_clv_ml["clv_12"]<outlier_bounds["clv_12"]["lower"]), "clv_12"] = outlier_bounds["clv_12"]["lower"]
        
        #Split in X and y again
        self.X_base_train = df_base_clv_ml.drop(columns=["clv_12"])
        self.y_base_train = df_base_clv_ml["clv_12"]
        
        self.outlier_bounds = outlier_bounds
        self.quartiles = quartiles
        
        
    def prepare_base_data_for_ml_model(self):
        
        self.prepare_base_data()
        X = self.df_base_clv_non_neg[["product", "commission", "channel", "age_bucket", "operating_system", "month_start", "day_start"]]
        y = self.df_base_clv_non_neg["clv_12"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, 
                                            random_state=1)
        if self.handle_outliers_flag==True:
            self.handle_outliers_base(X_train, y_train)
        else:
            self.X_base_train = X_train
            self.y_base_train = y_train
            
        self.X_base_test = X_test
        self.y_base_test = y_test
        
        
    def prepare_xs_data(self):
        
        self.prepare_base_data()        
        df_xs = self.df_xs
        df_base_clv_non_neg = self.df_base_clv_non_neg
        
        #Rename commission to clv_xs (CLV cross-sell)
        df_xs = df_xs.rename(columns={"commission": "clv_xs"})
        
        #Fill clv_xs NaNs with median commission of each product
        df_xs.loc[(df_xs["clv_xs"].isnull())&(df_xs["product"]=="product_x"), "clv_xs"] = df_xs[df_xs["product"]=="product_x"]["clv_xs"].median()
        df_xs.loc[(df_xs["clv_xs"].isnull())&(df_xs["product"]=="product_y"), "clv_xs"] = df_xs[df_xs["product"]=="product_y"]["clv_xs"].median()
        
        #Group cross sell data by user_id (as some users have more than one product)
        #and join with base dataset to access further features
        df_xs_grouped = df_xs.groupby(["user_id"])["clv_xs"].sum().reset_index()
        df_xs_with_base = df_xs_grouped.merge(df_base_clv_non_neg, on="user_id", how="inner")
        
        #Extract negative samples: all observations with start dates between the max and min of start dates of
        #users converted to xs products
        users_xs = df_xs["user_id"].unique().tolist()
        num_pos_samples = len(df_xs_with_base["user_id"].unique())
        min_start_date_xs = df_xs_with_base["customer_started_at"].min()
        max_start_date_xs = df_xs_with_base["customer_started_at"].max()
        df_neg_samples = df_base_clv_non_neg[(df_base_clv_non_neg["user_id"].isin(users_xs)==False) & \
                            (df_base_clv_non_neg["customer_started_at"]>=min_start_date_xs) & \
                            (df_base_clv_non_neg["customer_started_at"]<=max_start_date_xs)].sample(num_pos_samples)
        df_neg_samples["clv_xs"]=0
        
        #Append negative samples to data
        df_xs_with_base_and_neg_samples = df_xs_with_base.append(df_neg_samples)
        
        self.df_xs_with_base_and_neg_samples = df_xs_with_base_and_neg_samples
        
        
    def prepare_xs_data_for_ml_model(self):

        self.prepare_xs_data()
        X = self.df_xs_with_base_and_neg_samples[["product", "commission", "channel", "age_bucket", "operating_system", "month_start", "day_start", "clv_12"]]
        y = self.df_xs_with_base_and_neg_samples["clv_xs"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, 
                                            random_state=1)
        self.X_xs_train = X_train
        self.X_xs_test = X_test
        self.y_xs_train = y_train
        self.y_xs_test = y_test
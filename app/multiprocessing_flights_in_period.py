
import pandas as pd
from multiprocessing import cpu_count, pool

cleaned_df = pd.read_csv('/Users/favea/Downloads/swiss-data/10122022_cleaned.csv')
cleaned_df.scheduled_time_departure = pd.to_datetime(cleaned_df.scheduled_time_departure)

def count_flights_within_one_hour(row):
    # Filter the DataFrame to only include rows with a departure time within one hour of the input datetime
    start_time = row["scheduled_time_departure"] - pd.Timedelta(hours=1)
    end_time = row["scheduled_time_departure"] + pd.Timedelta(hours=1)
    airport = row["origin"]
    filtered_data = cleaned_df[(cleaned_df["scheduled_time_departure"] >= start_time) & (cleaned_df["scheduled_time_departure"] <= end_time) & (cleaned_df['origin']==airport)]
    
    # Count the number of rows in the filtered DataFrame
    return filtered_data.count()['carrier']

if __name__ == '__main__':
        
    
    num_processes = cpu_count()
    pool = pool.Pool(num_processes)

    print("Computing...")
    results = [pool.apply_async(count_flights_within_one_hour, (row,)) for index, row in cleaned_df.iterrows()]

    # Wait for the parallel processing to finish
    pool.close()
    pool.join()

    # Store the results of the parallel processing in a list
    num_flights = [result.get() for result in results]

    pd.Series(num_flights).to_csv('/Users/favea/Downloads/swiss-data/num_flights_within_hour.csv')
import pandas as pd
from multiprocessing import cpu_count, pool

merged = pd.read_csv("/Users/favea/Downloads/swiss-data/merged_df.csv")
merged.wh_fleg_dep_day_scd = pd.to_datetime(
    merged.wh_fleg_dep_day_scd, dayfirst = True
)
merged.wh_fleg_rot_leg_i_prev = merged.wh_fleg_rot_leg_i_prev.astype(float)
merged.wh_fleg_leg_i = merged.wh_fleg_leg_i.astype(float)

def is_previous_delayed_same_day(row):
    # Filter the DataFrame to only include rows with a departure time within one hour of the input datetime
    previous_is_delayed  = False
    departure_date = row["wh_fleg_dep_day_scd"]
    print(departure_date)
    filtered_data = merged[merged.wh_fleg_rot_leg_i_prev == row.wh_fleg_leg_i]
    is_null = bool(len(filtered_data.wh_fleg_leg_i))
  
    if len(filtered_data.wh_fleg_leg_i) > 0:
        if filtered_data.wh_fleg_dep_day_scd.values[0] == departure_date:
            previous_is_delayed = filtered_data.is_delayed.values[0]

    return dict(wh_fleg_leg_i = int(row.wh_fleg_leg_i),
                previous_is_delayed_same_day = previous_is_delayed)
    





if __name__ == "__main__":

    num_processes = cpu_count()
    pool = pool.Pool(num_processes)

    print("Computing...")
    results = [
        pool.apply(is_previous_delayed_same_day, (row,))
        for index, row in merged.iterrows()
    ]

    # Wait for the parallel processing to finish
    pool.close()
    pool.join()

    # Store the results of the parallel processing in a list
    num_flights = [result for result in results]

    print(num_flights)

    pd.DataFrame(num_flights).to_csv(
        "/Users/favea/Downloads/swiss-data/previous_delayed_same_day.csv",
    )

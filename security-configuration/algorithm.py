import pandas as pd

data = pd.read_csv('ukf_energy_data.csv')
data.rename(columns={
    'local_time': 'Time',
    'SINRr': 'SINR',
    'd': 'Distance',
    'Er': 'Residual_Energy',
    'Sreq': 'Security_Requirement',
    'Sth': 'Max_Security_Threshold',
    't': 'Interval_Duration',
    'Eh(i)': 'Harvested_Energy'
}, inplace=True)

K = len(data)
t = data['Interval_Duration'].iloc[0] 

chosen_suites = []

def energy_aware_security_selection(data):
    i = 0
    while i < K:
        max_energy_levels = []
        security_suites = []

        # Calculate energy levels and select security suites
        for j in range(i, K):
            # Calculate Emax for the interval [i, j]
            residual_energy = data.loc[i, 'Residual_Energy']
            harvested_energy = data.loc[i:j + 1, 'Harvested_Energy'].sum()
            e_max = (residual_energy + harvested_energy) / ((j - i + 1) * t)

            # Determine the highest possible security suite
            req_security = max(data.loc[i:j + 1, 'Security_Requirement'])
            max_security_threshold = data.loc[i, 'Max_Security_Threshold']
            suite = min(req_security, max_security_threshold)  # Apply security threshold

            # Store results for the interval
            max_energy_levels.append(e_max)
            security_suites.append(suite)

        # Select the minimum required security level and its index
        min_security = min(security_suites)
        min_security_index = security_suites.index(min_security) + i

        # Record the selected security suite and interval
        chosen_suites.append({
            "Start_Interval": i + 1,
            "End_Interval": min_security_index + 1,
            "Chosen_Security_Suite": min_security
        })

        # Ensure progress by moving to the next interval
        if min_security_index <= i:
            # If no meaningful progress, increment by 1
            i += 1
        else:
            i = min_security_index + 1

    return chosen_suites

results = energy_aware_security_selection(data)

output_df = pd.DataFrame(results)
print(output_df.head(5))

# Save results to a CSV file
output_df.to_csv('chosen_security_suites.csv', index=False)

# Load CSV data
data = pd.read_csv('ukf_energy_data.csv')

data.rename(columns={
    'local_time': 'Time',
    'SINRr': 'SINR',
    'd': 'Distance',
    'Er': 'Residual_Energy',
    'Sreq': 'Security_Requirement',
    'Sth': 'Max_Security_Threshold',
    't': 'Interval_Duration',
    'Eh(i)': 'Harvested_Energy'
}, inplace=True)

K = len(data) 
t = data['Interval_Duration'].iloc[0]


def original_security_selection(data):
    i = 0
    chosen_suites_original = []

    while i < K:
        max_energy_levels = []
        security_suites = []

        for j in range(i, K):
            # Calculate Emax for the interval [i, j]
            residual_energy = data.loc[i, 'Residual_Energy']
            harvested_energy = data.loc[i:j + 1, 'Harvested_Energy'].sum()
            e_max = (residual_energy + harvested_energy) / ((j - i + 1) * t)

            # Select security suite based on requirements (ignoring thresholds in original)
            req_security = max(data.loc[i:j + 1, 'Security_Requirement'])
            security_suites.append(req_security)
            max_energy_levels.append(e_max)

        # Find the highest security level that can be maintained
        min_security = min(security_suites)
        min_security_index = security_suites.index(min_security) + i

        # Record the interval and selected suite
        chosen_suites_original.append({
            "Start_Interval": i + 1,
            "End_Interval": min_security_index + 1,
            "Chosen_Security_Suite": min_security
        })

        i = min_security_index + 1

    return pd.DataFrame(chosen_suites_original)


def modified_security_selection(data):
    i = 0
    chosen_suites_modified = []

    while i < K:
        max_energy_levels = []
        security_suites = []

        for j in range(i, K):
            # Calculate Emax for the interval [i, j]
            residual_energy = data.loc[i, 'Residual_Energy']
            harvested_energy = data.loc[i:j + 1, 'Harvested_Energy'].sum()
            e_max = (residual_energy + harvested_energy) / ((j - i + 1) * t)

            # Select security suite with threshold consideration
            req_security = max(data.loc[i:j + 1, 'Security_Requirement'])
            max_security_threshold = data.loc[i, 'Max_Security_Threshold']
            suite = min(req_security, max_security_threshold)  # Apply security threshold
            security_suites.append(suite)
            max_energy_levels.append(e_max)

        # Find the highest security level that can be maintained
        min_security = min(security_suites)
        min_security_index = security_suites.index(min_security) + i

        # Record the interval and selected suite
        chosen_suites_modified.append({
            "Start_Interval": i + 1,
            "End_Interval": min_security_index + 1,
            "Chosen_Security_Suite": min(min_security, Sth)
        })

        # Update i to move to the next interval
        if min_security_index == i:
            i += 1
        else:
            i = min_security_index + 1

    return pd.DataFrame(chosen_suites_modified)


def compare_algorithms(data):
    # Run both algorithms
    original_results = original_security_selection(data)
    modified_results = modified_security_selection(data)

    # Merge results for comparison
    comparison = pd.concat([original_results, modified_results], axis=1, keys=['Original', 'Modified'])

    # Calculate metrics
    total_security_original = original_results['Chosen_Security_Suite'].sum()
    total_security_modified = modified_results['Chosen_Security_Suite'].sum()

    energy_consumed_original = sum(
        data.loc[start - 1:end - 1, 'Residual_Energy'].sum()
        for start, end in zip(original_results['Start_Interval'], original_results['End_Interval'])
    )
    energy_consumed_modified = sum(
        data.loc[start - 1:end - 1, 'Residual_Energy'].sum()
        for start, end in zip(modified_results['Start_Interval'], modified_results['End_Interval'])
    )

    # Display comparison
    print("Comparison of Algorithms")
    print("=========================")
    print("Total Security Levels (Original):", total_security_original)
    print("Total Security Levels (Modified):", total_security_modified)

    # Save comparison to CSV
    comparison.to_csv('algorithm_comparison.csv', index=False)
    print("\nDetailed comparison saved to 'algorithm_comparison.csv'.")

    return comparison


# Run the comparison
comparison_results = compare_algorithms(data)
print(comparison_results)

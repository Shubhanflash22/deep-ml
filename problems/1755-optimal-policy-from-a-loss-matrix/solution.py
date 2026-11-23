import numpy as np

def expected_losses(posterior, loss):
    return posterior @ loss

def bayes_policy(posterior, loss):
    return np.argmin(expected_losses(posterior, loss), axis=1)

def covid_table(prevalence, sens, spec, drug_cost,
                qaly_young=60.0, qaly_old=10.0):

    # Step 1: Bayes' theorem
    p_negative = (
        (1 - sens) * prevalence /
        ((1 - sens) * prevalence + spec * (1 - prevalence))
    )

    p_positive = (
        sens * prevalence /
        (sens * prevalence + (1 - spec) * (1 - prevalence))
    )

    # Step 2: Build output table
    results = []

    for test, p_covid in [(0, p_negative), (1, p_positive)]:
        for age, qaly in [(0, qaly_young), (1, qaly_old)]:

            # Step 3: Expected costs
            cost_nothing = p_covid * qaly
            cost_drugs = drug_cost

            # Step 4: Optimal action
            action = int(cost_drugs < cost_nothing)

            results.append([
                test,
                age,
                float(p_covid),
                float(cost_nothing),
                float(cost_drugs),
                action
            ])

    return results
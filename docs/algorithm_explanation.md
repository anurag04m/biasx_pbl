# Optimized Preprocessing Algorithm Explanation

## Algorithm Name: Preferential Sampling (Smart Boundary Flipping)

You asked for an algorithm that optimizes preprocessing while producing better, more stable results. The algorithm we have implemented is known as **Preferential Sampling**.

### How It Works

Instead of randomly flipping labels to achieve fairness (which destroys data patterns), this algorithm uses the data itself to decide *which* specific individuals should have their outcomes changed. It operates on the principle of **"Benefit of the Doubt"**.

#### Step 1: Ranking
The algorithm first learns the existing patterns in your data by training a lightweight "Ranker" model (Logistic Regression). This assigns a **score** (0 to 1) to every individual, representing how "qualified" they look based on their non-protected features (like education, experience, credit score, etc.).

#### Step 2: Calculating the Gap
It calculates exactly how many individuals need to be changed to make the acceptance rates equal between the privileged and unprivileged groups.

#### Step 3: Smart Flipping (The Optimization)
It doesn't change just anyone. It targets the **borderline cases**:

1.  **Promoting Unprivileged Candidates**:
    It looks at unprivileged individuals who were rejected ($Y=0$) but had the **highest scores** among the rejected. These are the people who *almost* made it. The algorithm gives them the "benefit of the doubt" and flips them to Positive ($Y=1$).
    *   *Why?* This is the least disruptive way to increase fairness because these individuals are statistically most similar to the successful ones.

2.  **Demoting Privileged Candidates**:
    It looks at privileged individuals who were accepted ($Y=1$) but had the **lowest scores** among the accepted. These are the "borderline" accepted cases. The algorithm flips them to Negative ($Y=0$).
    *   *Why?* Removing these ensures we aren't unfairly favoring barely-qualified privileged candidates over highly-qualified unprivileged ones.

### Why This is Better
*   **Preserves Utility**: It maintains the strong correlations in your data. It doesn't flip a clearly unqualified person to "qualified" just to meet a quota; it flips the most qualified rejected person.
*   **Mathematical Precision**: It calculates the exact number of flips needed to reach 0.0 Disparate Impact (or as close as possible), preventing the "overshoot" you saw previously.

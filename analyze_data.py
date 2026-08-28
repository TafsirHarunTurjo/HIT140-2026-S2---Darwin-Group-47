import pandas as pd
import numpy as np
import statistics as stats_lib          
import scipy.stats as st                
import math

df = pd.read_csv("cleaned.csv")
## we will use pure forword and defense players as population
forword_pop = df[df["Pos"] == "FW"]["Min"]
defense_pop = df[df["Pos"] == "DF"]["Min"]
print(f"Population size for forword players: {len(forword_pop)}")
print(f"Population size for the defenders players: {len(defense_pop)}")
# STEP 2: Sampling
# population= 111 players 
#samples= forword and defense
#forword sample=30
#defense sample=30
np.random.seed(42)  
sample_length = 30
fw_size = len(forword_pop) if len(forword_pop) < sample_length else sample_length
df_size = len(defense_pop) if len(defense_pop) < sample_length else sample_length
fw_sample = forword_pop.sample(n=fw_size, random_state=42)
df_sample = defense_pop.sample(n=df_size, random_state=42)

# STEP 3: Descriptive statistics
def descriptive_stat(sample, label):
    print(f"\n---  Descriptive statistics: {label} ---")
    x_bar = stats_lib.mean(sample)
    print("Mean: %.2f" % x_bar)
    median = stats_lib.median(sample)
    print("Median: %.2f" % median)
    s_square = sample.var(ddof=1)
    s = sample.std(ddof=1)
    print("sample variance: %.2f. sample std. dev.: %.2f." % (s_square, s))
    return x_bar, s, len(sample)

fw_xbar, fw_s, fw_n = descriptive_stat(fw_sample, "Forwards (sample)")
df_xbar, df_s, df_n = descriptive_stat(df_sample, "Defenders (sample)")

# STEP 4: 95% Confidence Interval for the mean of forwards population.
x_bar = fw_xbar
s = fw_s
n = fw_n

print("\n--- Confidence Interval (Forwards mean minutes played) ---")
print("Mean: %.2f. Standard deviation: %.2f. Size: %d." % (x_bar, s, n))

z_score = st.norm.ppf(q=0.975)
print("Z-statistic: %.2f" % z_score)

std_err = s / math.sqrt(n)
print("Standard error: %.2f" % std_err)


mrg_err = z_score * std_err
print("Margin of error: %.2f" % mrg_err)

ci_low = x_bar - mrg_err
ci_upp = x_bar + mrg_err
print("Confidence Interval of the mean: %.2f to %.2f" % (ci_low, ci_upp))


# STEP 5: Two-sample t-test
# H0: mean minutes of forwords = mean minutes of defenders
# H1: mean minutes of forwords does not equal mean minutes of defenders

# sample 1 statistics (Forwards)
x_bar1 = fw_xbar
s1 = fw_s
n1 = fw_n

# defenders
x_bar2 = df_xbar
s2 = df_s
n2 = df_n

t_stats, p_val = st.ttest_ind_from_stats(
    x_bar1, s1, n1, x_bar2, s2, n2, equal_var=False, alternative='two-sided'
)

print("\t t-statistic (t) is: %.2f" % t_stats)


print("\t p-value is: %.4f" % p_val)

print("\n Conclusion:")
if p_val < 0.05:
    print("\t We reject the null hypothesis.")
    print("\t There IS a statistically significant difference in mean minutes played.")
else:
    print("\t We accept the null hypothesis.")
    print("\t There is NO statistically significant difference in mean minutes played.")

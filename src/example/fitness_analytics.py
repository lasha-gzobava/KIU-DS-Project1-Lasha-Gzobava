import numpy as np


class FitnessAnalytics:
    def __init__(self, seed=2025):
        self.rng = np.random.default_rng(seed)
        self.n_users = 100
        self.n_days = 90
        self.n_metrics = 4
        self.metric_names = ["Steps", "Calories", "Active Minutes", "Avg Heart Rate"]
        self.cube = None
        self.profile = None

        # Goals
        self.GOAL_STEPS = 8000
        self.GOAL_CALS = 2000
        self.GOAL_ACTIVE = 60

    def generate_data(self):
        """Generate synthetic fitness data"""
        print(f"Simulating {self.n_users} users × {self.n_days} days × {self.n_metrics} metrics…")

        self.cube = np.zeros((self.n_users, self.n_days, self.n_metrics), dtype=float)

        # Steps
        self.cube[:, :, 0] = self.rng.normal(loc=8500, scale=2500, size=(self.n_users, self.n_days))
        self.cube[:, :, 0] = np.clip(self.cube[:, :, 0], 2000, 15000)

        # Calories
        self.cube[:, :, 1] = self.rng.normal(loc=2500, scale=400, size=(self.n_users, self.n_days))
        self.cube[:, :, 1] = np.clip(self.cube[:, :, 1], 1500, 3500)

        # Active minutes
        self.cube[:, :, 2] = self.rng.normal(loc=100, scale=30, size=(self.n_users, self.n_days))
        self.cube[:, :, 2] = np.clip(self.cube[:, :, 2], 20, 180)

        # Heart rate
        self.cube[:, :, 3] = self.rng.normal(loc=85, scale=12, size=(self.n_users, self.n_days))
        self.cube[:, :, 3] = np.clip(self.cube[:, :, 3], 60, 120)

        print(f"Data shape: {self.cube.shape}")
        print("Example (User 1, Day 1):", np.round(self.cube[0, 0, :], 2))

        # User metadata
        self.profile = np.zeros((self.n_users, 3), dtype=float)
        self.profile[:, 0] = np.arange(1, self.n_users + 1)
        self.profile[:, 1] = self.rng.integers(18, 71, size=self.n_users)
        self.profile[:, 2] = self.rng.integers(0, 2, size=self.n_users)

        print("\nUser metadata (first five):")
        print("User | Age | Gender")
        for i in range(5):
            print(f"  {int(self.profile[i,0]):3d} | {int(self.profile[i,1]):2d} | {'Male' if self.profile[i,2]==1 else 'Female'}")

    def inject_noise(self):
        """Inject NaNs and outliers for cleaning demonstration"""
        total_vals = self.cube.size
        nan_n = int(total_vals * 0.05)
        outlier_n = int(total_vals * 0.02)

        flat = self.cube.reshape(-1)
        nan_positions = self.rng.choice(total_vals, size=nan_n, replace=False)
        flat[nan_positions] = np.nan
        self.cube = flat.reshape(self.cube.shape)
        print(f"\nInjected NaNs: {nan_n} ({nan_n/total_vals*100:.1f}%)")

        flat = self.cube.reshape(-1)
        outlier_positions = self.rng.choice(total_vals, size=outlier_n, replace=False)
        for idx in outlier_positions:
            m = idx % self.n_metrics
            if m == 0:
                flat[idx] = self.rng.choice([500, 50000])
            elif m == 1:
                flat[idx] = self.rng.choice([500, 8000])
            elif m == 2:
                flat[idx] = self.rng.choice([5, 400])
            else:
                flat[idx] = self.rng.choice([30, 180])
        self.cube = flat.reshape(self.cube.shape)
        print(f"Injected outliers: {outlier_n} ({outlier_n/total_vals*100:.1f}%)")

    def impute_by_metric_mean(self):
        """Fill NaN values with metric means"""
        for m in range(self.cube.shape[2]):
            col = self.cube[:, :, m]
            mean_val = np.nanmean(col)
            col[np.isnan(col)] = mean_val
            self.cube[:, :, m] = col

    def winsorize_iqr(self, metric_index):
        """IQR-based outlier capping"""
        col = self.cube[:, :, metric_index].ravel()
        valid = ~np.isnan(col)
        q1, q3 = np.percentile(col[valid], [25, 75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        med = np.nanmedian(col)
        mask = (col < low) | (col > high)
        count = int(np.sum(mask & valid))
        col[mask] = med
        self.cube[:, :, metric_index] = col.reshape(self.cube.shape[0], self.cube.shape[1])
        return count, low, high

    def clean_data(self):
        """Execute data cleaning pipeline"""
        print("\n— Cleaning pipeline —")
        start_nans = int(np.isnan(self.cube).sum())
        print(f"Initial NaN count: {start_nans}")

        for m_idx, m_name in enumerate(self.metric_names):
            removed, lo, hi = self.winsorize_iqr(m_idx)
            print(f"  {m_name}: outliers adjusted={removed}, valid range≈[{lo:.2f}, {hi:.2f}]")

        self.impute_by_metric_mean()
        end_nans = int(np.isnan(self.cube).sum())
        print(f"Remaining NaNs after imputation: {end_nans}")

    def run_analysis(self):
        """Execute complete fitness analytics"""
        print("\n=== PART C: FITNESS DATA PIPELINE ===")

        self.generate_data()
        self.inject_noise()
        self.clean_data()

        self._summary_stats()
        self._per_user_analysis()
        self._top_performers()
        self._consistency_analysis()
        self._activity_tiers()
        self._temporal_trends()
        self._weekday_analysis()
        self._period_comparison()
        self._metric_correlations()
        self._demographic_analysis()
        self._health_score()
        self._goal_tracking()

    def _summary_stats(self):
        print("\nCleaned data summary (per metric):")
        for m_idx, m_name in enumerate(self.metric_names):
            col = self.cube[:, :, m_idx]
            print(f"  {m_name}: mean={col.mean():.2f}, std={col.std():.2f}, "
                  f"min={col.min():.2f}, max={col.max():.2f}")

    def _per_user_analysis(self):
        per_user_avg = self.cube.mean(axis=1)
        print("\nPer-user averages (first five):")
        print("User | Steps | Calories | Active | HR")
        for i in range(5):
            uid = int(self.profile[i, 0])
            a = per_user_avg[i]
            print(f" {uid:3d} | {a[0]:6.1f} | {a[1]:8.1f} | {a[2]:6.1f} | {a[3]:6.1f}")

    def _top_performers(self):
        per_user_avg = self.cube.mean(axis=1)
        z = np.zeros_like(per_user_avg)
        for j in range(self.n_metrics):
            mu, sd = per_user_avg[:, j].mean(), per_user_avg[:, j].std()
            z[:, j] = (per_user_avg[:, j] - mu) / sd
        combo = z.sum(axis=1)
        top10 = np.argsort(combo)[-10:][::-1]

        print("\nTop 10 users by combined z-score:")
        print("Rank | User | Combo-Z | Steps | Calories")
        for r, idx in enumerate(top10, 1):
            uid = int(self.profile[idx, 0])
            print(f"  {r:2d}  | {uid:4d} | {combo[idx]:8.2f} | {per_user_avg[idx,0]:6.1f} | {per_user_avg[idx,1]:8.1f}")

    def _consistency_analysis(self):
        per_user_avg = self.cube.mean(axis=1)
        step_stability = self.cube[:, :, 0].std(axis=1)
        consistent = np.argsort(step_stability)[:10]
        print("\nMost consistent users (steps std):")
        print("Rank | User | Std | Avg Steps")
        for r, idx in enumerate(consistent, 1):
            uid = int(self.profile[idx, 0])
            print(f"  {r:2d}  | {uid:4d} | {step_stability[idx]:6.2f} | {per_user_avg[idx,0]:8.1f}")

    def _activity_tiers(self):
        per_user_avg = self.cube.mean(axis=1)
        q25 = np.percentile(per_user_avg[:, 0], 25)
        q75 = np.percentile(per_user_avg[:, 0], 75)
        low = np.sum(per_user_avg[:, 0] < q25)
        mid = np.sum((per_user_avg[:, 0] >= q25) & (per_user_avg[:, 0] <= q75))
        hi = np.sum(per_user_avg[:, 0] > q75)
        print("\nActivity distribution (by avg steps):")
        print(f"  Low (<{q25:.0f}): {low} users")
        print(f"  Mid ({q25:.0f}–{q75:.0f}): {mid} users")
        print(f"  High (>{q75:.0f}): {hi} users")

    def _temporal_trends(self):
        pop_daily = self.cube.mean(axis=0)
        window = 7
        rolling7 = np.zeros((self.n_days - window + 1, self.n_metrics))
        kern = np.ones(window) / window
        for j in range(self.n_metrics):
            rolling7[:, j] = np.convolve(pop_daily[:, j], kern, mode="valid")
        print("\nRolling 7-day averages:")
        print("First 5 (Steps):", np.round(rolling7[:5, 0], 2))

    def _weekday_analysis(self):
        pop_daily = self.cube.mean(axis=0)
        weekday_avg = np.zeros((7, self.n_metrics))
        for d in range(7):
            weekday_avg[d] = pop_daily[d::7].mean(axis=0)
        labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        print("\nAverage by day of week:")
        print("Day  | Steps | Calories | Active | HR")
        for d, name in enumerate(labels):
            row = weekday_avg[d]
            print(f"{name:3s} | {row[0]:6.1f} | {row[1]:8.1f} | {row[2]:6.1f} | {row[3]:6.1f}")

    def _period_comparison(self):
        pop_daily = self.cube.mean(axis=0)
        early = pop_daily[:30].mean(axis=0)
        late = pop_daily[-30:].mean(axis=0)
        print("\nFirst 30 vs Last 30 days:")
        for j, name in enumerate(self.metric_names):
            delta = late[j] - early[j]
            pct = (delta / early[j]) * 100
            print(f"  {name}: {early[j]:.1f} → {late[j]:.1f} ({'↑' if delta>0 else '↓'} {abs(pct):.1f}%)")

        m1 = pop_daily[:30].mean(axis=0)
        m2 = pop_daily[30:60].mean(axis=0)
        m3 = pop_daily[60:90].mean(axis=0)
        g12 = (m2 - m1) / m1 * 100
        g23 = (m3 - m2) / m2 * 100
        print("\nMonth-to-month growth:")
        print("  M1→M2:")
        for j, name in enumerate(self.metric_names):
            print(f"    {name}: {g12[j]:+.2f}%")
        print("  M2→M3:")
        for j, name in enumerate(self.metric_names):
            print(f"    {name}: {g23[j]:+.2f}%")

    def _metric_correlations(self):
        flat_metrics = self.cube.reshape(-1, self.n_metrics)
        corr = np.corrcoef(flat_metrics.T)
        print("\nMetric correlation matrix:")
        header = "            " + "  ".join(f"{n:>10s}" for n in self.metric_names)
        print(header)
        for i, name in enumerate(self.metric_names):
            row = "  ".join(f"{v:10.3f}" for v in corr[i])
            print(f"{name:12s} {row}")

    def _demographic_analysis(self):
        per_user_avg = self.cube.mean(axis=1)

        # Age groups
        bands = [(18, 30), (31, 45), (46, 60), (61, 70)]
        band_avg = np.zeros((len(bands), self.n_metrics))
        for k, (lo_age, hi_age) in enumerate(bands):
            m = (self.profile[:, 1] >= lo_age) & (self.profile[:, 1] <= hi_age)
            if np.any(m):
                band_avg[k] = per_user_avg[m].mean(axis=0)
        print("\nAverage by age group:")
        print("Group      | Steps | Calories | Active")
        for (lo_age, hi_age), row in zip(bands, band_avg):
            print(f"{lo_age:02d}-{hi_age:02d}     | {row[0]:6.1f} | {row[1]:8.1f} | {row[2]:6.1f}")

        # Gender
        fem = self.profile[:, 2] == 0
        mal = self.profile[:, 2] == 1
        fem_avg = per_user_avg[fem].mean(axis=0)
        mal_avg = per_user_avg[mal].mean(axis=0)
        print("\nAverage by gender:")
        print("Gender | Steps | Calories | Active | HR")
        print(f"Female | {fem_avg[0]:6.1f} | {fem_avg[1]:8.1f} | {fem_avg[2]:6.1f} | {fem_avg[3]:6.1f}")
        print(f"Male   | {mal_avg[0]:6.1f} | {mal_avg[1]:8.1f} | {mal_avg[2]:6.1f} | {mal_avg[3]:6.1f}")

    def _health_score(self):
        per_user_avg = self.cube.mean(axis=1)

        def minmax(v):
            return (v - v.min()) / (v.max() - v.min()) * 100.0

        steps_n = minmax(per_user_avg[:, 0])
        cals_n = minmax(per_user_avg[:, 1])
        act_n = minmax(per_user_avg[:, 2])
        health = 0.4 * steps_n + 0.3 * cals_n + 0.3 * act_n
        best5 = np.argsort(health)[-5:][::-1]

        print("\nTop 5 by composite health score:")
        print("Rank | User | Score")
        for r, idx in enumerate(best5, 1):
            print(f"  {r}   | {int(self.profile[idx,0]):4d} | {health[idx]:8.2f}")

    def _goal_tracking(self):
        print("\nDaily goals:",
              f"{self.GOAL_STEPS} steps, {self.GOAL_CALS} kcal, {self.GOAL_ACTIVE} active min")

        hit_steps = (self.cube[:, :, 0] >= self.GOAL_STEPS).sum(axis=1) / self.n_days * 100
        hit_cals = (self.cube[:, :, 1] >= self.GOAL_CALS).sum(axis=1) / self.n_days * 100
        hit_active = (self.cube[:, :, 2] >= self.GOAL_ACTIVE).sum(axis=1) / self.n_days * 100
        hit_all = ((self.cube[:, :, 0] >= self.GOAL_STEPS) &
                   (self.cube[:, :, 1] >= self.GOAL_CALS) &
                   (self.cube[:, :, 2] >= self.GOAL_ACTIVE)).sum(axis=1) / self.n_days * 100

        print("\nGoal achievement (averages):")
        print(f"  Steps:   {hit_steps.mean():.1f}%")
        print(f"  Calories:{hit_cals.mean():.1f}%")
        print(f"  Active:  {hit_active.mean():.1f}%")
        print(f"  All 3:   {hit_all.mean():.1f}%")

        per_user_avg = self.cube.mean(axis=1)
        consistent = np.where(hit_all > 80)[0]
        print(f"\nUsers with >80% all-goals days: {consistent.size}")
        if consistent.size:
            print("User  | Rate% | Avg Steps | Avg Calories | Avg Active")
            for idx in consistent[:10]:
                print(f"{int(self.profile[idx,0]):5d} | {hit_all[idx]:5.1f} | "
                      f"{per_user_avg[idx,0]:9.1f} | {per_user_avg[idx,1]:11.1f} | {per_user_avg[idx,2]:9.1f}")


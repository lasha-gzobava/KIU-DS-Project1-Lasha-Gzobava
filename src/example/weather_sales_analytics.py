"""
Weather and Sales Analytics Module
Handles climate data generation, sales analysis, and array operations
"""
import numpy as np


class WeatherSalesAnalytics:
    def __init__(self, seed=1337):
        self.rng = np.random.default_rng(seed)
        self.climate_grid = None
        self.sales_book = None

    def generate_data(self):
        """Generate climate and sales data"""
        # 5 locations × 365 days (values in °C)
        self.climate_grid = self.rng.uniform(low=-10.0, high=40.0, size=(5, 365))

        # Sales: 12 months × 4 categories (integers)
        self.sales_book = self.rng.integers(low=2000, high=5001, size=(12, 4))

    def run_analysis(self):
        """Execute complete weather and sales analysis"""
        print("\n=== PART A: ARRAYS, INDEXING & STATS ===")
        print("Generating climate and sales data…")

        self.generate_data()

        # Basic metadata
        print(f"Shape: {self.climate_grid.shape}  |  ndim: {self.climate_grid.ndim}  |  dtype: {self.climate_grid.dtype}")
        print("Preview (all locations, first 3 days):")
        print(np.round(self.climate_grid[:, :3], 2))

        # Sales Matrix
        print("\n— Sales Matrix —")
        print(f"Matrix shape: {self.sales_book.shape}")
        print(self.sales_book)

        # Utility Arrays
        self._display_utility_arrays()

        # Slicing examples
        self._slicing_examples()

        # Boolean indexing
        self._boolean_queries()

        # Fancy indexing & transforms
        self._fancy_indexing()

        # Stats
        self._location_stats()

        # Sales summaries
        self._sales_summaries()

        # Rolling & normalization
        self._rolling_and_normalization()

    def _display_utility_arrays(self):
        print("\n— Utility Arrays —")
        ident = np.eye(5, dtype=float)
        ramp = np.linspace(0, 100, num=50)
        print("Identity(5×5):")
        print(ident)
        print("Linspace 0→100 (n=50):")
        print(f"Head: {np.round(ramp[:5], 2)}")
        print(f"Tail: {np.round(ramp[-5:], 2)}")

    def _slicing_examples(self):
        print("\n[ Slicing examples ]")
        jan = self.climate_grid[:, 0:31]
        summer = self.climate_grid[:, 152:243]
        weekends = self.climate_grid[:, 4::7]
        print(f"January block: {jan.shape}, Summer block: {summer.shape}, Weekends: {weekends.shape}")

    def _boolean_queries(self):
        print("\n[ Boolean queries ]")
        hot_any_city = np.any(self.climate_grid > 35.0, axis=0)
        hot_days = np.flatnonzero(hot_any_city)
        print(f"Days with at least one city >35°C: {hot_days.size}")

        all_freezing_days = np.flatnonzero(np.all(self.climate_grid < 0.0, axis=0))
        print(f"Days when EVERY city <0°C: {all_freezing_days.size}")

        cozy_mask = np.all((self.climate_grid >= 15.0) & (self.climate_grid <= 25.0), axis=0)
        print(f"Days all cities are 15–25°C: {np.count_nonzero(cozy_mask)}")

        cold_before = np.count_nonzero(self.climate_grid < -5.0)
        self.climate_grid = np.where(self.climate_grid < -5.0, -5.0, self.climate_grid)
        print(f"Cleaned extreme colds: {cold_before} entries raised to -5°C")

    def _fancy_indexing(self):
        print("\n[ Fancy indexing / Transformations ]")
        grab_days = [0, 100, 200, 300, 364]
        picked = self.climate_grid[:, grab_days]
        print(f"Temperatures on selected days {grab_days}:")
        print(np.round(picked, 2))

        quarters = np.array_split(self.climate_grid, 4, axis=1)
        quarterly_means = np.vstack([q.mean(axis=1) for q in quarters]).T
        print("\nQuarterly means (locations × 4 quarters):")
        print(np.round(quarterly_means, 2))

        yearly_means = self.climate_grid.mean(axis=1)
        ranked = np.argsort(yearly_means)[::-1]
        print("\nLocations ranked by annual mean temperature:")
        for idx, loc in enumerate(ranked, 1):
            print(f"  #{idx}: Location {loc+1} — {yearly_means[loc]:.2f}°C")

    def _location_stats(self):
        print("\n[ Location stats ]")
        loc_means = self.climate_grid.mean(axis=1)
        loc_stds = self.climate_grid.std(axis=1)
        for loc_id in range(self.climate_grid.shape[0]):
            print(f"  Loc {loc_id+1}: mean={loc_means[loc_id]:.2f}°C, std={loc_stds[loc_id]:.2f}°C")

        coldest_day_idx = self.climate_grid.argmin(axis=1)
        hottest_day_idx = self.climate_grid.argmax(axis=1)
        coldest_vals = self.climate_grid[np.arange(5), coldest_day_idx]
        hottest_vals = self.climate_grid[np.arange(5), hottest_day_idx]

        print("\nExtremes by location:")
        for loc_id in range(5):
            print(f"  Loc {loc_id+1}: min {coldest_vals[loc_id]:.2f}°C (day {coldest_day_idx[loc_id]+1}), "
                  f"max {hottest_vals[loc_id]:.2f}°C (day {hottest_day_idx[loc_id]+1})")

        ranges = hottest_vals - coldest_vals
        print("\nDaily range (max-min) per location:")
        for loc_id, span in enumerate(ranges, 1):
            print(f"  Loc {loc_id}: {span:.2f}°C")

        corr_map = np.corrcoef(self.climate_grid)
        print("\nCorrelation matrix (locations × locations):")
        print(np.round(corr_map, 3))

    def _sales_summaries(self):
        print("\n[ Sales summaries ]")
        per_category_total = self.sales_book.sum(axis=0)
        for cat_id, total in enumerate(per_category_total, 1):
            print(f"  Category {cat_id}: ${total:,}")

        monthly_means_across_cats = self.sales_book.mean(axis=1)
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        print("\nAverage monthly sales (across categories):")
        for m, val in zip(months, monthly_means_across_cats):
            print(f"  {m}: ${val:.2f}")

        monthly_totals = self.sales_book.sum(axis=1)
        best_month = int(monthly_totals.argmax())
        best_cat = int(per_category_total.argmax())
        print(f"\nTop month: {months[best_month]} (total ${monthly_totals[best_month]:,})")
        print(f"Best category overall: Category {best_cat+1} (total ${per_category_total[best_cat]:,})")

    def _rolling_and_normalization(self):
        print("\n[ Rolling averages & normalization ]")
        win = 7
        mov_avg = np.vstack([
            np.convolve(self.climate_grid[r], np.ones(win)/win, mode="valid")
            for r in range(self.climate_grid.shape[0])
        ])
        print(f"7-day moving average shape: {mov_avg.shape}")
        print("Example (Loc 1, first 5):", np.round(mov_avg[0, :5], 2))

        mean_cols = self.climate_grid.mean(axis=1, keepdims=True)
        std_cols = self.climate_grid.std(axis=1, keepdims=True)
        zed = (self.climate_grid - mean_cols) / std_cols
        print("\nZ-Score sanity check per location (mean≈0, std≈1):")
        for loc_id in range(5):
            print(f"  Loc {loc_id+1}: mean={zed[loc_id].mean():.5f}, std={zed[loc_id].std():.5f}")

        p25 = np.percentile(self.climate_grid, 25, axis=1)
        p50 = np.percentile(self.climate_grid, 50, axis=1)
        p75 = np.percentile(self.climate_grid, 75, axis=1)

        print("\nPercentiles by location:")
        for loc_id in range(5):
            iqr = p75[loc_id] - p25[loc_id]
            print(f"  Loc {loc_id+1}: 25th={p25[loc_id]:.2f}°C, 50th={p50[loc_id]:.2f}°C, "
                  f"75th={p75[loc_id]:.2f}°C, IQR={iqr:.2f}°C")


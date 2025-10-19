
from weather_sales_analytics import WeatherSalesAnalytics
from student_grading import StudentGradingSystem
from fitness_analytics import FitnessAnalytics


def main():
    """Run all analytics modules"""

    # Part A: Weather and Sales Analytics
    weather_sales = WeatherSalesAnalytics(seed=1337)
    weather_sales.run_analysis()

    # Part B: Student Grading System
    student_system = StudentGradingSystem()
    student_system.run_analysis()

    # Part C: Fitness Analytics
    fitness = FitnessAnalytics(seed=2025)
    fitness.run_analysis()

    # Final message
    print("\n" + "=" * 60)
    print("Completed — All 3 modules executed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

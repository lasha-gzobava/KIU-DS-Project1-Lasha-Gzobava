
class StudentGradingSystem:
    def __init__(self):
        self.roster = {
            "S001": {"full_name": "Luka Kapanadze", "marks": [91, 87, 84, 90], "attended": 28},
            "S002": {"full_name": "Emma Roberts", "marks": [78, 82, 75, 79], "attended": 26},
            "S003": {"full_name": "Nino Gelashvili", "marks": [89, 93, 88, 92], "attended": 30},
            "S004": {"full_name": "Noah Smith", "marks": [66, 71, 69, 73], "attended": 22},
            "S005": {"full_name": "Levan Chavchavadze", "marks": [96, 94, 98, 95], "attended": 29},
            "S006": {"full_name": "Sophia Wilson", "marks": [81, 79, 84, 80], "attended": 27},
            "S007": {"full_name": "Dato Tsiklauri", "marks": [62, 67, 65, 70], "attended": 19},
            "S008": {"full_name": "James Taylor", "marks": [85, 89, 83, 88], "attended": 30},
            "S009": {"full_name": "Mariam Jorjadze", "marks": [76, 80, 77, 81], "attended": 25},
            "S010": {"full_name": "Lasha Gzobava", "marks": [100,100,100,100], "attended": 29},
        }
        self.max_lectures = 30

    def avg_score(self, points):
        """Calculate average score"""
        return round(sum(points) / len(points), 2)

    def letter_grade(self, mean_val):
        """Convert numeric grade to letter grade"""
        return "A" if mean_val >= 90 else \
               "B" if mean_val >= 80 else \
               "C" if mean_val >= 70 else \
               "D" if mean_val >= 60 else "F"

    def eligibility(self, info):
        """
        Check if student passes based on grades and attendance
        Returns (is_pass, reason_str)
        """
        mean_pts = self.avg_score(info["marks"])
        presence_pct = (info["attended"] / self.max_lectures) * 100.0

        if mean_pts < 60:
            return False, f"Insufficient average ({mean_pts})"
        if presence_pct < 75:
            return False, f"Insufficient attendance ({info['attended']}/{self.max_lectures})"
        return True, "Passed"

    def leaderboard(self, k=5):
        """Get top k students by average score"""
        buff = []
        for sid, s in self.roster.items():
            buff.append((sid, self.avg_score(s["marks"])))
        buff.sort(key=lambda t: t[1], reverse=True)
        return buff[:k]

    def class_report(self):
        """Generate comprehensive class report"""
        n = len(self.roster)
        means = [self.avg_score(s["marks"]) for s in self.roster.values()]
        presences = [s["attended"] for s in self.roster.values()]
        passes = [self.eligibility(s)[0] for s in self.roster.values()]
        return {
            "total": n,
            "passed": sum(passes),
            "failed": n - sum(passes),
            "class_avg": round(sum(means) / n, 2),
            "max_avg": max(means),
            "min_avg": min(means),
            "avg_attendance_rate": round(sum(presences) / (n * self.max_lectures) * 100, 2)
        }

    def run_analysis(self):
        """Execute complete student evaluation analysis"""
        print("\n=== PART B: STUDENT EVALUATION SYSTEM ===")

        # Top performers
        print("\nTop 5 performers:")
        for rnk, (sid, mean_val) in enumerate(self.leaderboard(5), start=1):
            print(f"  {rnk}. {sid} — {self.roster[sid]['full_name']}: {mean_val} ({self.letter_grade(mean_val)})")

        # Students who failed
        print("\nStudents who did not pass:")
        failures = []
        for sid, s in self.roster.items():
            ok, because = self.eligibility(s)
            if not ok:
                failures.append((sid, because))

        if failures:
            for sid, because in failures:
                print(f"  {sid} — {because}")
        else:
            print("  None 🎉")

        # Class summary
        print("\nClass summary:")
        summary = self.class_report()
        total = summary["total"]
        passed = summary["passed"]
        failed = summary["failed"]
        print("-" * 48)
        print(f"Total: {total}")
        print(f"Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total*100:.1f}%)")
        print(f"Class Average: {summary['class_avg']}")
        print(f"Highest Average: {summary['max_avg']}")
        print(f"Lowest  Average: {summary['min_avg']}")
        print(f"Avg Attendance Rate: {summary['avg_attendance_rate']}%")

        # Grade distribution
        grades = [self.letter_grade(self.avg_score(s["marks"])) for s in self.roster.values()]
        by_grade = {g: grades.count(g) for g in sorted(set(grades))}
        print("\nGrade distribution:")
        for g, cnt in by_grade.items():
            print(f"  {g}: {cnt} student{'s' if cnt != 1 else ''}")


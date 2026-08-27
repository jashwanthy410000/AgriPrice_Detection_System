"""AgriPrice AI end-to-end pipeline runner."""

import os
import subprocess
import sys
import time
from datetime import datetime
try:
	import schedule
except ImportError:
	schedule = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_step(script_relative_path: str, title: str) -> None:
	script_path = os.path.join(BASE_DIR, script_relative_path)
	print(f"\n[START] {title}")
	result = subprocess.run([sys.executable, script_path], cwd=BASE_DIR, check=False)
	if result.returncode != 0:
		raise RuntimeError(f"Step failed: {title}")
	print(f"[DONE]  {title}")


def run_pipeline(fetch_live_data: bool = False) -> None:
	os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
	os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

	print("=" * 58)
	print("AgriPrice AI - Starting Full Pipeline")
	print("=" * 58)

	if fetch_live_data or not os.path.exists(os.path.join(BASE_DIR, "data", "raw_data.csv")):
		run_step("src/fetch_real_data.py", "Fetching Real Data from data.gov.in AGMARKNET")

	run_step("src/data_preprocessing.py", "Data Preprocessing")
	run_step("src/eda.py", "Exploratory Data Analysis")
	run_step("src/model.py", "Model Training")
	run_step("src/evaluation.py", "Evaluation and Forecasting")

	print("\n" + "=" * 58)
	print("Pipeline completed successfully.")
	print("Outputs: outputs/  Models: models/")
	print("=" * 58)


def launch_dashboard() -> None:
	app_path = os.path.join(BASE_DIR, "app.py")
	print("Launching dashboard at http://localhost:8501")
	subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], cwd=BASE_DIR, check=False)


def daily_pipeline() -> None:
	print(f"Auto-running pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
	run_pipeline(fetch_live_data=True)


if __name__ == "__main__":
	if len(sys.argv) > 1:
		cmd = sys.argv[1].lower()
		if cmd == "dashboard":
			launch_dashboard()
		elif cmd in ["fetch", "fetch-data"]:
			run_step("src/fetch_real_data.py", "Fetching Real Data from data.gov.in AGMARKNET")
		elif cmd == "schedule":
			print("Scheduler started — pipeline will run daily at 06:00 AM")
			schedule.every().day.at("06:00").do(daily_pipeline)
			while True:
				schedule.run_pending()
				time.sleep(1)
		else:
			print(f"Unknown command: {cmd}")
			print("Available commands: [none] (run pipeline), fetch, dashboard, schedule")
			sys.exit(1)
	else:
		run_pipeline()


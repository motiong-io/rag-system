import subprocess

streamlit_script = "ui/streamlit_app.py"

subprocess.run(["streamlit", "run", streamlit_script])
 
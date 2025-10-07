!pip install streamlit pyngrok --quiet
from pyngrok import ngrok
import getpass, os, time

# Kill old tunnels first
ngrok.kill()

# Token setup
if "NGROK_TOKEN" not in os.environ:
    os.environ["NGROK_TOKEN"] = getpass.getpass("🔑 Enter your ngrok authtoken: ")
ngrok.set_auth_token(os.environ["NGROK_TOKEN"])

# Start Streamlit
!streamlit run app.py &>/dev/null &
time.sleep(8)  # wait for Streamlit to boot

# Open new tunnel
public_url = ngrok.connect(addr="8501", proto="http")
print("🌍 Streamlit app live at:", public_url)

# Control Center

## Configuring the Password

1. Open the `config.toml` file in the ``directionalscalper/controlcenter/`` directory in a text editor:
   ```
   nano config.toml
   ```
2. Add or update the `[credentials]` section with your desired password:
   ```
   [credentials]
   password = "your_password_here"
   ```
3. Save the file and close the editor.

## Starting the Streamlit Dashboard

To start the Streamlit dashboard (Control Center) for Directional Scalper, follow these steps:

1. Navigate to the `directionalscalper/controlcenter/` directory:
   ```
   cd directionalscalper/controlcenter/
   ```
2. Ensure you have Streamlit installed. If not, you can install it using:
   ```
   pip install streamlit
   ```
3. Run the dashboard using Streamlit:
   ```
   streamlit run dashboard.py
   ```
4. The dashboard will prompt for a password. You need to configure this password in the `config.toml` file located in the same directory.

## Accessing the Dashboard

Once the dashboard is running, you can access it through your web browser by navigating to the local URL provided by Streamlit (usually `http://localhost:8501`). Ensure that your firewall or network settings allow access to this port if you are accessing it remotely.

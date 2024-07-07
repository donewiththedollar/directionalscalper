# Installation

## VPS Setup

To set up a VPS (Virtual Private Server) for running the bot, follow these steps:

1. **Choose a VPS Provider**: Select a VPS provider such as [Hetzner](https://hetzner.cloud/?ref=ZfukXC6DHVAZ), DigitalOcean, AWS, or Linode. For optimal performance, it is recommended to use a Hetzner 4 core 8GB Intel server. Create an account and set up a new VPS instance with at least 4 Cores and 8GB of RAM and a suitable operating system (e.g., Ubuntu 20.04).

2. **Connect to Your VPS**: Use SSH to connect to your VPS. Replace `your_vps_ip` with the IP address of your VPS.
   ``ssh root@your_vps_ip``

3. **Update and Upgrade Packages**: Ensure your VPS is up to date.
   ``sudo apt update && sudo apt upgrade -y``

4. **Install Python 3.11**: Follow the instructions in the "Installing Python 3.11" section below to install Python 3.11 on your VPS.

5. **Install Git**: If Git is not already installed, install it using:
   ``sudo apt install git -y``

6. **Clone the Repository**: Clone the Directional Scalper repository to your VPS:
   ``git clone https://github.com/donewiththedollar/directionalscalper.git``

7. **Install Dependencies**: Install the required Python dependencies.
   ``pip3.11 install -r requirements.txt``

8. **Configure the Bot**: Rename the configuration files and add your API keys as described in the "Quickstart" section.

9. **Run the Bot**: Start the bot using the appropriate command as described in the "Quickstart" section.

## Prerequisites

Before you begin, ensure you have the following prerequisites:

- Python 3.11+
- pip
- git

## Quickstart

1. Clone the repository:
   ``git clone https://github.com/donewiththedollar/directionalscalper.git``

    Change current working directory to the project directory:
    ``cd directionalscalper``

2. Install the required dependencies:
   ```
   pip3.11 install -r requirements.txt
   ```

3. Rename the ``config_example.json`` to ``config.json`` located in the ``/configs`` folder. Also rename ``account_example.json`` to ``account.json`` and add your API key(s) to the ``account.json`` file located in the ``/configs`` folder.

4. Run the bot:
   - To display the menu and select a strategy, use the following command:
     ```
     python3.11 multi_bot_signalscreener.py --config configs/config.json
     ```
     or the old method
     ```
     python3.11 multi_bot.py --config config.json (outdated)
     ```
   - Alternatively, you can run the bot with command line parameters:
     - For the multi-bot auto symbol rotator strategy:
       ```
       python3.11 multi_bot_signalscreener_multicore.py --exchange bybit --account_name account_1 --strategy qsgridob --config configs/config.json
       ```
     - For the old single coin strategy (outdated):
       ```
       python3.11 bot.py --exchange bybit --symbol DOGEUSDT --strategy qstrendob --config configs/config.json
       ```

## Installing Python 3.11

To install Python 3.11 on your system, follow these steps:

1. Update the package list and install the necessary dependencies:
   ```
   sudo apt update
   sudo apt install -y build-essential gdb lcov pkg-config \
   libbz2-dev libffi-dev libgdbm-dev libgdbm-compat-dev liblzma-dev \
   libncurses5-dev libreadline-dev libsqlite3-dev libssl-dev \
   lzma lzma-dev tk-dev uuid-dev zlib1g-dev
   ```

2. Download the Python 3.11.0 source code:
   ```
   cd /opt
   wget https://www.python.org/ftp/python/3.11.0/Python-3.11.0.tgz
   ```

3. Extract the downloaded archive:
   ```
   sudo tar -xzf Python-3.11.0.tgz
   ```

4. Change to the extracted directory:
   ```
   cd Python-3.11.0
   ```

5. Configure the build with optimizations:
   ```
   sudo ./configure --enable-optimizations --prefix=/usr/local
   ```

6. Compile Python:
   ```
   sudo make -j 8
   ```

7. Install Python 3.11:
   ```
   sudo make altinstall
   ```

8. Verify the installation:
   ```
   python3.11 --version
   ```

You should see the installed Python version displayed.

That's it! You have now installed Python 3.11 on your system.

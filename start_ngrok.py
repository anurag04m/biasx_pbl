#!/usr/bin/env python3
"""
Ngrok tunnel setup for BiasX PBL Flask backend
Allows GitHub Pages frontend to access local Flask server via HTTPS
"""

import os
import subprocess
import time
import requests
import logging
from pyngrok import ngrok, conf
from pyngrok.exception import PyngrokNgrokHTTPError, PyngrokError

# Configure logging for pyngrok
pyngrok_logger = logging.getLogger("pyngrok")
pyngrok_logger.setLevel(logging.DEBUG)

if not any(isinstance(h, logging.StreamHandler) for h in pyngrok_logger.handlers):
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    pyngrok_logger.addHandler(stream_handler)

# Your ngrok auth token
NGROK_AUTH_TOKEN = "2xqv1sHmp2ouMeqJ4WEXkz18POi_4v9UHWzs7WMVgsMc84wnq"

def check_flask_health(max_retries=6, retry_delay=5):
    """Check if Flask server is running and healthy on localhost:5000"""
    flask_local_health_url = "http://127.0.0.1:5000/metrics"  # Using /metrics as health check

    print("⏳ Checking if Flask server is running on localhost:5000...")

    for i in range(max_retries):
        try:
            response = requests.get(flask_local_health_url, timeout=5)
            if response.status_code == 200:
                print(f"✅ Flask is up and healthy locally: {response.status_code}")
                return True
            else:
                print(f"Attempt {i+1}/{max_retries}: Flask returned status {response.status_code}. Retrying in {retry_delay}s...")
        except requests.exceptions.ConnectionError:
            print(f"Attempt {i+1}/{max_retries}: Flask not responding. Retrying in {retry_delay}s...")
        except Exception as e:
            print(f"Attempt {i+1}/{max_retries}: Error checking Flask: {e}. Retrying in {retry_delay}s...")

        if i < max_retries - 1:  # Don't sleep on last attempt
            time.sleep(retry_delay)

    print("❌ Flask server is not responding on localhost:5000")
    print("   Make sure you've started Flask with: python flask_app.py")
    return False

def start_ngrok_tunnel(port=5000, use_https=True):
    """Start ngrok tunnel with enhanced debugging and error handling"""

    # 1. Check if Flask is running
    if not check_flask_health():
        return None

    # 2. Kill any existing ngrok processes
    print("🧹 Killing any existing ngrok processes...")
    try:
        ngrok.kill()
        time.sleep(2)
    except Exception as e:
        print(f"Note: Error killing ngrok (might not be running): {e}")

    # 3. Configure and start ngrok tunnel
    print("🌐 Configuring and starting ngrok tunnel...")
    public_url = None

    try:
        # Set auth token
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)

        # Determine protocol (https for GitHub Pages, http for local testing)
        proto = "http"
        bind_tls = True if use_https else False

        print(f"Attempting to connect ngrok tunnel to port {port} with bind_tls={bind_tls}")

        # Start the tunnel
        tunnel = ngrok.connect(
            addr=port,
            proto=proto,
            bind_tls=bind_tls,  # True = HTTPS only, False = HTTP and HTTPS
            host_header=f"localhost:{port}"
        )

        print(f"DEBUG: Tunnel object created: Name='{tunnel.name}'")
        print(f"DEBUG: Attempting to access tunnel.public_url...")

        public_url = tunnel.public_url

        print(f"✅ NGROK URL obtained: {public_url}")
        print(f"   Tunnel details: Name={tunnel.name}, Proto={tunnel.proto}")

        # Verify with get_tunnels()
        print("DEBUG: Verifying with ngrok.get_tunnels()...")
        active_tunnels = ngrok.get_tunnels()
        print(f"DEBUG: Active tunnels: {[t.public_url for t in active_tunnels]}")

        if not any(t.public_url == public_url for t in active_tunnels):
            print(f"❌ Tunnel with URL {public_url} not found in active tunnels")
            ngrok.kill()
            return None

        # 4. Health check via ngrok
        health_url_ngrok = f"{public_url}/metrics"
        success = False
        print(f"🩺 Performing health check via ngrok: {health_url_ngrok}")

        for i in range(3):
            try:
                response = requests.get(
                    health_url_ngrok,
                    headers={'ngrok-skip-browser-warning': 'true'},
                    timeout=20
                )
                if response.status_code == 200:
                    print(f"✅ Health check via ngrok passed! Status: {response.status_code}")
                    success = True
                    break
                else:
                    print(f"⚠️ Health check failed (Attempt {i+1}/3): Status {response.status_code}")
            except Exception as e_health:
                print(f"⚠️ Health check error (Attempt {i+1}/3): {str(e_health)}")

            if not success and i < 2:
                time.sleep(5)

        if not success:
            print("❌ Health check through ngrok failed. The tunnel might be unstable.")
            ngrok.kill()
            return None

        return public_url

    except PyngrokNgrokHTTPError as e_http:
        print(f"❌ Ngrok HTTP Error: {str(e_http)}")
        print(f"    Status Code: {getattr(e_http, 'status_code', 'N/A')}")
        print(f"    Error Message: {getattr(e_http, 'error', 'N/A')}")
        if hasattr(e_http, 'status_code') and e_http.status_code == 404:
            print("    404 'tunnel not found' - The ngrok agent failed to create the tunnel.")
            print("    This can happen if the ngrok service is down or your account has issues.")
        ngrok.kill()
        return None

    except PyngrokError as e_pyngrok:
        print(f"❌ Pyngrok Error: {str(e_pyngrok)}")
        ngrok.kill()
        return None

    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        ngrok.kill()
        return None

def save_url_for_frontend(url):
    """Save ngrok URL to a file and provide instructions"""
    # Save to a text file
    with open('ngrok_url.txt', 'w') as f:
        f.write(url)
    print(f"\n📄 Ngrok URL saved to: {os.path.abspath('ngrok_url.txt')}")

    # Print instructions for frontend
    print("\n" + "="*80)
    print("🎯 FRONTEND SETUP INSTRUCTIONS")
    print("="*80)
    print("\nFor GitHub Pages (https://shubhk2.github.io/biasx_pbl/frontend/):")
    print(f"   Open: https://shubhk2.github.io/biasx_pbl/frontend/?api={url}")
    print(f"   Or manually click 'Change API URL' and enter: {url}")

    print("\nFor Local Testing (file:///...):")
    print(f"   Open: file:///home/pokemon/PycharmProjects/biasx_pbl/frontend/index.html?api={url}")
    print(f"   Or it will auto-use: http://127.0.0.1:5000")

    print("\n⚠️  IMPORTANT:")
    print("   - The ngrok URL will change each time you restart this script")
    print("   - Keep this terminal window open while using the frontend")
    print("   - Press Ctrl+C to stop the tunnel when done")
    print("="*80 + "\n")

def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("BiasX PBL - Ngrok Tunnel Setup")
    print("="*80 + "\n")

    # Check if Flask is already running
    print("Step 1: Checking Flask server...")
    if not check_flask_health(max_retries=2, retry_delay=2):
        print("\n❌ Flask server is not running!")
        print("\nPlease start Flask in another terminal first:")
        print("   cd /home/pokemon/PycharmProjects/biasx_pbl")
        print("   python flask_app.py")
        print("\nThen run this script again.")
        return

    # Start ngrok tunnel
    print("\nStep 2: Starting ngrok tunnel...")
    public_url = start_ngrok_tunnel(port=5000, use_https=True)

    if public_url:
        print(f"\n🎉 SUCCESS! Ngrok tunnel is running!")
        save_url_for_frontend(public_url)

        # Keep the tunnel alive
        print("🔄 Tunnel is active. Press Ctrl+C to stop...\n")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down ngrok tunnel...")
            ngrok.kill()
            print("✅ Tunnel closed. Goodbye!")
    else:
        print("\n❌ Failed to start ngrok tunnel. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Ensure Flask is running: python flask_app.py")
        print("2. Check if port 5000 is not blocked by firewall")
        print("3. Verify ngrok auth token is valid")
        print("4. Try running: pip install pyngrok --upgrade")

if __name__ == "__main__":
    main()

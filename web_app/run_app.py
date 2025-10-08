#!/usr/bin/env python3
"""
Launcher script for the NovaBert Recommendation System web application.
This script handles environment setup and launches the Streamlit app.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all required files exist"""
    base_path = Path("/Users/baonguyen/IU/thesis")
    
    required_files = [
        "src/models/models_item_with_novabert_gatingfusor_/fold_1/best_model.pth",
        "data/clean_data/data_with_bertopic_column.csv",
        "data/topic_info.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = base_path / file_path
        if not full_path.exists():
            missing_files.append(str(full_path))
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ All required files found!")
    return True

def check_dependencies():
    """Check if required Python packages are installed"""
    required_packages = [
        "streamlit",
        "torch", 
        "pandas",
        "numpy",
        "plotly",
        "scikit-learn"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall them with: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages installed!")
    return True

def main():
    """Main launcher function"""
    print("🚀 Starting NovaBert Recommendation System...")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Please ensure all required files are in place before running the app.")
        sys.exit(1)
    
    if not check_dependencies():
        print("\n❌ Please install missing dependencies before running the app.")
        sys.exit(1)
    
    print("\n🎯 Launching Streamlit application...")
    print("=" * 50)
    
    # Change to the web_app directory
    app_dir = Path(__file__).parent
    os.chdir(app_dir)
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Setup script for the NovaBert Recommendation System web application.
This script handles installation, dependency checking, and initial setup.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header():
    """Print setup header"""
    print("=" * 60)
    print("🤖 NovaBert Recommendation System - Setup")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported.")
        print("   Please use Python 3.8 or higher.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible.")
    return True

def check_system_requirements():
    """Check system requirements"""
    print("\n💻 Checking system requirements...")
    
    # Check available memory (rough estimate)
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 4:
            print(f"⚠️  Warning: Only {memory_gb:.1f}GB RAM available. 4GB+ recommended.")
        else:
            print(f"✅ Available memory: {memory_gb:.1f}GB")
    except ImportError:
        print("ℹ️  psutil not available, skipping memory check")
    
    # Check if we're on macOS (for MPS support)
    if platform.system() == "Darwin":
        print("✅ macOS detected - MPS acceleration available")
    elif platform.system() == "Linux":
        print("✅ Linux detected - CUDA acceleration may be available")
    else:
        print("ℹ️  Windows detected - CPU/GPU acceleration depends on setup")
    
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        # Upgrade pip first
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        
        # Install requirements
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        
        print("✅ Dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        print("   Please install manually: pip install -r requirements.txt")
        return False

def check_data_files():
    """Check if required data files exist"""
    print("\n📁 Checking data files...")
    
    base_path = Path("/Users/baonguyen/IU/thesis")
    required_files = [
        "src/models/models_item_with_novabert_gatingfusor_/fold_1/best_model.pth",
        "data/clean_data/data_with_bertopic_column.csv",
        "data/topic_info.csv"
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  Some required files are missing.")
        print("   Please ensure you have:")
        print("   1. Trained model weights")
        print("   2. Processed dataset")
        print("   3. Topic information file")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    print("\n📂 Creating directories...")
    
    directories = [
        "logs",
        "cache",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ {directory}/")
    
    return True

def test_imports():
    """Test if all required modules can be imported"""
    print("\n🧪 Testing imports...")
    
    required_modules = [
        "streamlit",
        "torch",
        "pandas",
        "numpy",
        "plotly",
        "sklearn",
        "fastapi",
        "uvicorn",
        "pydantic"
    ]
    
    failed_imports = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️  Failed to import: {', '.join(failed_imports)}")
        print("   Please install missing packages manually.")
        return False
    
    return True

def create_launcher_scripts():
    """Create convenient launcher scripts"""
    print("\n🚀 Creating launcher scripts...")
    
    # Streamlit launcher
    streamlit_script = """#!/bin/bash
cd "$(dirname "$0")"
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
"""
    
    with open("run_streamlit.sh", "w") as f:
        f.write(streamlit_script)
    
    os.chmod("run_streamlit.sh", 0o755)
    print("✅ run_streamlit.sh")
    
    # API launcher
    api_script = """#!/bin/bash
cd "$(dirname "$0")"
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""
    
    with open("run_api.sh", "w") as f:
        f.write(api_script)
    
    os.chmod("run_api.sh", 0o755)
    print("✅ run_api.sh")
    
    return True

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 60)
    print("🎉 Setup Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print()
    print("1. 🌐 Start the Streamlit web app:")
    print("   ./run_streamlit.sh")
    print("   or")
    print("   python run_app.py")
    print()
    print("2. 🔌 Start the REST API (optional):")
    print("   ./run_api.sh")
    print("   or")
    print("   python api.py")
    print()
    print("3. 📖 View the API documentation:")
    print("   http://localhost:8000/docs")
    print()
    print("4. 🌍 Access the web interface:")
    print("   http://localhost:8501")
    print()
    print("For more information, see README.md")
    print()

def main():
    """Main setup function"""
    print_header()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check system requirements
    if not check_system_requirements():
        print("⚠️  System requirements check failed, but continuing...")
    
    # Check data files
    if not check_data_files():
        print("⚠️  Some data files are missing. Please check the file paths.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("⚠️  Dependency installation failed. Please install manually.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Test imports
    if not test_imports():
        print("⚠️  Some imports failed. Please check your installation.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Create launcher scripts
    create_launcher_scripts()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()


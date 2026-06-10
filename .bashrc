# Auto-activate virtual environment when opening Git Bash in this directory
if [ -f ".venv/Scripts/activate" ]; then
    echo "Activating virtual environment..."
    source .venv/Scripts/activate
    echo "Virtual environment activated!"
else
    echo "Warning: Virtual environment not found at .venv/Scripts/activate"
fi

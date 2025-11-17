packages=(torch transformers pandas scikit-learn)
for item in "${packages[@]}"; do
    if pip show $item &> /dev/null; then
        echo "Skipping install of $item"
    else
        echo "Installing $item..."
        pip install $item &> /dev/null
        if [ $? -eq 0 ]; then
            echo "Completed install of $item"
        else
            echo "Failed to install $item"
        fi 
    fi
done 

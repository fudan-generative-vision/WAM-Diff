conda env create -f envs.yml
conda activate WAM-Diff

#!/bin/bash

# log file for failed installations
FAIL_LOG="failed_packages.log"

# rm existing log file
> $FAIL_LOG

while IFS= read -r line; do
    # skip empty lines and comments
    if [[ -z "$line" || "$line" =~ ^# ]]; then
        continue
    fi

    echo "====================================="
    echo "Installing: $line"
    echo "====================================="

    # Execute installation command
    pip install $line $PIP_MIRROR

    # Check if installation was successful
    if [ $? -ne 0 ]; then
        echo "Installation failed: $line" >> $FAIL_LOG
        echo "====================================="
        echo "⚠️  $line installation failed, logged to $FAIL_LOG"
        echo "====================================="
    else
        echo "====================================="
        echo "✅ $line installed successfully"
        echo "====================================="
    fi
    sleep 1

done < "requirements.txt"

echo "====================================="
echo "Installation complete!"
if [ -s $FAIL_LOG ]; then
    echo "❌ The following packages failed to install:"
    cat $FAIL_LOG
else
    echo "✅ All packages installed successfully!"
fi
echo "====================================="
#!/bin/bash

set -e

# Setup the environment if it doesn't exist
poetry install --no-root

# Generate the requirements.txt file
poetry export --without-hashes --format=requirements.txt > requirements.txt

# Remove Windows and Darwin requirements
sed -i '/platform_system == "Darwin"/d; /platform_system == "Windows"/d' requirements.txt

# Create the Project dependencies
req2flatpak --requirements-file=requirements.txt --outfile python-deps.json --target-platforms '312-x86_64' '312-aarch64'

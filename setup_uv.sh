uv pip install -e bddl
uv pip install -e OmniGibson

export OMNI_KIT_ACCEPT_EULA=YES
install_isaac_packages() {
    local temp_dir=$(mktemp -d)
    local packages=(
        "omniverse_kit-106.5.0.162521" "isaacsim_kernel-4.5.0.0" "isaacsim_app-4.5.0.0"
        "isaacsim_core-4.5.0.0" "isaacsim_gui-4.5.0.0" "isaacsim_utils-4.5.0.0"
        "isaacsim_storage-4.5.0.0" "isaacsim_asset-4.5.0.0" "isaacsim_sensor-4.5.0.0"
        "isaacsim_robot_motion-4.5.0.0" "isaacsim_robot-4.5.0.0" "isaacsim_benchmark-4.5.0.0"
        "isaacsim_code_editor-4.5.0.0" "isaacsim_ros1-4.5.0.0" "isaacsim_cortex-4.5.0.0"
        "isaacsim_example-4.5.0.0" "isaacsim_replicator-4.5.0.0" "isaacsim_rl-4.5.0.0"
        "isaacsim_robot_setup-4.5.0.0" "isaacsim_ros2-4.5.0.0" "isaacsim_template-4.5.0.0"
        "isaacsim_test-4.5.0.0" "isaacsim-4.5.0.0" "isaacsim_extscache_physics-4.5.0.0"
        "isaacsim_extscache_kit-4.5.0.0" "isaacsim_extscache_kit_sdk-4.5.0.0"
    )
    
    local wheel_files=()
    for pkg in "${packages[@]}"; do
        local pkg_name=${pkg%-*}
        local filename="${pkg}-cp310-none-manylinux_2_34_x86_64.whl"
        local url="https://pypi.nvidia.com/${pkg_name//_/-}/$filename"
        local filepath="$temp_dir/$filename"
        
        echo "Downloading $pkg..."
        if ! curl -sL "$url" -o "$filepath"; then
            echo "ERROR: Failed to download $pkg"
            rm -rf "$temp_dir"
            return 1
        fi
        
        wheel_files+=("$filepath")
    done
    
    echo "Installing Isaac Sim packages..."
    uv pip install "${wheel_files[@]}"
    rm -rf "$temp_dir"
    
    # Verify installation
    if ! python -c "import isaacsim" 2>/dev/null; then
        echo "ERROR: Isaac Sim installation verification failed"
        return 1
    fi
}

install_isaac_packages || { echo "ERROR: Isaac Sim installation failed"; exit 1; }

echo "Downloading OmniGibson robot assets..."
python -c "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()" || {
    echo "ERROR: OmniGibson robot assets installation failed"
    exit 1
}

echo "Downloading BEHAVIOR-1K assets..."
python -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=True)" || {
    echo "ERROR: Dataset installation failed"
    exit 1
}

echo "Downloading 2025 BEHAVIOR Challenge Task Instances..."
python -c "from omnigibson.utils.asset_utils import download_2025_challenge_task_instances; download_2025_challenge_task_instances()" || {
    echo "ERROR: 2025 BEHAVIOR Challenge Task Instances installation failed"
    exit 1
}

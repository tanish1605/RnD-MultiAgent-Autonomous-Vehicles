import depthai as dai
import numpy as np
import yaml

def format_np_matrix(matrix, var_name=None):
    """Formats a nested list/matrix into a clean np.array string."""
    np_arr = np.array(matrix)
    rows_str = ",\n    ".join([str(list(row)) for row in np_arr])
    formatted = f"np.array([\n    {rows_str}\n], dtype=np.float64)\n"
    return f"{var_name} = {formatted}" if var_name else formatted

def format_np_array(vector, var_name=None):
    """Formats a 1D vector into a clean np.array string."""
    formatted = f"np.array({list(vector)}, dtype=np.float64)\n"
    return f"{var_name} = {formatted}" if var_name else formatted

def main():
    print("Connecting to OAK device and reading EEPROM...")
    with dai.Device() as device:
        calibData = device.readCalibration()
        print("Calibration read successfully!\n")
        
        # Explicit mapping according to Luxonis OAK-D S2 standards
        camera_mapping = {
            "CAM_A_COLOR_RGB": dai.CameraBoardSocket.CAM_A,
            "CAM_B_LEFT_MONO": dai.CameraBoardSocket.CAM_B,
            "CAM_C_RIGHT_MONO": dai.CameraBoardSocket.CAM_C
        }
        
        yaml_data = {}
        
        # 1. Process Intrinsics and Distortion for each camera
        for cam_name, socket in camera_mapping.items():
            try:
                intrinsics = calibData.getCameraIntrinsics(socket)
                distortion = calibData.getDistortionCoefficients(socket)
                
                yaml_data[cam_name] = {
                    "intrinsics": format_np_matrix(intrinsics),
                    "distortion_coefficients": format_np_array(distortion)
                }
            except Exception as e:
                print(f"Could not retrieve data for {cam_name}: {e}")
        
        # 2. Process Extrinsics
        extrinsics_config = {
            "left_mono_to_color_rgb": (dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_A),
            "right_mono_to_color_rgb": (dai.CameraBoardSocket.CAM_C, dai.CameraBoardSocket.CAM_A),
            "left_mono_to_right_mono": (dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C)
        }
        
        yaml_data["extrinsics_spatial_relationships"] = {}
        
        for ext_name, (from_socket, to_socket) in extrinsics_config.items():
            try:
                ext_matrix = calibData.getCameraExtrinsics(from_socket, to_socket)
                yaml_data["extrinsics_spatial_relationships"][ext_name] = format_np_matrix(ext_matrix)
            except Exception as e:
                print(f"Could not retrieve extrinsics for {ext_name}: {e}")
                
        # 3. Save to YAML configuration with literal block mapping
        yaml_filename = "oak_d_s2_explicit_calibration.yaml"
        
        def str_presenter(dumper, data):
            if '\n' in data:
                return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
            return dumper.represent_scalar('tag:yaml.org,2002:str', data)
        
        yaml.add_representer(str, str_presenter)
        
        with open(yaml_filename, "w") as yaml_file:
            # Inject descriptive header comments manually into the text stream
            yaml_file.write(
                "# ==================================================================\n"
                "# OAK-D S2 CAMERA SYSTEM CALIBRATION FILE\n"
                "# \n"
                "# CAMERA MAPPING IDENTIFICATION:\n"
                "#   - CAM_A_COLOR_RGB: Central high-resolution color camera\n"
                "#   - CAM_B_LEFT_MONO: Left global shutter grayscale camera (Stereo pairs)\n"
                "#   - CAM_C_RIGHT_MONO: Right global shutter grayscale camera (Stereo pairs)\n"
                "# ==================================================================\n\n"
            )
            yaml.dump(yaml_data, yaml_file, default_flow_style=False, sort_keys=False)
            
        print(f"[Success] Explicitly formatted data saved to: {yaml_filename}")

if __name__ == "__main__":
    main()
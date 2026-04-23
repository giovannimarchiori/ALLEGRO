# retrieve detector geometry numbers for calculation of noise

# geometry info is retrieved from log of simulation while segmentation info is from the xml file
# in principle everything could be extracted from the xml file but its complicated because there are formulae
# and units that should be properly parsed and evaluated..

# Initialize an empty list to store the extracted numbers
extracted_numbers = {}

sim_log_path = 'log/dryRunForNoise.log'
import os
k4geopath = os.environ.get("K4GEO","")
ecalbarrel_xml_path = k4geopath + '/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ECalBarrel_thetamodulemerged.xml'
capacitance_script_path = "create_capacitance_file_theta.py"

parameters_to_read = ["ecal_active_rmin_mm",
                      "ecal_active_thickness_mm",
                      "absorber_thickness_mm",
                      "pcb_thickness_mm",
                      "grid_size_theta",
                      "offset_theta",
                      "strip_layer",
                      "theta_merging",
                      "module_merging",
                      "electrode_inclination_deg",
                      "Nplanes",
                      "readoutLayerParallelLengths",
                      "total_electrode_length_mm"]
                      
# Detector geometry: from the log file of the simulation
print("Reading geometry parameters from", sim_log_path)
import re
with open(sim_log_path, 'r') as file:
    for line in file:
        if not "createECalBarrelInclined" in line:
            continue

        # Use regex to match the numbers in each relevant line
        if "ECAL calorimeter volume rmin (cm)" in line:
            match = re.search(r'rmin \(cm\) =\s+([\d.]+)', line)
            if match:
                extracted_numbers["ecal_active_rmin_mm"] = float(match.group(1))*10.0
        
        if "ECAL thickness of calorimeter (cm)" in line:
            match = re.search(r'thickness of calorimeter \(cm\) =\s+([\d.]+)', line)
            if match:
                extracted_numbers["ecal_active_thickness_mm"] = float(match.group(1)) * 10.0
        
        if "total thickness of absorber (cm)" in line:
            match = re.search(r'total thickness of absorber \(cm\) =\s+([\d.]+)', line)
            if match:
                extracted_numbers["absorber_thickness_mm"] = float(match.group(1)) * 10.0
        
#        if "rotation angle (radians)" in line:
#            match = re.search(r'\(degrees\) =\s+([\d.]+)', line)
#            if match:
#                extracted_numbers["electrode_inclination_deg"] = float(match.group(1))
        
#        if "number of planes (calculated)" in line:
#            match = re.search(r'number of planes \(calculated\) =\s+(\d+)', line)
#            if match:
#                extracted_numbers["num_absorbers"] = int(match.group(1))

        if "thickness of readout planes" in line:
            match = re.search(r'thickness of readout planes \(cm\) =\s+([\d.]+)', line)
            if match:
                extracted_numbers["pcb_thickness_mm"] = float(match.group(1))*10.0


# Detector segmentation: from xml file
print("Reading segmentation parameters from", ecalbarrel_xml_path)
import xml.etree.ElementTree as ET
tree = ET.parse(ecalbarrel_xml_path)
root = tree.getroot()

# Find the <readout> element with name="ECalBarrelModuleThetaMerged"
merged_modules = []
merged_cells_theta = []
for readout in root.findall('.//readout'):
    if readout.get('name') == 'ECalBarrelModuleThetaMerged':
        segmentation = readout.find('segmentation')
        if segmentation is not None:
            # Extract mergedCells_Theta as a list of integers
            merged_cells_theta = list(map(int, segmentation.get('mergedCells_Theta').split()))
            
            # Extract mergedModules as a list of integers
            merged_modules = list(map(int, segmentation.get('mergedModules').split()))
            
            # Extract grid_size_theta and offset_theta as floating-point numbers
            extracted_numbers["grid_size_theta"] = eval(segmentation.get('grid_size_theta'))  # safely evaluates "0.009817477/4"
            extracted_numbers["offset_theta"] = float(segmentation.get('offset_theta'))

# Find the position of the element equal to 1
extracted_numbers["strip_layer"] = merged_cells_theta.index(1)

# Create a new list excluding the element equal to 1
other_elements = [x for i, x in enumerate(merged_cells_theta) if i != extracted_numbers["strip_layer"]]

# Check if all other elements are equal to each other
all_elements_equal = len(set(other_elements)) == 1
assert(all_elements_equal)

# Get the common value of the other elements if they are equal
extracted_numbers["theta_merging"] = other_elements[0] if all_elements_equal else None

all_elements_equal = len(set(merged_modules)) == 1
assert(all_elements_equal)

extracted_numbers["module_merging"] = merged_modules[0] if all_elements_equal else None


# Extract inclination angle and number of planes
for constant in root.findall(".//constant"):
    if constant.get('name') == 'InclinationAngle':
        value_attr = constant.get('value')
        # Use a regular expression to extract the number before "*degree"
        match = re.match(r'([\d.]+)\*degree', value_attr)
        if match:
            extracted_numbers["electrode_inclination_deg"] = float(match.group(1))
    if constant.get('name') == 'ECalBarrelNumPlanes':
        extracted_numbers["Nplanes"] = int(constant.get('value'))


# Extract electrode length vs layer
# List to store the final thicknesses
thicknesses = []
#for layer in root.findall(".//layer"):
#    if layer.tag != 'layer' or layer.getparent().tag != 'layers':
#        continue
layers = root.findall(".//layers")
assert(len(layers)==1)
layers = layers[0].findall(".//layer")
for layer in layers:
    thickness_str = layer.get('thickness')
    repeat = int(layer.get('repeat', 1))  # Get repeat attribute, default to 1 if missing
    
    # Extract the numerical part and the unit (*cm or *mm)
    if '*cm' in thickness_str:
        thickness_value = float(thickness_str.replace('*cm', '')) * 10  # Convert cm to mm
    elif '*mm' in thickness_str:
        thickness_value = float(thickness_str.replace('*mm', ''))  # Already in mm
        
    # Add the thickness repeated by the number of times specified in "repeat"
    thicknesses.extend([thickness_value] * repeat)
        
    # Output the list of thicknesses
    extracted_numbers["readoutLayerParallelLengths"] = thicknesses
extracted_numbers["total_electrode_length_mm"] = sum(extracted_numbers["readoutLayerParallelLengths"])


# Output the extracted values
print("Extracted numbers:", extracted_numbers)

exit(0)
# Update the create_capacitance_file_theta.py file

# Function to format numbers with four decimal places
def format_number(num):
    return f"{num:.4f}" if isinstance(num, float) else str(num)

# Read the content of the file
with open(capacitance_script_path, 'r') as file:
    lines = file.readlines()

# Replace the numbers in the file with those from the dictionary
for i in range(len(lines)):
    for key, value in extracted_numbers.items():
        # Convert list to string representation if it's a list
        if isinstance(value, list):
            value = "[" + ", ".join(format_number(v) for v in value) + "]"
        else:
            value = str(value)  # Ensure the value is a string for replacement
            
        # Use regular expression to find and replace the key and its value
        pattern = rf'^(?P<key>{key})\s*=\s*(?P<old_value>.*?)(#.*)?$'
        match = re.match(pattern, lines[i])
        if match:
            comment = match.group(3) if match.group(3) else ""  # Preserve the comment if it exists
            lines[i] = f"{key} = {value}  {comment}\n"  # Create a new line with the updated value and comment
            
#       if re.match(pattern,lines[i]) and lines[i][0]!="#":
#           lines[i] = f"{key} = {value}\n"  # Create a new line with the updated value

if not all(item in extracted_numbers for item in parameters_to_read):
    print("ERROR: not all parameters found in input files")
    exit(-1)
else:
    print("SUCCESS: all parameters found in input files")
    
# Write the modified content to a new file (or overwrite the original)
output_file_path = capacitance_script_path
with open(output_file_path, 'w') as file:
    file.writelines(lines)
                        
print("File "+output_file_path+" updated")
                        

import sys
import os

def main():
    if len(sys.argv) != 5:
        print("Usage: python embed.py <input_file> <output_file> <namespace> <variable_name>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    namespace = sys.argv[3]
    variable_name = sys.argv[4]

    try:
        with open(input_file, 'rb') as f:
            content = f.read()
    except IOError as e:
        print(f"Error reading input file {input_file}: {e}")
        sys.exit(1)

    # Convert the binary data to a list of hexadecimal strings
    hex_values = [f"0x{byte:02x}" for byte in content]
    
    # Add a null terminator for safety, allowing the data to be used as a C-string
    hex_values.append("0x00")

    # Format the hex values into readable lines of 16 bytes each
    formatted_hex = ""
    for i in range(0, len(hex_values), 16):
        line = ", ".join(hex_values[i:i+16])
        formatted_hex += f"    {line},\n"

    # Create the new C++ header content using the char array format
    header_content = f"""
#pragma once
#include <cstddef>

namespace {namespace} {{
    // Use an unsigned char array to store the raw shader data.
    // This format has no compiler-imposed size limit.
    constexpr unsigned char {variable_name}[] = {{
{formatted_hex.strip().rstrip(',')}
    }};

    // The length is the size of the array minus our added null terminator.
    constexpr size_t {variable_name}_len = sizeof({variable_name}) - 1;
}}
"""

    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(header_content)
    except IOError as e:
        print(f"Error writing output file {output_file}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
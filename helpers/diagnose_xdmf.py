#!/usr/bin/env python3
# diagnose_xdmf.py
# Diagnose XDMF file structure and content

import os
import h5py
import xml.etree.ElementTree as ET

def diagnose_xdmf_files():
    """Diagnose XDMF and HDF5 files for potential issues"""
    
    print("=== XDMF FILE DIAGNOSTICS ===\n")
    
    # Check file existence
    xdmf_file = "poisson_solution.xdmf"
    h5_file = "poisson_solution.h5"
    
    print("1. FILE EXISTENCE CHECK:")
    for fname in [xdmf_file, h5_file]:
        if os.path.exists(fname):
            size = os.path.getsize(fname)
            print(f"   ✓ {fname}: {size} bytes")
        else:
            print(f"   ✗ {fname}: MISSING")
            return
    
    # Analyze XDMF structure
    print("\n2. XDMF FILE STRUCTURE:")
    try:
        with open(xdmf_file, 'r') as f:
            content = f.read()
        print(f"   Content preview (first 500 chars):")
        print(f"   {'-'*50}")
        print(f"   {content[:500]}")
        print(f"   {'-'*50}")
        
        # Parse XML
        tree = ET.parse(xdmf_file)
        root = tree.getroot()
        print(f"\n   XML Root tag: {root.tag}")
        print(f"   XML attributes: {root.attrib}")
        
        # Find key elements
        domains = root.findall('.//Domain')
        grids = root.findall('.//Grid')
        geometries = root.findall('.//Geometry')
        topologies = root.findall('.//Topology')
        attributes = root.findall('.//Attribute')
        
        print(f"   Found elements:")
        print(f"     - Domains: {len(domains)}")
        print(f"     - Grids: {len(grids)}")
        print(f"     - Geometries: {len(geometries)}")
        print(f"     - Topologies: {len(topologies)}")
        print(f"     - Attributes: {len(attributes)}")
        
        # Check DataItem references to HDF5
        data_items = root.findall('.//DataItem')
        print(f"     - DataItems: {len(data_items)}")
        
        for i, item in enumerate(data_items):
            format_attr = item.get('Format', 'Unknown')
            dimensions = item.get('Dimensions', 'Unknown')
            print(f"       DataItem {i}: Format={format_attr}, Dimensions={dimensions}")
            if item.text:
                print(f"         Content: {item.text.strip()}")
        
    except Exception as e:
        print(f"   ✗ Error parsing XDMF: {e}")
        return
    
    # Analyze HDF5 structure
    print("\n3. HDF5 FILE STRUCTURE:")
    try:
        with h5py.File(h5_file, 'r') as f:
            print(f"   HDF5 file opened successfully")
            
            def print_structure(name, obj):
                indent = "     " + "  " * name.count('/')
                if isinstance(obj, h5py.Dataset):
                    print(f"{indent}Dataset: {name}")
                    print(f"{indent}  Shape: {obj.shape}")
                    print(f"{indent}  Dtype: {obj.dtype}")
                    print(f"{indent}  Size: {obj.size} elements")
                    
                    # Show some sample data for small datasets
                    if obj.size < 20:
                        print(f"{indent}  Data: {obj[...]}")
                    elif len(obj.shape) == 1 and obj.shape[0] < 10:
                        print(f"{indent}  Data preview: {obj[...]}")
                    elif len(obj.shape) == 2 and obj.shape[0] * obj.shape[1] < 50:
                        print(f"{indent}  Data preview: {obj[...]}")
                    else:
                        print(f"{indent}  Data preview: {obj.flat[:5]}... (showing first 5)")
                        
                elif isinstance(obj, h5py.Group):
                    print(f"{indent}Group: {name}")
            
            print(f"   Contents:")
            f.visititems(print_structure)
            
    except Exception as e:
        print(f"   ✗ Error reading HDF5: {e}")
        return
    
    # Check for common issues
    print("\n4. POTENTIAL ISSUES CHECK:")
    issues_found = []
    
    # Check file sizes
    xdmf_size = os.path.getsize(xdmf_file)
    h5_size = os.path.getsize(h5_file)
    
    if xdmf_size < 100:
        issues_found.append("XDMF file is very small (< 100 bytes)")
    if h5_size < 100:
        issues_found.append("HDF5 file is very small (< 100 bytes)")
    
    # Check for empty datasets
    try:
        with h5py.File(h5_file, 'r') as f:
            for name, obj in f.items():
                if isinstance(obj, h5py.Dataset) and obj.size == 0:
                    issues_found.append(f"Empty dataset found: {name}")
    except:
        pass
    
    if issues_found:
        print("   Issues found:")
        for issue in issues_found:
            print(f"     ⚠️  {issue}")
    else:
        print("   ✓ No obvious issues detected")
    
    print("\n5. RECOMMENDATIONS:")
    print("   Try these alternative approaches:")
    print("   a) Use Python visualization instead of ParaView")
    print("   b) Export to a different format (VTK, VTU)")
    print("   c) Try opening with a different XDMF reader")
    print("   d) Check ParaView version compatibility")

if __name__ == "__main__":
    diagnose_xdmf_files()
